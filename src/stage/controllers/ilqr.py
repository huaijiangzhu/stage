import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from tqdm import trange
from dotmap import DotMap

from stage.controllers.base import Controller
from stage.utils.nn import renew

class SVDError(Exception):
    pass

class ILQR(nn.Module):

    def __init__(self, dynamics, cost, actor, horizon,
                 alpha=1.0, decay=0.05):
        super().__init__()
        self.dynamics = dynamics
        self.nx = dynamics.nx
        self.actor = actor
        self.na = actor.na
        self.action_ub, self.action_lb = actor.action_ub, actor.action_lb
        self.cost = cost
        self.cost.actor = actor
        self.horizon = horizon

        self.mu = 1.0
        self.mu_min = 1e-6
        self.mu_max = 1e10
        self.delta0 = 2.0
        self.delta = self.delta0
        self.eps = 1e-8

    def optimize(self, x, actions=None, horizon=None, max_it=10, on_iteration=None):
        self.mu = 1.0
        self.delta = self.delta0
        exponent = -torch.arange(10)**2 * torch.log(1.1 * torch.ones(10))
        alphas = torch.exp(exponent)

        changed = True
        converged = False
        for it in range(max_it):
            accepted = False
            if changed:
                rollout = self.unroll(x, actions, horizon)
                changed = False
            try:
                sols = self.backward_pass(rollout)
                for alpha in alphas:
                    accepted = self.line_search(rollout, sols, alpha)
            except SVDError as e:
                # Qaa was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                # Increase regularization term.
                self.delta = max(1.0, self.delta) * self._delta0
                self.mu = max(self.mu_min, self.mu * self.delta)
                if self.mu_max and self.mu >= self.mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(it, sols, J_opt, accepted, converged)

            if converged:
                break

        return sols

                
    def line_search(self, rollout, sols, alpha):
        J_opt = self.compute_cost(rollout)

    def forward_pass(self, sols, alpha=1.0):
        pass

    def unroll(self, x, actions=None, horizon=None):

        rollout = []

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        if horizon is None:
            horizon = self.horizon

        if actions is None:
            actions = self.actor.sample(horizon)

        for t in range(horizon):

            if t == horizon - 1:
                terminal = True
            else:
                terminal = False

            a = actions[t]

            cost = self.cost.l(renew(x), renew(a), t, terminal, diff=True)
            prediction = self.dynamics.sample_predictions(renew(x), renew(a), n_particles=0, diff=True)

            # for now squeeze batch axis
            info = DotMap(x=x[0], 
                          fx=prediction.fx[0],
                          fa=prediction.fa[0],
                          l=cost.l[0],
                          lx=cost.lx[0],
                          lxx=cost.lxx[0],
                          la=cost.la[0],
                          laa=cost.laa[0],
                          lax=cost.lax[0])

            x = prediction.x
            rollout.append(info)

        return rollout


    @torch.no_grad()
    def backward_pass(self, rollout):

        sols = []

        Vx = rollout[-1].lx
        Vxx = rollout[-1].lxx

        for i in range(self.horizon - 1, -1, -1):
            Qx, Qa, Qxx, Qax, Qaa = self.q(rollout[i], Vx, Vxx)

            try:
                U, S, V = torch.svd(Qaa.cpu()) ## torch.svd on GPU is slow/unstable
            except RuntimeError:
                raise SVDError

            if torch.min(S) < self.eps:

                eps = 1e-8
                S[S < eps] = eps

            if torch.isnan(Qaa).any():
                raise Exception # TODO make the error more specific

            Qaa_inv = V.mm(torch.diag(1. / S)).mm(U.T)
            Qaa = U.mm(torch.diag(S)).mm(V.T)

            k = -Qaa_inv.mv(Qa)
            K = -Qaa_inv.mm(Qax)

            Vx = Qx + K.T.mm(Qaa).mv(k)
            Vx += K.T.mv(Qa) + Qax.T.mv(k)

            Vxx = Qxx + K.T.mm(Qaa).mm(K)
            Vxx += K.T.mm(Qax) + Qax.T.mm(K)
            Vxx = 0.5 * (Vxx + Vxx.T) 

            sol = DotMap(k=k,
                         K=K,
                         Vx=Vx,
                         Vxx=Vxx,
                         Qx=Qx,
                         Qxx=Qxx,
                         Qa=Qa,
                         Qax=Qax,
                         Qaa=Qaa)

            sols.append(sol)

        return sols

    def q(self, info, Vx, Vxx):

        fx, fa = info.fx, info.fa
        lx, lxx = info.lx, info.lxx
        la, lax, laa = info.la, info.lax, info.laa

        # TODO: rewrite this to batch version someday

        Qx = lx + fx.T.mv(Vx)
        Qa = la + fa.T.mv(Vx)
        Qxx = lxx + fx.T.mm(Vxx).mm(fx)

        reg = self.mu * torch.eye(self.nx)
        Qax = lax + fa.T.mm(Vxx + reg).mm(fx)
        Qaa = laa + fa.T.mm(Vxx + reg).mm(fa)

        return Qx, Qa, Qxx, Qax, Qaa

    def compute_cost(self, rollout):
        J = 0
        for info in rollout:
            J += rollout.l
        return J


    def regularize(self, ns):
        pass

    def reset(self):
        pass




