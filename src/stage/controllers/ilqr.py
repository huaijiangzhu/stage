import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from tqdm import trange
from dotmap import DotMap

from stage.controllers.base import Controller
from stage.controllers.trivial import Identity, OpenLoop
from stage.utils.nn import renew

class SVDError(Exception):
    pass

class ILQR(nn.Module):

    def __init__(self, dynamics, cost, actor, plan_horizon,
                 alpha=1.0, decay=0.05):
        super().__init__()
        self.dynamics = dynamics
        self.nx = dynamics.nx
        self.actor = actor
        self.na = actor.na
        self.action_ub, self.action_lb = actor.action_ub, actor.action_lb
        self.cost = cost
        self.cost.actor = actor
        self.plan_horizon = plan_horizon
        self.rollout = None
        self.sols = None
        self.reset()

    def forward(self, x, params, random=False):
        t = params
        if t > 0:
            max_it = 1
        else:
            max_it = 50
        rollout = self.optimize(x, actions_init=self.prev_actions, 
                                horizon=self.plan_horizon,
                                max_it=max_it)
        actions = torch.stack([info.a for info in self.rollout])
        a = actions[0]
        random_action = self.actor.sample()
        random_action = random_action.unsqueeze(0)
        self.prev_actions = torch.cat((actions[1:], random_action), dim=0)
        return renew(a)

    def reset(self):
        self.mu = 1.0
        self.mu_min = 1e-6
        self.mu_max = 1e10
        self.delta0 = 2.0
        self.delta = self.delta0
        self.eps = 1e-8

        self.rollout = None
        self.sols = None
        self.prev_actions = torch.stack([self.actor.sample() for t in range(self.plan_horizon)])
    
    def update(self, x):
        pass
        # if self.rollout is not None:
        #     actions = torch.stack([info.a for info in self.rollout])
        #     self.optimize(x, actions)
        # else:
        #     self.optimize(x)


    def optimize(self, x, actions_init=None, horizon=None, max_it=10, on_iteration=None, tol=1e-6):
        exponent = -torch.arange(10)**2 * torch.log(1.1 * torch.ones(10))
        alphas = torch.exp(exponent)

        changed = True
        converged = False
        actions = actions_init

        if horizon is None:
            horizon = self.plan_horizon

        if actions is None:
            actions = [self.actor.sample() for t in range(horizon)]

        for it in range(max_it):
            accepted = False
            if changed:          
                rollout = self.unroll(x, actions, horizon)
                J_opt = self.compute_cost(rollout)
                changed = False
            try:
                sols = self.backward_pass(rollout)

                # line search
                for alpha in alphas:

                    rollout_new = self.control(rollout, sols, alpha)
                    J_new = self.compute_cost(rollout_new)

                    if J_new < J_opt:
                        if torch.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True

                        J_opt = J_new
                        rollout = rollout_new
                        changed = True

                        # Decrease regularization term.
                        self.delta = min(1.0, self.delta) / self.delta0
                        self.mu *= self.delta
                        if self.mu <= self.mu_min:
                            self.mu = 0.0

                        # Accept this.
                        accepted = True
                        break

            except SVDError as e:
                # Qaa was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn("ill-conditioned Qaa")

            if not accepted:
                # Increase regularization term.
                self.delta = max(1.0, self.delta) * self.delta0
                self.mu = max(self.mu_min, self.mu * self.delta)
                if self.mu_max and self.mu >= self.mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if converged:
                break

        self.rollout = rollout

        actions = torch.stack([info.a for info in self.rollout])
        self.openloop = OpenLoop(self.nx, self.actor, actions)

        return rollout

    @torch.no_grad()
    def clamp_action(self, a):
        return torch.max(self.actor.action_lb, torch.min(a, self.actor.action_ub))

    @torch.no_grad()
    def control(self, rollout, sols, alpha):
        rollout_new = []
        horizon = len(rollout)
        x = rollout[0].x

        for t in range(horizon):
            if t == horizon - 1:
                terminal = True
            else:
                terminal = False
            x_ = rollout[t].x
            a_ = rollout[t].a
            k = sols[t].k
            K = sols[t].K
            a = a_ + alpha * k + K.mv(x - x_)

            # still need to do this because boxqp only constrains a_ + k
            a = self.clamp_action(a)

            cost = self.cost.l(renew(x), renew(a), t, terminal, diff=False)
            prediction = self.dynamics.sample_predictions(x, a, n_particles=0, diff=False)
            info = DotMap(x=x,
                          a=a,
                          l=cost.l[0])
            x = prediction.x[0]
            rollout_new.append(info)


        return rollout_new


    def unroll(self, x, actions, horizon):

        rollout = []

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        for t in range(horizon):

            a = actions[t]
            prediction = self.dynamics.sample_predictions(x, a, n_particles=0, diff=False)
            # handle terminal cost
            if t == horizon - 1:
                cost = self.cost.l(renew(x), renew(a), t, terminal=True, diff=True)
                info = DotMap(x=x,
                              a=a, 
                              fx=prediction.fx[0],
                              fa=prediction.fa[0],
                              l=cost.l[0],
                              lx=cost.lx[0],
                              lxx=cost.lxx[0],
                              la=cost.la[0],
                              laa=cost.laa[0],
                              lax=cost.lax[0])

            else:
                info = DotMap(x=x, a=a)
            x = prediction.x[0]
            rollout.append(info)

        X = torch.stack([info.x for info in rollout])
        A = torch.stack([info.a for info in rollout])

        prediction = self.dynamics.sample_predictions(X, A, n_particles=0, diff=True)
        cost = self.cost.l(renew(X), renew(A), t, terminal=False, diff=True)

        for i in range(horizon):
            rollout[i].fx = prediction.fx[i]
            rollout[i].fa = prediction.fa[i]
            if i < horizon - 1:
                rollout[i].l=cost.l[i]
                rollout[i].lx=cost.lx[i]
                rollout[i].lxx=cost.lxx[i]
                rollout[i].la=cost.la[i]
                rollout[i].laa=cost.laa[i]
                rollout[i].lax=cost.lax[i]

        return rollout


    @torch.no_grad()
    def backward_pass(self, rollout):

        sols = []

        Vx = rollout[-1].lx
        Vxx = rollout[-1].lxx

        horizon = len(rollout)
        for i in range(horizon - 1, -1, -1):
            Qx, Qa, Qxx, Qax, Qaa = self.q(rollout[i], Vx, Vxx)

            try:
                U, S, V = torch.svd(Qaa.cpu()) ## torch.svd on GPU is slow/unstable
            except RuntimeError:
                raise SVDError

            U = U.cuda()
            S = S.cuda()
            V = V.cuda()

            if torch.min(S) < self.eps:

                eps = 1e-8
                S[S < eps] = eps

            if torch.isnan(Qaa).any():
                raise SVDError

            Qaa_inv = V.mm(torch.diag(1. / S)).mm(U.T)
            Qaa = U.mm(torch.diag(S)).mm(V.T)

            k = -Qaa_inv.mv(Qa)
            K = -Qaa_inv.mm(Qax)

            Vx = Qx + K.T.mm(Qaa).mv(k)
            Vx += K.T.mv(Qa) + Qax.T.mv(k)

            Vxx = Qxx + K.T.mm(Qaa).mm(K)
            Vxx += K.T.mm(Qax) + Qax.T.mm(K)
            Vxx = 0.5 * (Vxx + Vxx.T)

            a = rollout[i].a
            b_lower = self.actor.action_lb - a
            b_upper = self.actor.action_ub - a
            k, clamped_idx = self.box_qp(a, Qaa, Qa, b_lower, b_upper)
            for c_idx in clamped_idx:
                K[c_idx, :] = torch.zeros_like(K[c_idx, :])

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

        sols.reverse()

        return sols

    @torch.no_grad()
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

    @torch.no_grad()
    def compute_cost(self, rollout):
        J = 0
        for info in rollout:
            J += info.l
        return J


    def regularize(self, ns):
        pass

    @torch.no_grad()
    def box_qp(self, x, H, q, b_lower, b_upper, tol=1e-3):
        """
        Projected-Newton QP Solver. Used to find the optimal control for the
        backwards pass of iLQR when taking bounds for the controls into account.

        Implementation based on Appendix I from the paper:
            Control-Limited Differential Dynamic Programming
            Yuval Tassa, Nicolas Mansard, Emo Todorov
            https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

        Assuming the goal is to find k to minimize 1/2 k'Hk + q'k , such that -b_lower <= k <= b_upper.
        """
        fx = lambda x: 0.5 * x.dot(H.mv(x)) + q.dot(x)
        clamp = lambda v: torch.max(b_lower, torch.min(v, b_upper))

        x = clamp(x.clone())

        while True:
            # Gradient.
            g = q + H.mv(x)

            # Clamped and free indices.
            c_idx = (x.eq(b_lower) & (g > 0)) | (x.eq(b_upper) & (g < 0))
            f_idx = ~c_idx

            dF = torch.sum(f_idx)

            # Sort by the free and clamped indices.
            H_fc = torch.cat((H[f_idx, :], H[c_idx, :]), dim=0)
            Hff, Hfc = H_fc[:dF, :dF], H_fc[:dF, dF:]

            xf, xc = x[f_idx], x[c_idx]
            qf = q[f_idx]

            # Compute the free-gradient and exit optimization if small enough.
            # Otherwise compute the step-direction of the optimization.
            gf = qf + Hff.mv(xf) + Hfc.mv(xc)

            if torch.norm(gf, p=2) < tol:
                return x, torch.where(c_idx)

            step_x = torch.zeros(x.shape[0])
            step_x[f_idx] = -Hff.inverse().mv(gf)

            ###
            # Doing backtracking line-search to find optimal alpha value.
            alpha = 1.0  # Step-size
            beta = 0.95  # Step-size reduction on each iteration
            gamma = 0.1

            f_x = fx(x)

            while True:
                x_alpha = clamp(x + alpha * step_x)

                if f_x > gamma * g.dot(x - x_alpha) + fx(x_alpha):
                    break

                alpha *= beta

                if torch.norm(alpha * step_x, p=2) < tol:
                    return x_alpha, torch.where(c_idx)

            x = x_alpha





