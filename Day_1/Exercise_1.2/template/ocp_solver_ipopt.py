import numpy as np
import casadi as ca
from model import Model


class OCPsolver():
    def __init__(self, model: Model, stage_cost: ca.SX, terminal_cost: ca.SX, N: int, u_max: float, u_min: float):

        self.N = N
        self.model = model

        nx = model.nx
        nu = model.nu

        stage_cost_fun = ca.Function('l', [model.x_expr, model.u_expr], [stage_cost])
        terminal_cost_fun = ca.Function('lN', [model.x_expr], [terminal_cost])

        # setup OCP in single shooting formulation
        u_traj = ca.SX.sym('u_traj', nu, N)

        x0_bar = ca.SX.sym('x0_bar', nx)

        # single shooting formulation but excluding x0
        F_single_shooting = self.model.f.mapaccum(N)(x0_bar, u_traj)

        x_traj = ca.horzcat(x0_bar, F_single_shooting)

        constraints = ca.vertcat(ca.vec(u_traj))

        self.ubg = np.tile(u_max, reps=N)
        self.lbg = np.tile(u_min, reps=N)

        objective = ca.sum2(stage_cost_fun(x_traj[:, :-1], u_traj)) + terminal_cost_fun(x_traj[:, -1])

        self.ocp = {'f': objective, 'x': ca.vec(u_traj), 'g': constraints, 'p': x0_bar}
        self.solver = ca.nlpsol('solver', 'ipopt', self.ocp)

        self.u_current = np.zeros((nu * N, 1))


    def solve(self, x0, u_traj_init=None):

        if u_traj_init is not None:
            self.u_current = ca.vec(u_traj_init)

        # solve the NLP
        sol = self.solver(x0=self.u_current, lbg=self.lbg, ubg=self.ubg, p=x0)

        u_opt = sol['x'].full()
        cost = sol['f'].full()
        self.status = not self.solver.stats()["return_status"] == 'Solve_Succeeded'

        self.u_current = u_opt

        return (ca.reshape(u_opt, (self.model.nu, self.N)).full(), cost)
