from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from scipy.linalg import block_diag
import numpy as np
from dataclasses import dataclass
import casadi as ca

@dataclass
class MpcCstrParameters:
    umin: np.ndarray  # lower bound on u
    umax: np.ndarray  # upper bound on u
    Q: np.ndarray
    R: np.ndarray
    Tf: float = 0.25 * 15  # horizon length
    N: int = 15
    dt: float = 0.25
    linear_mpc: bool = False

    # NOTE: computed with setup_linearized_model()
    P: np.ndarray = np.array(
        [
            [5.92981953e-01, -8.40033347e-04, -1.54536980e-02],
            [-8.40033347e-04, 7.75225208e-06, 2.30677411e-05],
            [-1.54536980e-02, 2.30677411e-05, 2.59450075e00],
        ]
    )

    def __init__(self, xs, us):
        self.Q = np.diag(1.0 / xs**2)
        self.R = np.diag(1.0 / us**2)
        # from slide
        # self.umin = np.array([0.975, 0.75]) * us
        # self.umax = np.array([1.025, 1.25]) * us
        # from figure code
        self.umin = np.array([0.95, 0.85]) * us
        self.umax = np.array([1.05, 1.15]) * us


def integrate_RK4(x_expr, u_expr, p_expr, xdot_expr, dt, n_steps=1) -> ca.Function:
    h = dt / n_steps

    x_end = x_expr

    xdot_fun = ca.Function('xdot', [x_expr, u_expr, p_expr], [xdot_expr])

    for _ in range(n_steps):
        k_1 = xdot_fun(x_end, u_expr, p_expr)
        k_2 = xdot_fun(x_end + 0.5 * h * k_1, u_expr, p_expr)
        k_3 = xdot_fun(x_end + 0.5 * h * k_2, u_expr, p_expr)
        k_4 = xdot_fun(x_end + k_3 * h, u_expr, p_expr)

        x_end = x_end + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h

    F_expr = x_end
    F_fun = ca.Function('F', [x_expr, u_expr, p_expr], [F_expr])

    return F_fun


class CasadiOcpSolver():
    """
    A simple implementation of a CasADi-based MPC solver for the CSTR problem.
    This class uses:
    - single shooting
    - multiple steps of an RK4 integrator on each shooting interval
    - IPOPT as an NLP solver
    -> this is a simple implementation for educational purposes
    -> using multiple shooting and a better integrator could improve the controller
    """
    def __init__(self, model: AcadosModel, mpc_params: MpcCstrParameters, cstr_params):

        self.N = mpc_params.N
        self.model = model

        x = model.x
        u = model.u
        nx = x.rows()
        nu = u.rows()
        self.nx = nx
        self.nu = nu

        cost_y_expr = ca.vertcat(x, u)
        cost_y_expr_e = x
        ny = cost_y_expr.rows()

        yref = ca.SX.sym("yref", ny)
        yref_e = ca.SX.sym("yref_e", nx)
        W = block_diag(mpc_params.Q, mpc_params.R)
        W_e = mpc_params.P

        dt = mpc_params.Tf / mpc_params.N
        stage_residual = cost_y_expr - yref
        terminal_residual = cost_y_expr_e - yref_e
        stage_cost = dt * ca.mtimes([stage_residual.T, W, stage_residual])
        terminal_cost = dt * ca.mtimes([terminal_residual.T, W_e, terminal_residual])

        stage_cost_fun = ca.Function('l', [x, u, yref], [dt * stage_cost])
        terminal_cost_fun = ca.Function('lN', [x, yref_e], [terminal_cost])

        # setup OCP in single shooting formulation
        u_traj = ca.SX.sym('u_traj', nu, self.N)
        x0_bar = ca.SX.sym('x0_bar', nx)

        nlp_parameters = ca.vertcat(x0_bar, yref, yref_e, model.p)

        # single shooting formulation but excluding x0
        f_discrete = integrate_RK4(x, u, model.p, model.f_expl_expr, dt, n_steps=20)
        F_single_shooting = f_discrete.mapaccum(self.N)(x0_bar, u_traj, model.p)

        x_traj = ca.horzcat(x0_bar, F_single_shooting)

        constraints = ca.vertcat(ca.vec(u_traj))

        self.ubg = np.tile(mpc_params.umax, reps=self.N)
        self.lbg = np.tile(mpc_params.umin, reps=self.N)

        objective = ca.sum2(stage_cost_fun(x_traj[:, :-1], u_traj, yref)) + terminal_cost_fun(x_traj[:, -1], yref_e)

        self.ocp = {'f': objective, 'x': ca.vec(u_traj), 'g': constraints, 'p': nlp_parameters}
        self.solver = ca.nlpsol('solver', 'ipopt', self.ocp)

        self.yref_val = np.zeros((ny,))
        self.x0_val = np.zeros((nx,))
        self.cstr_params = cstr_params

    def set_yref(self, yref):
        self.yref_val = yref

    def set_x0(self, x0):
        self.x0_val = x0

    def solve(self):
        # solve the NLP
        p_val = ca.vertcat(self.x0_val, self.yref_val, self.yref_val[:self.nx], self.cstr_params.F0)
        sol = self.solver(lbg=self.lbg, ubg=self.ubg, p=p_val)
        self.u_traj_opt = sol['x'].full().flatten()
        self.stats = self.solver.stats()
        self.status = self.stats["return_status"] == 'Solve_Succeeded'

        u0 = self.u_traj_opt[:self.nu]
        return u0

    def get_timing(self):
        return self.solver.stats()['t_wall_total']


def setup_acados_ocp_solver(
    model: AcadosModel, mpc_params: MpcCstrParameters, cstr_params, use_rti=False
) -> AcadosOcpSolver:

    ocp = AcadosOcp()

    # set model
    ocp.model = model
    x = model.x
    u = model.u
    nx = x.rows()
    nu = u.rows()

    # number of shooting intervals
    ocp.dims.N = mpc_params.N

    # set prediction horizon
    ocp.solver_options.tf = mpc_params.Tf

    # nominal parameter values
    ocp.parameter_values = np.array([cstr_params.F0])

    # set cost
    ocp.cost.W_e = mpc_params.P
    ocp.cost.W = block_diag(mpc_params.Q, mpc_params.R)

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.model.cost_y_expr = ca.vertcat(x, u)
    ocp.model.cost_y_expr_e = x

    ocp.cost.yref = np.zeros((nx + nu,))
    ocp.cost.yref_e = np.zeros((nx,))

    # set constraints
    ocp.constraints.lbu = mpc_params.umin
    ocp.constraints.ubu = mpc_params.umax
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.x0 = cstr_params.xs

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.print_level = 1
    if use_rti:
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
    else:
        ocp.solver_options.nlp_solver_type = "SQP"

    if mpc_params.linear_mpc:
        ocp.solver_options.integrator_type = "DISCRETE"
    else:
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 5

    ocp.solver_options.levenberg_marquardt = 1e-5
    ocp.solver_options.line_search_use_sufficient_descent

    ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    return ocp_solver


if __name__ == "__main__":
    setup_acados_ocp_solver()
