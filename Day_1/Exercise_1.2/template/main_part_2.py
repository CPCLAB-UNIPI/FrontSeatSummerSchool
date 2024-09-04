
import numpy as np
import casadi as ca
from utils import integrate_RK4, plot_cstr
from ocp_solver_ipopt import OCPsolver
from dataclasses import dataclass
from model import Model

@dataclass
class CSTRParameters:
    # nominal parameter values
    F0: float = 0.1  # m^3/min
    T0: float = 350.0  # K
    c0: float = 1.0  # kmol/m^3
    r: float = 0.219  # m
    k0: float = 7.2 * 1e10  # 1/min
    EbR: float = 8750  # K
    U: float = 54.94  # kJ / (min*m^2*K)
    rho: float = 1000  # kg / m^3
    Cp: float = 0.239  # kJ / (kg*K)
    dH: float = -5 * 1e4  # kJ / kmol
    # to avoid division by zero
    eps: float = 1e-7
    xs: np.ndarray = np.array([0.878, 324.5, 0.659])
    us: np.ndarray = np.array([300, 0.1])


@dataclass
class MpcCstrParameters:
    umin: np.ndarray  # lower bound on u
    umax: np.ndarray  # upper bound on u
    Q: np.ndarray
    R: np.ndarray
    Tf: float = 0.25 * 15  # horizon length
    N: int = 20
    dt: float = 0.25
    num_rk4_steps: int = 10

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

        self.umin = np.array([0.95, 0.85]) * us
        self.umax = np.array([1.05, 1.15]) * us


def setup_cstr_model(dt: float, num_steps: int, params: CSTRParameters):

    # set up states
    c = ca.SX.sym("c")  # molar concentration of species A
    T = ca.SX.sym("T")  # reactor temperature
    h = ca.SX.sym("h")  # level of the tank

    x = ca.vertcat(c, T, h)

    # controls
    Tc = ca.SX.sym("Tc")  # temperature of coolant liquid
    F = ca.SX.sym("F")  # outlet flowrate

    u = ca.vertcat(Tc, F)

    # dynamics
    A_const = np.pi * params.r**2
    denom = A_const * h
    k = params.k0 * ca.exp(-params.EbR / T)
    rate = k * c

    f_expl = ca.vertcat(
        params.F0 * (params.c0 - c)/denom - rate,
        params.F0 * (params.T0 - T)/denom
        - params.dH/(params.rho * params.Cp) * rate
        + 2*params.U/(params.r * params.rho * params.Cp) * (Tc - T),
        (params.F0 - F)/A_const,
    )

    f_discrete = integrate_RK4(x, u, f_expl, dt, num_steps)

    model = Model(x, u, f_discrete, params.xs, params.us, name='cstr')

    return model


def main_mpc_open_loop():

    params = CSTRParameters()
    dt = 0.25
    num_rk4_steps = 10
    model = setup_cstr_model(dt, num_rk4_steps, params)

    x0 = np.array([0.05, 0.75, 0.5]) * model.x_steady_state.ravel()
    xs = params.xs
    us = params.us
    N_horizon = 20

    # TODO setup cost expressions
    # Use model.x_expr and model.u_expr to set up the stage and terminal cost of the OCP
    # the references are given by xs and us
    # the weighting matrices are given by Q, R, and P
    
    # NOTE: computed with setup_linearized_model()
    P = np.array(
        [
            [5.92981953e-01, -8.40033347e-04, -1.54536980e-02],
            [-8.40033347e-04, 7.75225208e-06, 2.30677411e-05],
            [-1.54536980e-02, 2.30677411e-05, 2.59450075e00],
        ]
    )
    
    Q = np.diag(1.0 / xs**2)
    R = np.diag(1.0 / us**2)

    delta_x = model.x_expr - model.x_steady_state
    delta_u = model.u_expr - model.u_steady_state
    stage_cost = 0.5 * delta_x.T @ Q @ delta_x + 0.5 * delta_u.T @ R @ delta_u
    terminal_cost = 0.5 * delta_x.T @ P @ delta_x

    umin = None #np.array([0.95, 0.85]) * us
    umax = None #np.array([1.05, 1.15]) * us

    ocp_solver = OCPsolver(model, stage_cost, terminal_cost, N_horizon, umax, umin)

    x_ref  = np.array([model.x_steady_state]*(N_horizon+1)).T
    u_ref  = np.array([model.u_steady_state]*N_horizon).T

    # we use umin as initial guess
    u_init  = np.array([mpc_params.umin]*N_horizon).T
    (u_traj, cost) = ocp_solver.solve(x0, u_traj_init=u_init)

    x_traj = model.simulate_traj(x0, u_traj)

    plot_cstr(mpc_params.dt, [x_traj], [u_traj], x_ref, u_ref, mpc_params.umin, mpc_params.umax, ['open-loop solution'])



if __name__ == "__main__":

    main_mpc_open_loop()