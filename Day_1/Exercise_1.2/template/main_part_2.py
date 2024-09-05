
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
    
    # Nonlinear model
    #model = setup_cstr_model(dt, num_rk4_steps, params)

    A = np.array([[0.2681, -3.38*1e-03, -7.28*1e-03], [9.703, .3279, -25.44], [0.0, 0.0, 1.0]])
    B = np.array([[-5.37*1e-03, 0.1655], [1.297, 97.91], [0.0, -6.637]])
    C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])

    # set up states
    x = ca.SX.sym("x",3)

    # controls
    u = ca.SX.sym("u",2)

    # TODO setup model expressions
    # Use x and u to set up the model expression
    # the references are given by xs and us
    # the model matrices are given by A, B, and C
    model = 42

    xs = params.xs
    us = params.us
    x0 = np.array([0.05, 0.75, 0.5]) * xs.ravel()
    N_horizon = 20

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

    # TODO setup cost expressions
    # Use model.x_expr and model.u_expr to set up the stage and terminal cost of the OCP
    # the references are given by xs and us
    # the weighting matrices are given by Q, R, and P
    stage_cost = 42
    terminal_cost = 42

    umin = np.array([0.95, 0.85]) * us
    umax = np.array([1.05, 1.15]) * us

    ocp_solver = OCPsolver(model, stage_cost, terminal_cost, N_horizon, umax, umin)

    x_ref  = np.array([model.x_steady_state]*(N_horizon+1)).T
    u_ref  = np.array([model.u_steady_state]*N_horizon).T

    # we use umin as initial guess
    u_init  = np.array([umin]*N_horizon).T
    (u_traj, cost) = ocp_solver.solve(x0, u_traj_init=u_init)

    x_traj = model.simulate_traj(x0, u_traj)

    plot_cstr(dt, [x_traj], [u_traj], x_ref, u_ref, umin, umax, ['open-loop solution'])



if __name__ == "__main__":

    main_mpc_open_loop()