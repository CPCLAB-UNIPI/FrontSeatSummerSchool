from cstr_model import CSTRParameters, setup_cstr_model, setup_linearized_model
from setup_ocp_solver import (
    MpcCstrParameters,
    setup_acados_ocp_solver,
    CasadiOcpSolver,
    AcadosOcpSolver,
)
from setup_acados_integrator import setup_acados_integrator, AcadosSimSolver
import numpy as np
from cstr_utils import plot_cstr
from typing import Optional, Union


def simulate(
    controller: Optional[Union[AcadosOcpSolver, CasadiOcpSolver]],
    plant: AcadosSimSolver,
    x0: np.ndarray,
    Nsim: int,
    X_ref: np.ndarray,
    U_ref: np.ndarray,
):

    nx = X_ref.shape[1]
    nu = U_ref.shape[1]

    X = np.ndarray((Nsim + 1, nx))
    U = np.ndarray((Nsim, nu))
    timings_solver = np.zeros((Nsim))
    timings_integrator = np.zeros((Nsim))

    # closed loop
    xcurrent = x0
    X[0, :] = xcurrent

    for i in range(Nsim):

        if controller is None:
            U[i, :] = U_ref[i, :]
        elif isinstance(controller, AcadosOcpSolver):

            yref = np.concatenate((X_ref[i, :], U_ref[i, :]))
            for stage in range(controller.acados_ocp.dims.N):
                controller.set(stage, "yref", yref)
            controller.set(controller.acados_ocp.dims.N, "yref", X_ref[i, :])

            # solve ocp
            U[i, :] = controller.solve_for_x0(xcurrent)
            timings_solver[i] = controller.get_stats("time_tot")

        elif isinstance(controller, CasadiOcpSolver):
            controller.set_yref(np.concatenate((X_ref[i, :], U_ref[i, :])))
            controller.set_x0(xcurrent)
            U[i, :] = controller.solve()
            timings_solver[i] = controller.get_timing()

        # simulate system
        plant.set("x", xcurrent)
        plant.set("u", U[i, :])

        if plant.acados_sim.solver_options.integrator_type == "IRK":
            plant.set("xdot", np.zeros((nx,)))

        status = plant.solve()
        if status != 0:
            raise Exception(
                f"acados integrator returned status {status} in simulation step {i}. Exiting."
            )

        timings_integrator[i] = plant.get("time_tot")
        # update state
        xcurrent = plant.get("x")
        X[i + 1, :] = xcurrent

    return X, U, timings_solver, timings_integrator


def main():
    # TODO: exercise: switch on/off the different controllers
    with_nmpc_controller = False
    with_casadi_nmpc_controller = False
    with_rti_controller = False
    with_linear_mpc_controller = False
    Tsim = 25
    dt_plant = 0.25  # [min]

    cstr_params = CSTRParameters()
    mpc_params = MpcCstrParameters(xs=cstr_params.xs, us=cstr_params.us)
    model = setup_cstr_model(cstr_params)
    linearized_model = setup_linearized_model(model, cstr_params, mpc_params)
    plant_model = setup_cstr_model(cstr_params)

    Nsim = int(Tsim / dt_plant)
    if not (Tsim / dt_plant).is_integer():
        print("WARNING: Tsim / dt_plant should be an integer!")

    integrator = setup_acados_integrator(plant_model, dt_plant, cstr_param=cstr_params)

    # steady-state
    xs = np.array([[0.878, 324.5, 0.659]]).T
    us = np.array([[300, 0.1]]).T

    # constant ref
    X_ref = np.tile(xs, Nsim + 1).T
    U_ref = np.tile(us, Nsim).T

    # reference jump
    xs2 = np.array([0.7, 337, 0.75])
    us2 = np.array([305, 0.1])
    # Njump = int(Nsim/4)
    Njump = int(Nsim / 3)
    X_ref[Njump : 2 * Njump, :] = xs2
    U_ref[Njump : 2 * Njump, :] = us2

    # initial state
    x0 = np.array([0.05, 0.75, 0.5]) * xs.ravel()

    X_all = []
    U_all = []
    labels_all = []
    timings_solver_all = []

    # simulation with constant reference input
    label = "constant reference input"
    print(f"\n\nRunning simulation with {label}\n\n")
    X, U, timings_solver, _ = simulate(None, integrator, x0, Nsim, X_ref, U_ref)
    X_all.append(X)
    U_all.append(U)
    timings_solver_all.append(timings_solver)
    labels_all.append(label)

    # simulation with NMPC controller
    if with_casadi_nmpc_controller:
        label = "casadi NMPC"
        print(f"\n\nRunning simulation with {label}\n\n")
        ocp_solver = CasadiOcpSolver(model, mpc_params, cstr_params=cstr_params)

        X, U, timings_solver, _ = simulate(
            ocp_solver, integrator, x0, Nsim, X_ref=X_ref, U_ref=U_ref
        )
        X_all.append(X)
        U_all.append(U)
        timings_solver_all.append(timings_solver)
        labels_all.append(label)
        ocp_solver = None

    # simulation with NMPC controller
    if with_nmpc_controller:
        label = "NMPC"
        print(f"\n\nRunning simulation with {label}\n\n")
        ocp_solver = setup_acados_ocp_solver(model, mpc_params, cstr_params=cstr_params)

        X, U, timings_solver, _ = simulate(
            ocp_solver, integrator, x0, Nsim, X_ref=X_ref, U_ref=U_ref
        )
        X_all.append(X)
        U_all.append(U)
        timings_solver_all.append(timings_solver)
        labels_all.append(label)
        ocp_solver = None

    # simulation with LMPC controller
    if with_linear_mpc_controller:
        label = "LMPC"
        print(f"\n\nRunning simulation with {label}\n\n")
        mpc_params.linear_mpc = True
        ocp_solver = setup_acados_ocp_solver(
            linearized_model, mpc_params, cstr_params=cstr_params, use_rti=True
        )
        mpc_params.linear_mpc = False

        X, U, timings_solver, _ = simulate(
            ocp_solver, integrator, x0, Nsim, X_ref=X_ref, U_ref=U_ref
        )
        X_all.append(X)
        U_all.append(U)
        timings_solver_all.append(timings_solver)
        labels_all.append(label)
        ocp_solver = None

    # simulation with NMPC RTI controller
    if with_rti_controller:
        label = "NMPC-RTI"
        print(f"\n\nRunning simulation with {label}\n\n")
        ocp_solver = setup_acados_ocp_solver(
            model, mpc_params, cstr_params=cstr_params, use_rti=True
        )

        X, U, timings_solver, _ = simulate(
            ocp_solver, integrator, x0, Nsim, X_ref=X_ref, U_ref=U_ref
        )
        X_all.append(X)
        U_all.append(U)
        timings_solver_all.append(timings_solver)
        labels_all.append(label)
        ocp_solver = None

    # Evaluation
    max_label_length = max([len(l) for l in labels_all])
    print(f"\n{'Timings in ms':{max_label_length}}  | {'miniumum':<8} | {'mean':<8} | {'maximum':<8} \n------------------")

    for i in range(len(labels_all)):
        label = labels_all[i]
        timings_solver = timings_solver_all[i] * 1e3  # scale to milliseconds
        print(
            f"{label:{max_label_length}}   {f'{np.min(timings_solver):.3f}':>10} {f'{np.mean(timings_solver):.3f}':>10} {f'{np.max(timings_solver):.3f}':>10}"
        )

    # plot results
    plot_cstr(
        dt_plant,
        X_all,
        U_all,
        X_ref,
        U_ref,
        mpc_params.umin,
        mpc_params.umax,
        labels_all,
        fig_filename='cstr_acados_RTI.pdf',
    )


if __name__ == "__main__":
    main()
