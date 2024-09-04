import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

def integrate_RK4(x_expr, u_expr, xdot_expr, dt, n_steps=1):
    h = dt / n_steps

    x_end = x_expr

    xdot_fun = ca.Function('xdot', [x_expr, u_expr], [xdot_expr])

    for _ in range(n_steps):
        k_1 = xdot_fun(x_end, u_expr)
        k_2 = xdot_fun(x_end + 0.5 * h * k_1, u_expr)
        k_3 = xdot_fun(x_end + 0.5 * h * k_2, u_expr)
        k_4 = xdot_fun(x_end + k_3 * h, u_expr)

        x_end = x_end + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h

    F_expr = x_end

    return F_expr


def plot_cstr(
    dt,
    X_list,
    U_list,
    X_ref,
    U_ref,
    u_min,
    u_max,
    labels_list,
    fig_filename=None,
    x_min=None,
    x_max=None,
):

    nx = X_list[0].shape[0]
    nu = U_list[0].shape[0]

    Nsim = U_list[0].shape[1]

    ts = dt * np.arange(0, Nsim + 1)

    states_lables = ["$c$ [kmol/m$^3$]", "$T$ [K]", "$h$ [m]"]
    controls_lables = ["$T_c$ [K]", "$F$ [m$^3$/min]"]

    # latexify_plot()
    fig, axes = plt.subplots(ncols=2, nrows=nx)

    for i in range(nx):
        for X, label in zip(X_list, labels_list):
            axes[i, 0].plot(ts, X[i], label=label, alpha=0.7)

        axes[i, 0].step(
            ts,
            X_ref[i],
            alpha=0.8,
            where="post",
            label="reference",
            linestyle="dotted",
            color="k",
        )
        axes[i, 0].set_ylabel(states_lables[i])
        axes[i, 0].grid()
        axes[i, 0].set_xlim(ts[0], ts[-1])

        if x_min is not None:
            axes[i, 0].set_ylim(bottom=x_min[i])

        if x_max is not None:
            axes[i, 0].set_ylim(top=x_max[i])

    for i in range(nu):
        for U, label in zip(U_list, labels_list):
            axes[i, 1].step(ts, np.append([U[i, 0]], U[i]), label=label, alpha=0.7)
        axes[i, 1].step(
            ts,
            np.append([U_ref[i, 0]], U_ref[i]),
            alpha=0.8,
            label="reference",
            linestyle="dotted",
            color="k",
        )
        axes[i, 1].set_ylabel(controls_lables[i])
        axes[i, 1].grid()

        axes[i, 1].hlines(
            u_max[i], ts[0], ts[-1], linestyles="dashed", alpha=0.8, color="k"
        )
        axes[i, 1].hlines(
            u_min[i], ts[0], ts[-1], linestyles="dashed", alpha=0.8, color="k"
        )
        axes[i, 1].set_xlim(ts[0], ts[-1])
        axes[i, 1].set_ylim(bottom=0.98 * u_min[i], top=1.02 * u_max[i])

    axes[1, 1].legend(bbox_to_anchor=(0.5, -1.75), loc="lower center")
    axes[-1, 0].set_xlabel("$t$ [min]")
    axes[1, 1].set_xlabel("$t$ [min]")

    fig.delaxes(axes[-1, 1])

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, hspace=0.3, wspace=0.4
    )
    if fig_filename is not None:
        plt.savefig(
            fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05
        )
        print(f"\nstored figure in {fig_filename}")

    plt.show()
