import numpy as np
import casadi as ca

class Model:
    def __init__(self, x_expr, u_expr, f_expr, x_steady_state, u_steady_state, name='model'):
        self.name = name

        self.x_steady_state = x_steady_state
        self.u_steady_state = u_steady_state

        self.x_expr = x_expr
        self.u_expr = u_expr

        self.f_expr = f_expr
        self.f = ca.Function('f', [x_expr, u_expr], [f_expr])

        self.J_x = ca.Function('J_x', [x_expr, u_expr], [ca.jacobian(f_expr, x_expr)])
        self.J_u = ca.Function('J_u', [x_expr, u_expr], [ca.jacobian(f_expr, u_expr)])

        self.nx = x_expr.rows()
        self.nu = u_expr.rows()

    def simulate(self, x0, u0, w0=0.0):
        x1 = np.reshape(self.f(x0, u0).full(), (self.nx, 1)) + w0

        return x1

    def simulate_traj(self, x0, u_traj):
        n_sim = u_traj.shape[1]

        x_traj = np.zeros((self.nx, n_sim + 1))
        x_traj[:, 0] = x0.ravel()

        for n in range(n_sim):
            x1 = self.simulate(x_traj[:, n], u_traj[:, n])
            x_traj[:, [n + 1]] = x1

        return x_traj