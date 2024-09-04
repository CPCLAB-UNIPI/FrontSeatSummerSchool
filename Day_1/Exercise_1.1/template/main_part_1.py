import matplotlib.pyplot as plt
import casadi as ca
import numpy as np



def plot_circle(x_i, y_i, r, i):

    eps = 1e-8

    r_i = r if i < 3 else 2*r

    X = np.linspace(float(x_i-r_i), float(x_i+r_i), 100)

    Y_pos = np.sqrt(r_i**2 - (X - x_i)**2 + eps)+y_i
    Y_neg = -np.sqrt(r_i**2 - (X - x_i)**2 + eps)+y_i

    plt.plot(X, Y_pos, color = "steelblue")
    plt.plot(X, Y_neg, color = "steelblue")


def plot_center(x, y, i):

    plt.scatter(float(x), float(y), color = "steelblue", marker = "x")
    plt.text(float(x+0.2), float(y+0.2), "s_" + str(i+1))


def plot_plate(x_val, y_val, R_val, a):

    plt.figure(figsize=(6,6))
    n_p = x_val.shape[0]

    for i in range(n_p):

        plot_circle(x_val[i], y_val[i], R_val, i)
        plot_center(x_val[i], y_val[i], i)

    plt.xlim([0, a])
    plt.ylim([0, a])

    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid(True)

    # plt.savefig("/tmp/plate.eps", format = "eps", bbox_inches = "tight")


def main():

    a = 10.0

    n_p = 5

    x = ca.MX.sym("x", n_p)
    y = ca.MX.sym("y", n_p)

    R = ca.MX.sym("R", 1)

    r = np.array([R, R, R, 2*R, 2*R])

    # g is a list of constraint expressions
    # g_min and g_max are the lower and upper bounds that these expressions need to satisfy
    g = []
    g_min = []
    g_max = []

    for i in range(n_p):
        # TODO: add constraints 1. and 2. here using g, g_min and g_max
        g.append(x[i]-r[i])
        g_min.append(0)
        g_max.append(np.inf)
        g.append(x[i]+r[i]-a)
        g_min.append(-np.inf)
        g_max.append(0)
        g.append(y[i]-r[i])
        g_min.append(0)
        g_max.append(np.inf)
        g.append(y[i]+r[i]-a)
        g_min.append(-np.inf)
        g_max.append(0)
        
        pass

    for i in range(n_p):

        for j in range(i+1, n_p):

            # TODO: add constraints 3. here using g, g_min and g_max
            g.append((x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2)
            g_min.append(0)
            g_max.append(np.inf)

            pass

    g = ca.vertcat(*g)

    x_init = np.array([1, 3, 8, 7, 3])
    y_init = np.array([1, 3, 8, 3, 7])
    R_init = 2.0


    plot_plate(x_init, y_init, R_init, a)

    V = ca.vertcat(x, y, R)

    nlp = {"x": V, "f": -R, "g": g}

    solver = ca.nlpsol("solver", "ipopt", nlp)

    V_init = ca.vertcat(x_init, y_init, R_init)

    x_min = np.zeros(n_p)
    x_max = a * np.ones(n_p)

    y_min = np.zeros(n_p)
    y_max = a * np.ones(n_p)

    R_min = 0.0
    R_max = np.inf

    V_min = ca.vertcat(x_min, y_min, R_min)
    V_max = ca.vertcat(x_max, y_max, R_max)

    solution = solver(x0 = V_init, lbx = V_min, ubx = V_max, \
        lbg = g_min, ubg = g_max)

    V_opt = solution["x"]

    x_opt = V_opt[:n_p]
    y_opt = V_opt[n_p:-1]
    R_opt = V_opt[-1]
    print("The optimal radius found is = ", R_opt)
    plot_plate(x_opt, y_opt, R_opt, a)
    plt.show()


if __name__ == '__main__':
    main()