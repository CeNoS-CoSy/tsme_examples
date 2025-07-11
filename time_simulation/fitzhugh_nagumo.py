import numpy as np
from tsme.time_simulation import AbstractTimeSimulator
from tsme.utils.visualization import animate
import json as rick

np.random.seed(12389)

N = 256
Lx = 128
Ly = 128
domain = ((0, Lx), (0, Ly))
time_interval = [0, 150]
t_eval = np.linspace(time_interval[0], time_interval[-1], 200)

u0 = (np.random.random((N, N)) - 0.5) * 0.5
v0 = (np.random.random((N, N)) - 0.5) * 0.5


class FHNSim(AbstractTimeSimulator):
    def __init__(self, ic=None, dom=domain, params=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], bc="periodic",
                 diff="finite_difference"):
        if ic is None:
            n = 256  # Number of spatial points in each direction
            Lx = dom[0][1] - dom[0][0]
            Ly = dom[1][1] - dom[1][0]
            x_uniform = np.linspace(-Lx/2, Lx/2, n + 1) * 0.5
            y_uniform = np.linspace(-Ly/2, Ly/2, n + 1) * 0.5

            x = x_uniform[:n]
            y = y_uniform[:n]

            # Get 2D meshes in (x, y)
            X, Y = np.meshgrid(x, y)

            m = 1  # number of spirals

            # define our solution vectors
            u0 = np.zeros((len(x), len(y)))
            v0 = np.zeros((len(x), len(y)))

            # Initial conditions
            u0[:, :] = np.tanh(np.sqrt(X ** 2 + Y ** 2)) * np.cos(
                m * np.angle(X + 1j * Y) - (np.sqrt(X ** 2 + Y ** 2))
            )
            v0[:, :] = np.tanh(np.sqrt(X ** 2 + Y ** 2)) * np.sin(
                m * np.angle(X + 1j * Y) - (np.sqrt(X ** 2 + Y ** 2))
            )
            ic = np.array([u0, v0])

        super().__init__(ic, domain=dom, bc=bc, diff=diff)
        self.d_u = params[0]
        self.d_v = params[1]
        self.lamb = params[2]
        self.omega = params[3]
        self.kappa = params[4]
        self.tau = params[5]

    def rhs(self, t, u_in):
        u = u_in[0]
        v = u_in[1]
        f = self.lamb * u - u ** 3 - self.kappa
        u_next = self.d_u ** 2 * (self.diff.d_dx(u, 2) + self.diff.d_dy(u, 2)) + f - self.omega * v
        v_next = self.d_v ** 2 * (self.diff.d_dx(v, 2) + self.diff.d_dy(v, 2)) + u - v

        return np.array([u_next, self.tau ** (-1) * v_next])


sim = FHNSim(ic=np.array([u0, v0]), dom=domain, params=[0.5, 0.5, 0.5, 0.5, 0., 15])
# sim = FHNSim(ic=np.array([u0, v0]), dom=domain, params=[1, 1, 1, 1, 0, 15])
# pars = [0.5, 0.5, 0.5, 0.5, 0., 15] # [0.5, 0.5, 0.5, 0.5, 0., 15]
# sim = FHNSim(ic=None, dom=domain, params=pars)
sol = sim.simulate(time_interval, method="DOP853")  # , method="RK45", t_eval=t_eval)
sol_dot = sim.get_sol_dot()

# animate(sol,time=t_eval, savepath="tsme_examples/diagrams/PDEs/fhn_u.mp4", variable=0)
# animate(sol,time=t_eval, savepath="tsme_examples/diagrams/PDEs/fhn_v.mp4", variable=1)


with open("tsme_examples/data/time_series/fhn.json", "w") as file:
    dic = {"sol": sol.tolist(), "time": sim.time.tolist(), "domain": sim.domain}
    rick.dump(dic, file)
