import numpy as np
from tsme.time_simulation import AbstractTimeSimulator
from tsme.utils.visualization import animate
import matplotlib.pyplot as plt
import json as rick

from joblib import Parallel, delayed

np.random.seed(12389)

# N = 128
# Lx = 90
# Ly = 90
# domain = ((0, Lx), (0, Ly))
# time_interval = [0, 15]
# u0 = (np.random.random((N, N)) - 0.5) * 0.5
#
#
# class CHESim(AbstractTimeSimulator):
#     def __init__(self, ic=None, dom=domain, params=[1.0, 1.0], bc="periodic", diff="finite_difference"):
#         super().__init__(ic, domain=dom, bc=bc, diff=diff)
#         self.a = params[0]
#         self.b = params[1]
#
#     def rhs(self, t, u_in):
#         u = u_in[0]
#         # f = u ** 3 - u - self.a * (self.diff.d_dx(u, 2) + self.diff.d_dy(u, 2))
#         # u_next = self.diff.d_dx(f, 2) + self.diff.d_dy(f, 2)
#         u_next = - self.diff.d_dx(u, 2) - self.diff.d_dy(u, 2) \
#                  + self.diff.d_dx(u ** 3, 2) + self.diff.d_dy(u ** 3, 2) \
#                  - self.a * self.diff.d_dx(u, 4) - self.a * self.diff.d_dy(u, 4) \
#                  - self.a * 2 * self.diff.dd_dxdy(u, [2, 2])
#         # - self.a * 2 * self.diff.d_dx(self.diff.d_dy(u, 2), 2)
#
#         return np.array([u_next])


N = 128
Lx = 20
Ly = 20
domain = ((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2))
time_interval = [0, 10]
t_eval = np.linspace(time_interval[0], time_interval[-1], int(10 / 0.05))

x = np.linspace(domain[0][0], domain[0][1], N)  # spatial coordinates
y = np.linspace(domain[1][0], domain[1][1], N)
x_mesh, y_mesh = np.meshgrid(x, y)

m = 1
u0 = np.tanh(np.sqrt(x_mesh ** 2 + y_mesh ** 2)) * np.cos(
    m * np.angle(x_mesh + 1j * y_mesh) - (np.sqrt(x_mesh ** 2 + y_mesh ** 2))
)
v0 = np.tanh(np.sqrt(x_mesh ** 2 + y_mesh ** 2)) * np.sin(
    m * np.angle(x_mesh + 1j * y_mesh) - (np.sqrt(x_mesh ** 2 + y_mesh ** 2))
)


class ReactionDiffusionSim(AbstractTimeSimulator):
    def __init__(self, ic=None, dom=domain, params=[1.0], bc="periodic", diff="finite_difference"):
        super().__init__(ic, domain=dom, bc=bc, diff=diff)
        self.beta = params[0]

    def rhs(self, t, u_in):
        u = u_in[0]
        v = u_in[1]
        A = u ** 2 + v ** 2
        u_next = 0.1 * (self.diff.d_dx(u, 2) + self.diff.d_dy(u, 2)) + (1 - A) * u + self.beta * A * v
        v_next = 0.1 * (self.diff.d_dx(v, 2) + self.diff.d_dy(v, 2)) - self.beta * A * u + (1 - A) * v

        return np.array([u_next, v_next])


# N = 256
# Lx = 90
# domain = ((0, Lx), )
# time_interval = [0, 8]
#
# x = np.linspace(domain[0][0], domain[0][1], N)
#
# u0 = (1/np.cosh(0.2*(x-Lx/2)))**2
#
#
# class BurgersSim(AbstractTimeSimulator):
#     def __init__(self, ic=None, dom=domain, bc="periodic", diff="finite_difference"):
#         super().__init__(ic, domain=dom, bc=bc, diff=diff)
#
#     def rhs(self, t, u_in):
#         u = u_in[0]
#         u_next = self.diff.d_dx(u, 2) - u * self.diff.d_dx(u, 1)
#
#         return np.array([u_next])

# dts = np.arange(0.01, 1.1, 0.01)
dts = np.arange(1.1, 2.8, 0.01)


# for i, dt in enumerate(np.arange(0.1, 3.0, 0.01)):
def process(i, ic=None, folder=None):
    dt = dts[i]
    # t_eval = np.linspace(time_interval[0], time_interval[-1], n_timesteps)
    t_eval = np.arange(time_interval[0], time_interval[-1], dt)

    # sim = BurgersSim(ic=np.array([u0]), dom=domain)
    # sim = CHESim(ic=np.array([u0]), dom=domain, params=[1.0, 1.0])
    sim = ReactionDiffusionSim(ic=ic, dom=domain, params=[1.0])
    sol = sim.simulate(time_interval, method="DOP853", t_eval=t_eval)  # , method="RK45", t_eval=t_eval)

    with open("tsme_examples/compare/data/"+folder+f"/rd_{i}.json", "w") as file:
        dic = {"sol": sol.tolist(), "time": sim.time.tolist(), "domain": sim.domain}
        rick.dump(dic, file)


for r in np.arange(10, 20, 1):
    Parallel(n_jobs=6)(delayed(process)(i, ic=np.array([u0, v0]),
                                        folder=f"ultra_batch/run_{r}/") for i in range(len(dts)))
    u0 = (np.random.random((N, N)) - 0.5) * 0.5
    v0 = (np.random.random((N, N)) - 0.5) * 0.5
