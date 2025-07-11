import numpy as np
from tsme.time_simulation import AbstractTimeSimulator
from tsme.utils.visualization import animate
import json as rick

np.random.seed(12389)

N = 128
Lx = 90
Ly = 90
domain = ((0, Lx), (0, Ly))
time_interval = [0, 15]
t_eval = np.linspace(time_interval[0], time_interval[-1], 100)

u0 = (np.random.random((N, N)) - 0.5) * 0.5


class CHESim(AbstractTimeSimulator):
    def __init__(self, ic=None, dom=domain, params=[1.0, 1.0], bc="periodic", diff="finite_difference"):
        super().__init__(ic, domain=dom, bc=bc, diff=diff)
        self.a = params[0]
        self.b = params[1]

    def rhs(self, t, u_in):
        u = u_in[0]
        # f = u ** 3 - u - self.a * (self.diff.d_dx(u, 2) + self.diff.d_dy(u, 2))
        # u_next = self.diff.d_dx(f, 2) + self.diff.d_dy(f, 2)
        u_next = - self.diff.d_dx(u, 2) - self.diff.d_dy(u, 2) \
                 + self.diff.d_dx(u ** 3, 2) + self.diff.d_dy(u ** 3, 2) \
                 - self.a * self.diff.d_dx(u, 4) - self.a * self.diff.d_dy(u, 4)\
                 - self.a * 2 * self.diff.dd_dxdy(u, [2, 2])
                 # - self.a * 2 * self.diff.d_dx(self.diff.d_dy(u, 2), 2)

        return np.array([u_next])


sim = CHESim(ic=np.array([u0]), dom=domain, params=[1.0, 1.0])
sol = sim.simulate(time_interval, method="DOP853")  # , method="RK45", t_eval=t_eval)

# animate(sol, savepath="tsme_examples/diagrams/PDEs/che.mp4", time=sim.time)

with open("tsme_examples/data/time_series/che.json", "w") as file:
    dic = {"sol": sol.tolist(), "time": sim.time.tolist(), "domain": sim.domain}
    rick.dump(dic, file)
