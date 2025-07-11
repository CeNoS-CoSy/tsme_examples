from tsme.time_simulation import AbstractTimeDelaySimulator
import numpy as np
import matplotlib.pyplot as plt
from tsme.utils.visualization import animate

import json


N = 128
Lx = 90
domain = ((0, Lx),)
time_interval = [0, 8]
t = np.linspace(time_interval[0], time_interval[-1], 300)
x = np.linspace(domain[0][0], domain[0][1], N)

u0 = (1 / np.cosh(0.2 * (x - Lx / 2))) ** 2


class FisherKPP(AbstractTimeDelaySimulator):
    def __init__(self, delay, bc="periodic", diff="finite_difference", pars=None):
        super().__init__(delay, domain=domain, bc=bc, diff=diff)
        if pars is not None:
            self.D = pars[0]  # Diffusion constant
            self.r = pars[1]  # intrinsic growth rate
            self.K = pars[2]  # carrying capacity
        else:
            self.D = 1.0  # 0.01 to 10.0
            self.r = 1.0  # 0.1 to 2.0
            self.K = 1.0  # 1.0 to 100.0

    def values_before_zero(self, t):
        return np.array([u0])

    def model(self, u, u_d, t):
        du = self.D * self.diff.d_dx(u[0], 2) + self.r * u[0] * (1 - u_d[0] / self.K)
        return np.array([du])


model_fisher = FisherKPP(1)
sol_fisher = model_fisher.simulate_delay_model(t)
with open("tsme_examples/data/time_series/FisherKPP.json", "w") as file:
    dic = {"sol": sol_fisher.tolist(), "time": t.tolist(), "domain": domain}
    json.dump(dic, file)


# animate(sol_fisher, time=t, savepath="tsme_examples/diagrams/fisherKPP.mp4")
