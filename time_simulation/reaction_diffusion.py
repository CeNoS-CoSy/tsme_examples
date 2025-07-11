import numpy as np
from tsme.time_simulation import AbstractTimeSimulator
from tsme.utils.visualization import animate
import json as rick

np.random.seed(12389)

N = 128
Lx = 20
Ly = 20
domain = ((-Lx/2, Lx/2), (-Ly/2, Ly/2))
time_interval = [0, 18]
t_eval = np.linspace(time_interval[0], time_interval[-1], int(18 / 0.05))

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

u0 = (np.random.random((N, N)) - 0.5) * 0.5
v0 = (np.random.random((N, N)) - 0.5) * 0.5

class ReactionDiffusionSim(AbstractTimeSimulator):
    def __init__(self, ic=None, dom=domain, params=[1.0, 0.0, 1.0], bc="periodic", diff="finite_difference"):
        super().__init__(ic, domain=dom, bc=bc, diff=diff)
        self.D_1 = params[0]
        self.D_2 = params[1]
        self.beta = params[2]

    def rhs(self, t, u_in):
        u = u_in[0]
        v = u_in[1]
        A = u**2 + v**2

        u_next = (self.D_1*(self.diff.d_dx(u, 2) + self.diff.d_dy(u, 2)) -
                  self.D_2*(self.diff.d_dx(v, 2) + self.diff.d_dy(v, 2)) +
                  (1 - A)*u + self.beta*A*v)
        v_next = (self.D_1*(self.diff.d_dx(v, 2) + self.diff.d_dy(v, 2)) +
                  self.D_2*(self.diff.d_dx(u, 2) + self.diff.d_dy(u, 2)) -
                  self.beta*A*u + (1 - A)*v)

        return np.array([u_next, v_next])

# default params: [0.1, 0.0, 1.0]
# chaotic params: [1.0, 2.0, -1] // 1.0, 2.0, -4.0
# chaotic_alt params: [1.0, 0.0, 4]
sim = ReactionDiffusionSim(ic=np.array([u0, v0]), dom=domain, params=[1, 0.0, -4])

sol = sim.simulate(time_interval, method="DOP853")  # , method="RK45", t_eval=t_eval)
sol_dot = sim.get_sol_dot()

# animate(sol, savepath="tsme_examples/diagrams/PDEs/rd_u_chaotic_alt.mp4", variable=0)
# animate(sol, savepath="tsme_examples/diagrams/PDEs/rd_v_chaotic_alt.mp4", variable=1)


with open("tsme_examples/data/time_series/rd_chaotic_alt.json", "w") as file:
    dic = {"sol": sol.tolist(), "time": sim.time.tolist(), "domain": sim.domain}
    rick.dump(dic, file)
