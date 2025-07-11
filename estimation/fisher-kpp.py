import numpy as np
import json as rick
from tsme.model_estimation.model import Model
from tsme.utils.visualization import animate

with open("tsme_examples/data/DDEs/FisherKPP.json", "r") as file:
    data = rick.load(file)

sol = np.array(data["sol"])
time = np.array(data["time"])
domain = data["domain"]

mod = Model(sol, time, phys_domain=domain, delay=1)
mod.init_library(3, 4)
# mod.drop_library_terms([8, 14])
mod.add_library_terms(["u[0] * u_d[0]"])
mod.print_library()

mod.init_simulator()

# mod.optimize_sigma(backend="hyperopt", with_delay=False, max_evals=2000)
# "delay": {"type": "randint", "bounds": [0, 5]}
space = {"type": "uniform", "bounds": [0.001, 1.0], "delay": {"type": "uniform", "bounds": [0.001, 2.0]}}
# mod.optimize_sigma(backend="hyperopt", with_delay=True, max_evals=10, space=space, par_guess=3)
mod.optimize_sigma(backend="hyperopt", with_delay=True, max_evals=10, space=space)

mod.print_library_to_file("tsme_examples/data/estimates/FisherKPP.txt", append_delay=True)

sol_est = mod.simulator.simulate_delay_model(time)
animate(sol_est, time=time, savepath="tsme_examples/diagrams/FisherKPP_estimated.mp4")

with open(f"tsme_examples/data/estimates/FisherKPP.json", "w") as file:
    dic = {"sol": sol_est.tolist(), "time": time.tolist(), "domain": mod.domain}
    rick.dump(dic, file)