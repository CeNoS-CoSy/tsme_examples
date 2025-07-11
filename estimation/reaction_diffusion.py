#!/usr/bin/env python
# coding: utf-8


import numpy as np
import sys

import json as rick
from tsme.model_estimation.model import Model
from tsme.utils.visualization import animate

from tsme.model_estimation.model import _label_to_latex


from tsme.utils.visualization import animate
from tsme.utils.visualization import barplot_parameters
import matplotlib.pyplot as plt


name = "rd_chaotic_alt"
with open("tsme_examples/data/PDEs/" + name + ".json", "r") as file:
    data = rick.load(file)

time = np.array(data["time"])
cut = np.argwhere(time > 7.5).T[0]

time = time[cut]
sol = np.array(data["sol"])[:, cut]  # [800:] for old chaotic_alt i.e. beta=4
domain = data["domain"]

estimated_model = Model(sol, time, phys_domain=domain)
ode_order = 3
pde_order = 2
kind = "split"  # "pair" / "split"
norm = 0
estimated_model.init_library(ode_order, pde_order, kind=kind)

#ex = {1, 2, 6, 7, 8, 9, 28, 29, 31, 32}
#drop = list(set(range(0, 55)) - ex)
#estimated_model.drop_library_terms(drop)

estimated_model.print_library()

estimated_model.init_simulator()

#estimated_model.least_square_sigma_over_time(norm=norm)

#with open("tsme_examples/data/estimates/" + name + f"_{ode_order}_{pde_order}_{kind}_n{norm}_OT.json", "w") as file:
#    d = {"sigma_over_time": estimated_model.sigma_over_time.tolist(),
#         "time": time[:-1].tolist(),
#         "lib_strings": estimated_model.lib_strings.tolist(),
#         "print_strings": estimated_model.print_strings}
#    rick.dump(d, file)


estimated_model.init_simulator()
# estimated_model.sequential_threshold_ridge(0.5, 0.5, norm=0, max_it=100)

# estimated_model.optimize_sigma(lamb=0.01, thres=0.1, error="SINDy", backend="train")
# sigma_SINDy = estimated_model.sigma

# estimated_model.optimize_sigma(lamb=0.01, thres=0.1, error="BIC")
# estimated_model.optimize_sigma(lamb=0.1, thres=0.5, norm=0, error="integrate", max_it_train=50)

# normal
#estimated_model.optimize_sigma(backend="hyperopt", max_evals=20, space={"type": "log", "bounds": [0, 0.08]},
#                                scale_threshold=True)

# chaotic
estimated_model.optimize_sigma(backend="hyperopt", max_evals=20, space={"type": "log", "bounds": [-10, 0.1]})

# chaotic_alt
#estimated_model.optimize_sigma(backend="hyperopt", max_evals=25, space={"type": "log", "bounds": [-10, 0.1]}
#                                , subset_threshold=[[28, 29], [31, 32]])  # [1, 2], [6, 7, 8, 9], [28, 29], [31, 32]]
# estimated_model.sequential_threshold_ridge(thres=[0.5, 0.5])


# estimated_model.optimize_sigma(lamb=0.01, thres=0.08, error="BIC")


sigma_BIC = estimated_model.sigma

estimated_model.print_library_to_file('tsme_examples/data/estimates/' + name + f'_{ode_order}_{pde_order}.txt')
labels = estimated_model.print_strings


#labels_latex = np.array(list((map(_label_to_latex, labels))))
#ex = {1, 2, 6, 7, 8, 9, 28, 29, 31, 32, 44, 47, 50, 53}
#drop = list(set(range(0, 55)) - ex)
#print(", ".join(labels_latex[drop]))

# with open("tsme_examples/data/estimates/che_3_4_SINDy.json", "w") as file:
#     dic = {"sigma": sigma_SINDy.tolist(), "labels": labels, "lib_strings": estimated_model.lib_strings.tolist()}
#     rick.dump(dic, file)

with open("tsme_examples/data/estimates/" + name + f"_{ode_order}_{pde_order}.json", "w") as file:
    dic = {"sigma": sigma_BIC.tolist(), "labels": labels, "lib_strings": estimated_model.lib_strings.tolist()}
    rick.dump(dic, file)


t = estimated_model.time
sol_est = estimated_model.simulator.simulate([t[0], t[-1]], method="DOP853", t_eval=t)

animate(sol_est, time=t, savepath=f"tsme_examples/diagrams/PDEs/rd_chaotic_{ode_order}_{pde_order}.mp4")

with open(f"tsme_examples/data/estimates/rd_chaotic_{ode_order}_{pde_order}_ts.json", "w") as file:
    dic = {"sol": sol_est.tolist(), "time": t.tolist(), "domain": estimated_model.domain}
    rick.dump(dic, file)

