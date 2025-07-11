#!/usr/bin/env python
# coding: utf-8

# # FitzHugh-Nagumo Model
# 
# This notebook will produce example data for the FitzHugh-Nagumo Model and then identify the underlying partial differential equations.

# ## Time simulation
# 
# The `tsme` package provides a number of pre-implemented dynamical systems, so we simply import the class `FitzHughNagumo`. (See this [page](https://nonlinear-physics.zivgitlabpages.uni-muenster.de/ag-kamps/tsme/source/tsme.premade_models.html) for more models.)

# In[1]:

import sys

import numpy as np
import json as rick

from tsme.model_estimation import Model
from tsme.utils.visualization import animate



# The FitzHugh-Nagumo equation can be written as follows:
# $$
# \begin{align}
# f &= \lambda \, u - u^3 - \kappa \\
# \frac{\text{d}u}{\text{d}t} &= D_u^2 \, \nabla^2 u + f - \omega \, v \\
# \tau \, \frac{\text{d}v}{\text{d}t} &= D_v^2 \, \nabla^2 v + u - v,
# \end{align}
# $$
# with $u(x, y)$ and $v(x, y)$.
# 
# As we are dealing with a spatially extended 2D system we first set our spatial disctretization and domain sizes in $x$ and $y$ direction respectively.  Then we define a time interval as well as the time stamps at which we want to sample our trajectory. The initial condition is set to random noise between -0.25 and +0.25 for both fields.

with open("tsme_examples/data/PDEs/fhn.json", "r") as file:
    data = rick.load(file)

sol = np.array(data["sol"])
time = np.array(data["time"])
domain = data["domain"]


estimated_model = Model(sol, time, phys_domain=domain)
ode_order = 3
pde_order = 2
kind = "split"  # "pair" / "split"
norm = 0
estimated_model.init_library(ode_order, pde_order, kind=kind)
estimated_model.print_library()

estimated_model.init_simulator()

"""
estimated_model.least_square_sigma_over_time(norm=norm)

with open("tsme_examples/data/estimates/fhn" + f"_{ode_order}_{pde_order}_{kind}_n{norm}_OT.json", "w") as file:
    d = {"sigma_over_time": estimated_model.sigma_over_time.tolist(),
         "time": time[:-1].tolist(),
         "lib_strings": estimated_model.lib_strings.tolist(),
         "print_strings": estimated_model.print_strings}
    rick.dump(d, file)
"""



# estimated_model.optimize_sigma(lamb=0.001, thres=0.001, error="BIC")
# estimated_model.optimize_sigma(error="BIC", backend="hyperopt", max_evals=20, space="log")
estimated_model.sequential_threshold_ridge(thres=[0.08, 0.01])
estimated_model.print_library_to_file(f'tsme_examples/data/estimates/fhn_{ode_order}_{pde_order}_latex.txt', latex=True)
sigma_BIC = estimated_model.sigma

labels = estimated_model.print_strings
labels_latex = np.array(list((map(_label_to_latex, labels))))
ex = {1, 2, 6, 28, 29, 31, 32}
drop = list(set(range(0, 55)) - ex)
print(", ".join(labels_latex[drop]))


labels = estimated_model.print_strings
with open(f"tsme_examples/data/estimates/fhn_{ode_order}_{pde_order}.json", "w") as file:
    dic = {"sigma": estimated_model.sigma.tolist(), "labels": labels}
    rick.dump(dic, file)


# In[16]:


# from tsme.utils.visualization import barplot_parameters

# labels = estimated_model.print_strings
# barplot_parameters(sigma_BIC, sigma_ref=sigma_SINDy, labels=labels, figsize=(18, 10), table_fontsize=13)


# In[11]:

t = estimated_model.time
sol_est = estimated_model.simulator.simulate([t[0], t[-1]], method="DOP853", t_eval=t)

animate(sol_est, time=t, savepath=f"tsme_examples/diagrams/PDEs/fhn_{ode_order}_{pde_order}.mp4")

with open(f"tsme_examples/data/estimates/fhn_{ode_order}_{pde_order}_ts.json", "w") as file:
    dic = {"sol": sol_est.tolist(), "time": t.tolist(), "domain": estimated_model.domain}
    rick.dump(dic, file)



