import numpy as np
import sys

import json as rick
from tsme.model_estimation.model import Model
from tsme.utils.visualization import animate


from tsme.utils.visualization import animate
from tsme.utils.visualization import barplot_parameters
import matplotlib.pyplot as plt


with open("tsme_examples/data/PDEs/ace.json", "r") as file:
    data = rick.load(file)

sol = np.array(data["sol"])
time = np.array(data["time"])
domain = data["domain"]


estimated_model = Model(sol, time, phys_domain=domain)
ode_order = 3
pde_order = 4
kind = "split"  # "pair" / "split"
norm = 0
estimated_model.init_library(ode_order, pde_order, kind=kind)
estimated_model.print_library()

estimated_model.init_simulator()

estimated_model.least_square_sigma_over_time(norm=norm)

with open("tsme_examples/data/estimates/ace" + f"_{ode_order}_{pde_order}_{kind}_n{norm}_OT.json", "w") as file:
    d = {"sigma_over_time": estimated_model.sigma_over_time.tolist(),
         "time": time[:-1].tolist(),
         "lib_strings": estimated_model.lib_strings.tolist(),
         "print_strings": estimated_model.print_strings}
    rick.dump(d, file)



# Of course just after initialization the linear factors of all these library terms are yet to be determined. We tell our model to construct a simulator with the above library that can facilitate time simulations. (The library supports some basic manipulations at this point, but more on that in a future tutorial). Now we can call the optimization routine to find the best combination of our trial functions. (For more detail see [here](https://nonlinear-physics.zivgitlabpages.uni-muenster.de/ag-kamps/tsme/source/tsme.html#tsme.model_estimation.model.Model.optimize_sigma))

estimated_model.init_simulator()
# estimated_model.sequential_threshold_ridge(0.5, 0.5, norm=0, max_it=100)

# estimated_model.optimize_sigma(lamb=0.01, thres=0.1, error="SINDy", backend="train")
# sigma_SINDy = estimated_model.sigma

# estimated_model.optimize_sigma(lamb=0.01, thres=0.1, error="BIC")
# estimated_model.optimize_sigma(lamb=0.1, thres=0.5, norm=0, error="integrate", max_it_train=50)
estimated_model.optimize_sigma(error="BIC", backend="hyperopt", max_evals=10)
sigma_BIC = estimated_model.sigma

estimated_model.print_library_to_file(f'tsme_examples/data/estimates/ace_{ode_order}_{pde_order}.txt')

#original_stdout = sys.stdout  # Save a reference to the original standard output
#with open(f'tsme_examples/data/estimates/che_{ode_order}_{pde_order}.txt', 'w') as f:
#    sys.stdout = f  # Change the standard output to the file we created.
#    estimated_model.print_library()
#    sys.stdout = original_stdout  # Reset the standard output to its original value

labels = estimated_model.print_strings
#labels_latex = np.array(list((map(_label_to_latex, labels))))
#ex = {10, 11, 16, 17, 31, 32, 35}
#drop = list(set(range(0, 46)) - ex)
#print(", ".join(labels_latex[drop]))


# with open("tsme_examples/data/estimates/che_3_4_SINDy.json", "w") as file:
#     dic = {"sigma": sigma_SINDy.tolist(), "labels": labels, "lib_strings": estimated_model.lib_strings.tolist()}
#     rick.dump(dic, file)

with open(f"tsme_examples/data/estimates/ace_{ode_order}_{pde_order}.json", "w") as file:
    dic = {"sigma": sigma_BIC.tolist(), "labels": labels, "lib_strings": estimated_model.lib_strings.tolist()}
    rick.dump(dic, file)


t = estimated_model.time
sol_est = estimated_model.simulator.simulate([t[0], t[-1]], method="DOP853", t_eval=t)

animate(sol_est, time=t, savepath=f"tsme_examples/diagrams/PDEs/ace_{ode_order}_{pde_order}.mp4")

with open(f"tsme_examples/data/estimates/ace_{ode_order}_{pde_order}_ts.json", "w") as file:
    dic = {"sol": sol_est.tolist(), "time": t.tolist(), "domain": estimated_model.domain}
    rick.dump(dic, file)

