import numpy as np
from tsme.model_estimation.model import Model
import pysindy as ps
# Note:
# pysindy version: pysindy-1.7.6.dev463+g3e8a445
# numpy version: numpy-1.24.0
# scikit-learn version: scikit-learn-1.3.1
import json

from joblib import Parallel, delayed

from tsme_examples.compare.timeout import timeout

import warnings
warnings.filterwarnings('error')

# all_time_steps = np.arange(10, 310, 10)
indices = np.arange(0, 170)  # 38, 109, 290
true_par = np.zeros((2, 110))
# udot
true_par[0, 18] = 0.1  # dx**2 u
true_par[0, 12] = 0.1  # dy**2 u
true_par[0, 1] = 1.0  # u (from (1-A)*u)
true_par[0, 6] = -1.0  # u**3 (from (1-A)*u)
true_par[0, 8] = -1.0  # u*v**2 (from (1-A)*u)
true_par[0, 7] = 1.0  # vYfrom A*v)
true_par[0, 9] = 1.0  # v**3 (from A*v)
# vdot
true_par[1, 19] = 0.1  # dx**2 v
true_par[1, 13] = 0.1  # dy**2 v
true_par[1, 2] = 1.0  # v (from (1-A)*v)
true_par[1, 9] = -1.0  # v**3 (from (1-A)*u)
true_par[1, 7] = -1.0  # v*u**2 (from (1-A)*u)
true_par[1, 8] = 1.0  # u*v**2 (from A*u)
true_par[1, 6] = 1.0  # u**3 (from A*u)

solver = "DOP853"


# TODO: combine metrics into dicts
# dts = []
#
# error_tsme = []
# error_par_tsme = []
# error_pysindy = []
# error_par_pysindy = []
#

def process(i, folder=None):
    #f"tsme_examples/compare/data/batch_2/rd_{i}.json"
    with open("tsme_examples/compare/data/ultra_batch/"+folder+f"/rd_{i}.json") as file:
        data = json.load(file)

    sol = np.array(data["sol"])[:, :200]
    time = np.array(data["time"])[:200]
    domain = np.array(data["domain"])
    del data
    out = {}

    # Do TSME
    mod = Model(sol, time, phys_domain=domain)
    # Make the library as the one in PDE-FIND
    pde_order = 2
    # us = ["u[0]", "u[1]", "u[0]*u[0]*u[0]", "u[1]*u[1]*u[1]", "u[0]*u[1]*u[1]", "u[0]*u[0]*u[1]"]
    us = ["u[0]", "u[1]", "u[0]*u[0]", "u[0]*u[1]", "u[1]*u[1]", "u[0]*u[0]*u[0]", "u[0]*u[0]*u[1]", "u[0]*u[1]*u[1]",
          "u[1]*u[1]*u[1]"]
    dyus = [f"self.diff.d_dy(u[{i}],{der})" for der in range(1, pde_order + 1) for i in range(0, 2)]
    dxus = [f"self.diff.d_dx(u[{i}],{der})" for der in range(1, pde_order + 1) for i in range(0, 2)]
    dxyus = [f"self.diff.dd_dxdy(u[{i}],[{dx}, {dy}])" for dx in range(1, pde_order + 1) for dy in
             range(1, pde_order + 1 - dx)
             for i in range(0, 2)]
    dxyus_dxus = dxus[0:2] + dxyus + dxus[2:]
    mix = [f"{u}*{d}" for d in dyus + dxyus_dxus for u in us]
    terms = us + dyus + dxyus_dxus + mix

    mod.init_library(0, 0, custom_terms=terms)
    mod.print_library()

    mod.init_simulator()
    mod.sequential_threshold_ridge(thres=[0.08, 0.08])
    # mod.optimize_sigma(backend="hyperopt", simulate=False, max_evals=20, space="log", adapt_threshold=True)
    # mod.optimize_sigma(backend="train", error="SINDy", simulate=True, lamb=1e-5, thres=1.0)
    @timeout(900)
    def wrapped_simulate():
        return mod.simulator.simulate([time[0], time[-1]], t_eval=time, method=solver)
    try:
        estimate = wrapped_simulate()
        if not (len(estimate[0]) == len(sol[0])):
            rmse = np.linalg.norm(estimate - sol[:, :len(estimate[0])]) / np.sqrt(len(sol[0]))
        else:
            rmse = np.linalg.norm(estimate - sol) / np.sqrt(len(sol[0]))
    except:
        estimate = np.nan
        rmse = np.nan

    # error_tsme.append(rmse)
    # error_par_tsme.append(np.linalg.norm(mod.sigma - true_par))
    out["error_tsme"] = rmse
    out["error_par_tsme"] = np.linalg.norm(mod.sigma - true_par)

    # Do PDE-FIND
    N = sol.shape[-1]
    x = np.linspace(domain[0][0], domain[0][1], N)  # spatial coordinates
    y = np.linspace(domain[1][0], domain[1][1], N)
    x_mesh, y_mesh = np.meshgrid(x, y)
    # spatial_grid = np.linspace(domain[0, 0], domain[0, 1], sol.shape[-1])
    spatial_grid = np.asarray([x_mesh, y_mesh]).T
    dt = time[1] - time[0]
    # dts.append(dt)
    out["dts"] = dt
    sol_ps = sol.T

    # library_functions = [
    #     lambda x: x,
    #     lambda x: x * x * x,
    #     lambda x, y: x * y * y,
    #     lambda x, y: x * x * y,
    # ]
    #
    # library_function_names = [
    #     lambda x: x,
    #     lambda x: x + x + x,
    #     lambda x, y: x + y + y,
    #     lambda x, y: x + x + y,
    # ]

    pde_lib = ps.PDELibrary(
        # library_functions=library_functions,
        # function_names=library_function_names,
        function_library=ps.PolynomialLibrary(degree=3, include_bias=False),
        derivative_order=2,
        spatial_grid=spatial_grid,
        include_bias=True,
        is_uniform=True,
        periodic=True
    )

    print('STLSQ model: ')
    # optimizer = ps.STLSQ(threshold=1.0, alpha=1e-5, normalize_columns=True)
    optimizer = ps.STLSQ(threshold=10, alpha=1e-4,
                         normalize_columns=True, max_iter=200)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(sol_ps, t=dt)
    model.print()

    pysindy_par = model.coefficients()
    # error_par_pysindy.append(np.linalg.norm(pysindy_par - true_par))
    out["error_par_pysindy"] = np.linalg.norm(pysindy_par - true_par)

    mod.simulator.sigma = pysindy_par
    try:
        estimate_pysindy = mod.simulator.simulate([time[0], time[-1]], t_eval=time, method=solver)
        if not (len(estimate_pysindy[0]) == len(sol[0])):
            rmse_pysindy = np.linalg.norm(estimate_pysindy - sol[:, :len(estimate_pysindy[0])]) / np.sqrt(len(sol[0]))
        else:
            rmse_pysindy = np.linalg.norm(estimate_pysindy - sol) / np.sqrt(len(sol[0]))
    except:
        estimate_pysindy = np.nan
        rmse_pysindy = np.nan

    # error_pysindy.append(rmse_pysindy)
    out["error_pysindy"] = rmse_pysindy

    #f"tsme_examples/compare/data/batch_ident/batch_2/rd_{i}.json"
    with open("tsme_examples/compare/data/ultra_batch/"+folder+f"/id_rd_{i}.json", "w") as file:
        json.dump(out, file)
    return out


"""
data_ps = loadmat('tsme_examples/compare/data/burgers.mat')
t = np.ravel(data_ps['t'])
x = np.ravel(data_ps['x'])
u = np.real(data_ps['usol'])
u = u.reshape(len(x), len(t), 1)
"""
Parallel(n_jobs=3)(delayed(process)(i, folder=f"run_{10}") for i in indices)
#for r in range(10, 20):
#    Parallel(n_jobs=3)(delayed(process)(i, folder=f"run_{r}") for i in indices)
# results = Parallel(n_jobs=6)(delayed(process)(i) for i in indices)
# dts = [r["dts"] for r in results]
#
# error_tsme = [r["error_tsme"] for r in results]
# error_par_tsme = [r["error_par_tsme"] for r in results]
# error_pysindy = [r["error_pysindy"] for r in results]
# error_par_pysindy = [r["error_par_pysindy"] for r in results]
#
# # TODO: save results to file and plot somewhere else
# plt.figure()
# plt.plot(dts, error_tsme, label="TSME")
# plt.plot(dts, error_pysindy, label="PySINDy")
# plt.xlabel("time step size")
# plt.ylabel("RMSE estimated time series")
# plt.legend()
# plt.savefig("tsme_examples/compare/RMSE_vs_time_step_size_rd.png")
# plt.show()
#
# plt.figure()
# plt.plot(dts, error_par_tsme, label="TSME")
# plt.plot(dts, error_par_pysindy, label="PySINDy")
# plt.xlabel("time step size")
# plt.ylabel("L2-error in model parameters")
# plt.legend()
# plt.savefig("tsme_examples/compare/Par_vs_time_step_size_rd.png")
# plt.show()
