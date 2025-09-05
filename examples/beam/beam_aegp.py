import matplotlib.pyplot as plt
import numpy as np

# local imports
import logging
import datetime
import time
import tensorflow as tf
import sys
import os
from vindy import VENI
from vindy.libraries import PolynomialLibrary, ForceLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Gaussian, Laplace
from vindy.callbacks import (
    SaveCoefficientsCallback,
)
from utils import load_beam_data, plot_train_history, plot_coefficients_train_history

# Add the examples folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import config
import config

# tf.config.run_functions_eagerly(True)  # uncomment this line for debugging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

"""
BEAM MODEL
    dzdt    = z
    dzddt   = -w0^2 * z - 2 * xi * w0 * z_dot - gamma * z^3 + u 
            = - 0.29975625 * z - 0.01095 z_dot - gammma* z^3 + u
"""

logging.info(
    "################################   1. Loading    ################################"
)

# %% script parameters
# Load config
model_name = "beam"
identification_layer = "vindy"  # 'vindy' or 'sindy'
# Script parameter
reduced_order = 1
pca_order = 3
noise = True
nth_time_step = 8
second_order = True

beta_vindy = 1e-8  # 5e-9
beta_vae = 1e-8  # 1e-8
l_rec = 1e-3  # 1e-3
l_dz = 1e0  # 1e0
l_dx = 1e-5  # 1e-5
end_time_step = 14000

# this scripts path + results
result_dir = os.path.join(os.path.dirname(__file__), "results")

if noise:
    # for storage reasons we just load the pca components and not the full noisy data
    (
        t,
        params,
        x,
        dxdt,
        dxddt,
        t_test,
        params_test,
        x_test,
        dxdt_test,
        dxddt_test,
        ref_coords,
        V,
        n_sims,
        n_timesteps,
    ) = load_beam_data(
        config.beam["processed_data"],
        end_time_step=end_time_step,
        nth_time_step=nth_time_step,
        pca_order=pca_order,
    )
else:
    (
        t,
        params,
        param_int,
        f,
        x,
        dxdt,
        dxddt,
        x_int,
        dx_int,
        t_test,
        params_test,
        param_int_test,
        f_test,
        x_test,
        dxdt_test,
        dxddt_test,
        x_int_test,
        dx_int_test,
        V,
        n_sims,
        n_timesteps,
    ) = load_data(
        data_paths=config.beam_data,
        train_test_ratio=0.7,
        nth_time_step=nth_time_step,
        end_time_step=end_time_step,
        n_int=0,
        pca_order=pca_order,
    )

n_timesteps_test = x_test.shape[0] // n_sims
n_dof = x.shape[1]
dt = t[1] - t[0]

# %% create forcing terms from parameters
param_libraries = ForceLibrary(functions=[tf.cos])
params = param_libraries(params).numpy()
params_test = param_libraries(params_test).numpy()

# %% Create Model
logging.info(
    "################################   2. Training    ################################"
)
logging.info("Creating model")

libraries = [PolynomialLibrary(3)]
param_libraries = [ForceLibrary(functions=[tf.cos])]

# %% MORML
import morml as mm
from morml.methods import AutoRegressive, Dynamic, Generic
from morml.reduction import (
    Autoencoder,
    KernelPCA,
    POD,
    PCA,
    CUR,
    VariationalAutoencoder,
)

n_sims_train = 24
n_sims_val = n_sims - n_sims_train
x_train, dxdt_train, dxddt_train, params_train, t_train = [
    x[: n_sims_train * n_timesteps],
    dxdt[: n_sims_train * n_timesteps],
    dxddt[: n_sims_train * n_timesteps],
    params[: n_sims_train * n_timesteps],
    t[: n_sims_train * n_timesteps],
]
x_val, dxdt_val, dxddt_val, params_val, t_val = [
    x[n_sims_train * n_timesteps :],
    dxdt[n_sims_train * n_timesteps :],
    dxddt[n_sims_train * n_timesteps :],
    params[n_sims_train * n_timesteps :],
    t[n_sims_train * n_timesteps :],
]

reduction = mm.reduction.Autoencoder(
    reduced_order=reduced_order,
    shape=x_train.shape,
    X=x_train,
    scaler=None,
    hidden_layer_sizes=[32, 32, 32],
)


logging.info("Reduction")
# save weights of the autoencoder
try:
    reduction.autoencoder.build(input_shape=[x_train.shape])
    reduction.load_weights(os.path.join(result_dir, "beam_ae.weights.h5"))
except:
    train_history = reduction.fit(
        X=x_train,
        epochs=2500,
        batch_size=int(n_timesteps / nth_time_step),
        validation_data=x_val,
        verbose=2,
    )
    reduction.save_weights(os.path.join(result_dir, "beam_ae.weights.h5"))


method = mm.methods.Dynamic(reduction, None)

X_train = x_train.reshape((n_sims_train, n_timesteps, n_dof))
Params_train = params_train.reshape((n_sims_train, n_timesteps, -1))
T_train = t_train.reshape((n_sims_train, n_timesteps))
regression_input, z, _, regression_input_list, z_list, _ = method.dataset(
    simulation_data=X_train,
    parameter=Params_train,
    time_vectors=T_train,
)

n_sims_test = int(x_test.shape[0] / n_timesteps_test)
X_test = x_test.reshape((n_sims_test, n_timesteps_test, n_dof))
Params_test = params_test.reshape((n_sims_test, n_timesteps_test, -1))
T_test = t_test.reshape((n_sims_test, n_timesteps_test))
regression_input_test, z_test, _, regression_input_test_list, z_test_list, _ = (
    method.dataset(
        simulation_data=X_test,
        parameter=Params_test,
        time_vectors=T_test,
    )
)

i_sim = 0
# nn = mm.regression.NN(regression_input, z, [32, 32, 32])
# train_hist = nn.train(epochs=100, scaler="minmax", output_scaler="maxabs")
# mse, score = nn.evaluate(regression_input, z)
# logging.info("Train: The regression scored r2_score = %.4f", score)
# mse, score = nn.evaluate(regression_input_test_list[i_sim], z_test_list[i_sim])
# logging.info("Test: The regression scored r2_score = %.4f", score)
#
# method.set_algorithm(nn)

logging.info("Fit regression")
gp = mm.regression.GPR(regression_input, z, kernel="matern")
train_hist = gp.train(scaler="minmax", output_scaler="maxabs")

# mse, score = gp.evaluate(regression_input, z)
# logging.info("Train: The regression scored r2_score = %.4f", score)

mse, score = gp.evaluate(regression_input_test_list[i_sim], z_test_list[i_sim])
logging.info("Test: The regression scored r2_score = %.4f", score)

method.set_algorithm(gp)
# set regression


# %% inference
def predict_single_sample(self, x):
    """
    scale input and make prediction based on it
    :param x: element from reduced/latent space for which the prediction is to be made
    :return: prediction based on x
    """
    if self.scaler:
        if x.ndim == 3:
            x_ = []
            for i in range(x.shape[1]):
                x_.append(np.expand_dims(self.scaler.transform(x[:, i, :]), axis=1))
            x = np.concatenate(x_, axis=1)
        else:
            x = self.scaler.transform(np.atleast_2d(x))
    y, y_std = self.regressor.predict(x, return_std=True)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=1)
    if self.output_scaler:
        if y.ndim > 2:
            y = self.output_scaler.inverse_transform(
                y.reshape(y.shape[0] * y.shape[1], -1)
            )
            y = y.reshape((x.shape[0], x.shape[1], -1))
        else:
            y = self.output_scaler.inverse_transform(y)
    if y_std.ndim == 1:
        y_std = np.expand_dims(y_std, axis=1)
    if self.output_scaler:
        if y_std.ndim > 2:
            y_std = self.output_scaler.inverse_transform(
                y_std.reshape(y.shape[0] * y_std.shape[1], -1)
            )
            y_std = y_std.reshape((x.shape[0], x.shape[1], -1))
        else:
            y_std = self.output_scaler.inverse_transform(y)
    return y, y_std


for i_sim in range(n_sims_test):
    z_pred, z_std = predict_single_sample(gp, regression_input_test_list[i_sim])

    # plot prediction with uncertainty
    fig, ax = plt.subplots(1, 1)
    ax.plot(z_test_list[i_sim][:, 0], "gray", label="True")
    ax.plot(z_pred, "r--", label="Predicted")
    ax.fill_between(
        np.arange(len(z_pred)),
        (z_pred - 2 * z_std).squeeze(),
        (z_pred + 2 * z_std).squeeze(),
        color="r",
        alpha=0.3,
        zorder=3,  # Higher zorder to bring it to the foreground
    )
    ax.legend()
    plt.ylim([-50, 50])
    ax.set_title("Latent variable")
    # save figure
    plt.savefig(os.path.join(result_dir, f"beam_gp_prediction_simulation_{i_sim}.png"))

# # test
# z_pred, z_std = method.regression.regressor.predict(
#     regression_input_test_list[i_sim], return_std=True
# )
#
# # plot prediction with uncertainty
# fig, ax = plt.subplots(1, 1)
# ax.plot(z_pred, "r--", label="Predicted")
# ax.fill_between(
#     np.arange(len(z_pred)),
#     z_pred - 2 * z_std,
#     z_pred + 2 * z_std,
#     color="r",
#     alpha=0.3,
#     zorder=3,  # Higher zorder to bring it to the foreground
# )
# ax.plot(z_test_list[i_sim][:, 0], "gray", label="True")
# ax.legend()
# ax.set_title("Latent variable")
# plt.show()
