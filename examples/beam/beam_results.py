import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pysindy as ps
import os

# local imports
import logging
import datetime
from keras import backend as K
from aesindy import VAESindy, CAESindy, AutoencoderSindy, config
from aesindy.libraries import (
    PolynomialLibrary,
    ForceLibrary,
    FourierLibrary,
    ExponentialLibrary,
)
from sklearn.decomposition import PCA
from aesindy.layers import SindyLayer, VariationalSindyLayer
from aesindy.layers.distributions import Gaussian, Laplace
from aesindy.callbacks import (
    SindyCallback,
    SaveCoefficientsCallback,
    PDFThresholdCallback,
)
from preprocess_data import load_beam_data
from aesindy.utils import *
from visualizer import Visualizer

#
# tf.config.run_functions_eagerly(True)  # uncomment this line for debugging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
"""
BEAM MODEL
    dzdt = z
    dzddt = -w0^2 * z + 5e-2 * z^2 + u = - 0.29975625 * z + 0.5475 * z^2
"""

logging.info(
    "################################   1. Loading    ################################"
)
# %% script parameters
# Load config
model_name = "beam"
test = "vindy"  # 'vindy' or 'sindy'
# Script parameter
reduced_order = 1
pca_order = 3
convolutional = False
numerical_derivatives = True
noise = True
nth_time_step = 3

beta_vindy = 1e-8  # 5e-9
beta_vae = 1e-8  # 1e-8
l_rec = 1e-3  # 1e-3
l_dz = 1e0  # 1e0
l_dx = 1e-5  # 1e-5
end_time_step = 14000

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

n_timesteps_test = x_test.shape[0] // n_sims
n_dof = x.shape[1]


# %% Create Model
logging.info(
    "################################   2. Training    ################################"
)
logging.info("Creating model")

libraries = [
    PolynomialLibrary(3),
]
param_libraries = [
    ForceLibrary(functions=[tf.cos])
    # PolynomialLibrary(1, include_bias=False),
]
dt = t[1] - t[0]

second_order = True
kernel_regularizer = tf.keras.regularizers.L1L2(l1=1e-8, l2=0)

# create sindy layer
layer_params = dict(
    state_dim=reduced_order,
    param_dim=params.shape[1],
    feature_libraries=libraries,
    second_order=second_order,
    param_feature_libraries=param_libraries,
    x_mu_interaction=False,
    kernel_regularizer=kernel_regularizer,
    mask=None,
    fixed_coeffs=None,
)
if test == "vindy":
    sindy_layer = VariationalSindyLayer(
        beta=beta_vindy,
        priors=Laplace(0.0, 1.0),
        # priors=Laplace(0., 1.),
        **layer_params,
    )
elif test == "sindy":
    sindy_layer = SindyLayer(**layer_params)
else:
    raise ValueError('test must be either "vindy" or "sindy"')

aesindy = VAESindy(
    sindy_layer=sindy_layer,
    beta=beta_vae * reduced_order / n_dof,
    reduced_order=reduced_order,  # 1e-3
    x=x,
    mu=params,
    scaling="individual_sqrt",
    second_order=second_order,
    layer_sizes=[32, 32, 32],
    activation="elu",
    l_rec=l_rec,
    l_dz=l_dz,
    l_dx=l_dx,
    l1=0,
    l2=0,
    l_int=0,
    dt=dt,
)

# learning rate scheduler
aesindy.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
    # sindy_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3),
    loss="mse",
)

### Scale data
aesindy.define_scaling(x)
x_train_scaled, dxdt_train_scaled, dxddt_train_scaled = (
    aesindy.scale(x),
    aesindy.scale(dxdt),
    aesindy.scale(dxddt),
)
x_test_scaled, dxdt_test_scaled, dxddt_test_scaled = (
    aesindy.scale(x_test),
    aesindy.scale(dxdt_test),
    aesindy.scale(dxddt_test),
)

# %% Train model
weights_path = os.path.join(
    config.outdir,
    f"{model_name}/{model_name}_{reduced_order}_{aesindy.__class__.__name__}_{test}.weights.h5",
)
outdir = os.path.join(config.outdir, f"{model_name}")
if not outdir:
    os.mkdir(outdir)

aesindy.load_weights(os.path.join(weights_path))

# Coefficient Posterior Distributions
coefficient_distributions_to_csv(
    sindy_layer, outdir, var_names=["z", r"\stackrel{.}{z}"]
)

# get current
_, loc, log_scale = sindy_layer._coeffs
feature_names = np.array(
    [sindy_layer.get_feature_names()] * sindy_layer.state_dim
).flatten()
# cancel coefficients
for i, (loc_, log_scale_) in enumerate(zip(loc[:-1], log_scale[:-1])):
    # plot the distribution of the coefficients
    if isinstance(sindy_layer.priors, list):
        distribution = sindy_layer.priors[i]
    else:
        distribution = sindy_layer.priors
    scale = distribution.reverse_log(log_scale_)
    zero_density = distribution.prob_density_fcn(x=0, loc=loc_, scale=scale)
    if zero_density > 0.5:
        # cancel the coefficient
        loc[i].assign(0)
        log_scale[i].assign(-10)
        logging.info(f"Canceling coefficient {feature_names[i]}")
aesindy.print(z=["z", "dz"], precision=6)


# %% create faces for beam
# for each point find points within radius
# create faces
faces = []
import itertools

step_size = 2.675520
ref_coords_adj = ref_coords
ref_coords_adj[:, 0] = ref_coords_adj[:, 0] / step_size
ref_coords_adj[:, 1] = ref_coords_adj[:, 1] / 10 * 5
ref_coords_adj[:, 2] = ref_coords_adj[:, 2] / 24 * 3

# find nodes on edges of the beam, i.e. with (*, 0, 0) or (*, 0, 3) (*, 5, 0) or (*, 5, 3)
edge1 = np.where((ref_coords_adj[:, 1] == 0) & (ref_coords_adj[:, 2] == 0))[0]
edge2 = np.where(
    (ref_coords_adj[:, 1] == 0)
    & (np.abs(ref_coords_adj[:, 2] - ref_coords_adj[:, 2].max()) < 0.001)
)[0]
edge3 = np.where(
    (np.abs(ref_coords_adj[:, 1] - ref_coords_adj[:, 1].max()) < 0.001)
    & (np.abs(ref_coords_adj[:, 2] - ref_coords_adj[:, 2].max()) < 0.001)
)[0]
edge4 = np.where(
    (np.abs(ref_coords_adj[:, 1] - ref_coords_adj[:, 1].max()) < 0.001)
    & (ref_coords_adj[:, 2] == 0)
)[0]

elements1 = np.array([edge1[:-1], edge1[1::]])
elements2 = np.array([edge2[:-1], edge2[1::]])
elements3 = np.array([edge3[:-1], edge3[1::]])
elements4 = np.array([edge4[:-1], edge4[1::]])

elements = np.concatenate([elements1, elements2], axis=0)[:, :86]
elements = np.concatenate(
    [elements, np.concatenate([elements2, elements3], axis=0)[:, :86]], axis=1
)
elements = np.concatenate(
    [elements, np.concatenate([elements3, elements4], axis=0)[:, :86]], axis=1
)
elements = np.concatenate(
    [elements, np.concatenate([elements4, elements1], axis=0)[:, :86]], axis=1
)

faces = []
for element in elements.T:
    # faces.append(element[[2, 0, 1]])
    # faces.append(element[[3, 2, 1]])
    faces.append(element[[1, 0, 2]])
    faces.append(element[[1, 2, 3]])
faces = np.unique(np.array(faces), axis=0)

# 3d plot of the beam
visualizer = Visualizer()
coords = (
    20
    * (x[:1000] @ V[:, :pca_order].T).reshape(
        (-1, ref_coords.shape[0], ref_coords.shape[1])
    )
    + ref_coords
)
# color = np.zeros([coords.shape[0], coords.shape[1]])
# color[:, 263] = 1
visualizer.animate(
    [coords],
    range(1000),
    color=[[0.8, 0.8, 0.8]],
    # color=color,
    point_size=8,
    colormap="viridis",
    color_scale_limits=[0, 1],
    # shift=False,
    faces=[faces],
    # rotate
)


# %% load trainhist
trainhist = np.load(
    os.path.join(outdir, f"trainhist_{test}.npy"), allow_pickle=True
).item()

plt.plot(np.array(trainhist["coeffs_mean"]).squeeze())
plt.plot(np.array(trainhist["coeffs_scale"]).squeeze())

plt.figure()
plt.semilogy(trainhist["loss"])
plt.semilogy(trainhist["rec"])
plt.semilogy(trainhist["dz"])
plt.semilogy(trainhist["dx"])
plt.semilogy(trainhist["kl"])
plt.semilogy(trainhist["kl_sindy"])
plt.legend(["loss", "rec", "dz", "dx", "kl_loss", "kl_sindy"])

plt.semilogy(trainhist["val_loss"])

mean_over_epochs = np.array(trainhist["coeffs_mean"]).squeeze()
scale_over_epochs = np.array(trainhist["coeffs_scale"]).squeeze()
# create gif showing how the coefficient distributions evolve over time
for i, (mean_, scale_) in enumerate(zip(mean_over_epochs, scale_over_epochs)):
    if i % 10 == 0:
        x_range = 2 - (1.5 * i / len(mean_over_epochs))
        # dont show figure
        # plt.ioff()
        fig = sindy_layer._visualize_coefficients(
            mean_, scale_, x_range=[-x_range, x_range]
        )
        # fig title
        fig.suptitle(f"Epoch {i}")
        # save fig as frame for gif
        fig.savefig(
            os.path.join(config.outdir, f"{model_name}/coefficients/coeffs_{i}.png")
        )
        plt.close(fig)
# make gif from frames
import imageio

images = []
for i in range(0, len(mean_over_epochs), 10):
    images.append(
        imageio.imread(
            os.path.join(config.outdir, f"{model_name}/coefficients/coeffs_{i}.png")
        )
    )
imageio.mimsave(
    os.path.join(config.outdir, f"{model_name}/coefficients/coeffs.gif"),
    images,
    duration=100,
)

aesindy.print(z=["z", "dz"], precision=4)
coeffs = aesindy.sindy_coeffs()

# sindy_coeffs, sindy_mean, sindy_log_var = veni.sindy_layer._coeffs
# sindy_coeffs = sindy_coeffs.numpy()
# sindy_mean = sindy_mean.numpy()
# sindy_log_var = sindy_log_var.numpy()

# save model
if not os.path.isdir(os.path.join(config.outdir, f"{model_name}")):
    os.mkdir(os.path.join(config.outdir, f"{model_name}"))
# veni.save(weights_dir)


# %% PUBLICATION

X_test = x_test_scaled.numpy().reshape(-1, n_timesteps_test, pca_order)
DXDT_test = dxdt_test_scaled.numpy().reshape(-1, n_timesteps_test, pca_order)
DXDDT_test = dxddt_test_scaled.numpy().reshape(-1, n_timesteps_test, pca_order)
PARAMS_test = params_test.reshape(-1, n_timesteps_test, params.shape[1])
t_test = t_test.reshape(-1, n_timesteps_test)
i_test = 0
n_traj = 10
n_test = 2  # X_test.shape[0]


"""
Forward UQ:
- We sample SINDy coefficients from the predicted posterior distribution
- We integrate the ODE with the sampled coefficients and collect the trajectories
"""

z_test, dzdt_test, dzddt_test = aesindy.calc_latent_time_derivatives(
    x_test_scaled, dxdt_test_scaled, dxddt_test_scaled
)
# reshape to (n_test, n_times, r)
z_test = z_test.reshape(-1, n_timesteps_test, reduced_order)
dzdt_test = dzdt_test.reshape(-1, n_timesteps_test, reduced_order)
dzddt_test = dzddt_test.reshape(-1, n_timesteps_test, reduced_order)


# Store the original coefficients
kernel_orig, kernel_scale_orig = sindy_layer.kernel, sindy_layer.kernel_scale

uq_ts = []
uq_ys = []
for i_test in range(n_test):
    logging.info(f"test trajectory {i_test+1}/{n_test}")
    # List to store the solution trajectories in latent space
    sol_list = []
    sol_list_t = []
    for traj in range(n_traj):
        # print(f"Trajectory {traj}")
        logging.info(f"\tsample {traj+1}/{n_traj}")
        # Sample from the posterior distribution of the coefficients
        sampled_coeff, _, _ = sindy_layer._coeffs
        # for the beam we only learn the second order term
        sampled_coeff = sampled_coeff[1:]
        # Assign the sampled coefficients to the SINDy layer
        sindy_layer.kernel = tf.reshape(sampled_coeff, (-1, 1))

        # sample latent initial conditions
        z0, dzdt0 = aesindy.calc_latent_time_derivatives(
            X_test[i_test][0:1], DXDT_test[i_test][0:1]
        )
        sol = aesindy.integrate(
            np.concatenate([z0, dzdt0]).squeeze(),
            t_test[i_test].squeeze(),
            mu=PARAMS_test[i_test],
        )
        sol_list.append(sol.y)
        sol_list_t.append(sol.t)
    uq_ts.append(sol_list_t)
    uq_ys.append(sol_list)
sindy_layer.kernel, sindy_layer.kernel_scale = kernel_orig, kernel_scale_orig
# calculate mean and variance of the trajectories
uq_ys = np.array(uq_ys)
uq_ys_mean = np.mean(uq_ys, axis=1)
uq_ys_std = np.std(uq_ys, axis=1)

# plot the mean and 3*std of the trajectories
fig, axs = plt.subplots(2, n_test, figsize=(12, 4), sharex=True)
axs = np.atleast_2d(axs)
fig.suptitle(f"Integrated Test Trajectories")
for i_test in range(n_test):
    axs[0, i_test].set_title(f"Test Trajectory {i_test+1}")
    # for i in range(2):
    axs[0, i_test].plot(t_test[i_test], z_test[i_test][:, 0], color="blue")
    axs[0, i_test].plot(
        uq_ts[i_test][0], uq_ys_mean[i_test][0], color="red", linestyle="--"
    )
    axs[0, i_test].fill_between(
        uq_ts[i_test][0],
        uq_ys_mean[i_test][0] - 3 * uq_ys_std[i_test][0],
        uq_ys_mean[i_test][0] + 3 * uq_ys_std[i_test][0],
        color="red",
        alpha=0.3,
    )
    axs[0, i_test].set_xlabel("$t$")
    axs[0, i_test].set_ylabel("$z$")

    axs[1, i_test].plot(t_test[i_test], dzdt_test[i_test][:, 0], color="blue")
    axs[1, i_test].plot(
        uq_ts[i_test][1], uq_ys_mean[i_test][1], color="red", linestyle="--"
    )
    axs[1, i_test].fill_between(
        uq_ts[i_test][1],
        uq_ys_mean[i_test][1] - 3 * uq_ys_std[i_test][1],
        uq_ys_mean[i_test][1] + 3 * uq_ys_std[i_test][1],
        color="red",
        alpha=0.3,
    )
    axs[1, i_test].set_xlabel("$t$")
    axs[1, i_test].set_ylabel("$\dot{z}$")

# %% Decode the latent uncertainties
x_vindy = np.array([aesindy.decode(uq_ys_mean_[0]) for uq_ys_mean_ in uq_ys_mean])
x_vindy_ub = np.array(
    [
        aesindy.decode(uq_ys_mean_[0] + 3 * std_[0])
        for uq_ys_mean_, std_ in zip(uq_ys_mean, uq_ys_std)
    ]
)
x_vindy_lb = np.array(
    [
        aesindy.decode(uq_ys_mean_[0] - 3 * std_[0])
        for uq_ys_mean_, std_ in zip(uq_ys_mean, uq_ys_std)
    ]
)

# plot the mean and 3*std of the trajectories
fig, axs = plt.subplots(pca_order, n_test, figsize=(12, 4), sharex=True)
axs = np.atleast_2d(axs)
fig.suptitle(f"Integrated Test Trajectories")
for i_test in range(n_test):
    axs[0, i_test].set_title(f"Test Trajectory {i_test+1}")
    for i in range(pca_order):
        axs[i, i_test].plot(t_test[i_test], X_test[i_test][:, i], color="blue")
        axs[i, i_test].plot(
            uq_ts[i_test][0], x_vindy[i_test][:, i], color="red", linestyle="--"
        )
        axs[i, i_test].fill_between(
            uq_ts[i_test][0],
            x_vindy_ub[i_test][:, i],
            x_vindy_lb[i_test][:, i],
            color="red",
            alpha=0.3,
        )
        axs[i, i_test].set_xlabel("$t$")


# %% decode to physical space
x_phys = x_vindy @ V[:, :pca_order].T
x_phys_ub = x_vindy_ub @ V[:, :pca_order].T
x_phys_lb = x_vindy_lb @ V[:, :pca_order].T
X_ref = X_test @ V[:, :pca_order].T

dofs = [100, 200, 300]
n_dofs = len(dofs)

# plot the mean and 3*std of the trajectories
fig, axs = plt.subplots(n_dofs, n_test, figsize=(12, 4), sharex=True)
axs = np.atleast_2d(axs)
fig.suptitle(f"Integrated Test Trajectories")
for i_test in range(n_test):
    axs[0, i_test].set_title(f"Test Trajectory {i_test+1}")
    for i, dof in enumerate(dofs):
        axs[i, i_test].plot(t_test[i_test], X_ref[i_test][:, dof], color="blue")
        axs[i, i_test].plot(
            uq_ts[i_test][0], x_phys[i_test][:, dof], color="red", linestyle="--"
        )
        axs[i, i_test].fill_between(
            uq_ts[i_test][0],
            x_phys_ub[i_test][:, dof],
            x_phys_lb[i_test][:, dof],
            color="red",
            alpha=0.3,
        )
        axs[i, i_test].set_xlabel("$t$")


# %% compare time derivatives
z = aesindy.encode(x_test_scaled).numpy()
# calc time derivatives
dzdt = np.gradient(z, t_test.squeeze(), axis=0)
dzddt = np.gradient(dzdt, t_test.squeeze(), axis=0)
z, dzdt2, dzddt2 = aesindy.calc_latent_time_derivatives(
    x_test_scaled, dxdt_test_scaled, dxddt_test_scaled
)

plt.figure()
plt.plot(dzdt[:500])
plt.plot(dzdt2[:500])
plt.xlabel("time")
plt.ylabel("$\dot{z}$")
plt.legend(["dzdt numerically", "dzdt chain rule"])
plt.show()

plt.figure()
plt.plot(dzddt[:500])
plt.plot(dzddt2[:500])
plt.xlabel("time")
plt.ylabel("$\ddot{z}$")
plt.legend(["dzdt numerically", "dzdt chain rule"])
plt.show()


# %% compare time derivatives with sindy model
if second_order:
    z, dzdt2, dzddt2 = aesindy.calc_latent_time_derivatives(
        x_test_scaled, dxdt_test_scaled, dxddt_test_scaled
    )
    s_dot_sindy = aesindy.sindy(tf.concat([z, dzdt2, params_test], axis=1)).numpy()
    dzdt_sindy, dzddt_sindy = (
        s_dot_sindy[:, :reduced_order],
        s_dot_sindy[:, reduced_order:],
    )

    plt.figure()
    plt.plot(
        dzddt2[: 3 * n_timesteps],
        label="$\ddot{z}$ chain rule",
        linestyle="-",
        color="C0",
    )
    # plt.plot(dzddt[:2*n_timesteps], label='$\ddot{z}$ numeric', linestyle='--', color='blue')
    plt.plot(
        dzddt_sindy[: 3 * n_timesteps],
        label="$\ddot{z}_s$ SINDy",
        linestyle="--",
        color="C1",
    )
    plt.xlabel("time")
    plt.ylabel("$\ddot{z}$")
    plt.legend()
    plt.show()

else:
    z, dzdt2 = aesindy.calc_latent_time_derivatives(x_train_scaled, dxdt_train_scaled)
    s_dot_sindy = aesindy.sindy(tf.concat([z, params], axis=1)).numpy()
    dzdt_sindy = s_dot_sindy[:, :reduced_order]

    plt.figure()
    plt.plot(
        dzdt2[: 2 * n_timesteps, 1],
        label="$\dot{z}$ chain rule",
        linestyle="-",
        color="C0",
    )
    # plt.plot(dzddt[:2*n_timesteps], label='$\dot{z}$ numeric', linestyle='--', color='blue')
    plt.plot(
        dzdt_sindy[: 2 * n_timesteps, 1],
        label="$\dot{z}_s$ SINDy",
        linestyle="--",
        color="C1",
    )
    plt.xlabel("time")
    plt.ylabel("$\ddot{z}$")
    plt.legend()
    plt.show()

# %%
for i_test in range(n_sims):
    i_test = 10
    t_0 = i_test * int(n_timesteps_test)
    sol = aesindy.integrate(
        np.concatenate([z[t_0 : t_0 + 1], dzdt2[t_0 : t_0 + 1]], axis=1).squeeze(),
        t_test[:n_timesteps_test].squeeze(),
        mu=params_test[t_0 : t_0 + n_timesteps_test],
    )
    plt.figure(figsize=[10, 6])
    # subplot 1 for z
    plt.subplot(2, 1, 1)
    plt.plot(
        sol.t, z[t_0 : t_0 + n_timesteps_test - 1, 0], label="$z$ real", color="blue"
    )
    plt.plot(sol.t, sol.y[0], label="$z_s$ pred", linestyle="--", color="red")
    plt.xlabel("time")
    plt.ylabel("$z$")
    plt.legend()
    # subplot 2 for z_dot
    plt.subplot(2, 1, 2)
    plt.plot(
        sol.t,
        dzdt2[t_0 : t_0 + n_timesteps_test - 1, 0],
        label="$\dot{z}$",
        color="blue",
    )
    plt.plot(sol.t, sol.y[1], label="$\dot{z}_s$", linestyle="--", color="red")
    plt.xlabel("time")
    plt.ylabel("$\dot{z}$")
    plt.legend()
    plt.show()
