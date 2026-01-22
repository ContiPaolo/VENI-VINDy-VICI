#%%
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
from utils import load_reactiondiffusion_data, plot_train_history, plot_coefficients_train_history

# Add the examples folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import config
import config

# tf.config.run_functions_eagerly(True)  # uncomment this line for debugging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logging.info(
    "################################   1. Loading    ################################"
)

# %% script parameters
# Load config
model_name = "reactiondiffusion"
identification_layer = "vindy"  # 'vindy' or 'sindy'
# Script parameter
reduced_order = 2
pca_order = 32
noise = True
nth_time_step = 3
second_order = False

beta_vindy = 2e-5#1e-8  # 5e-9
beta_vae = 1e-4#1e-8  # 1e-8
l_rec = 1e-2#1e-3  # 1e-3
l_dz = 4e0 #1e0  # 1e0
l_dx = 1e-2 #1e-5  # 1e-5

# this scripts path + results
result_dir = os.path.join(os.path.dirname(__file__), "results")

# %% Load Data
import mat73
print("Loading data from ", config.reaction_diffusion["processed_data"])
data = mat73.loadmat(config.reaction_diffusion["processed_data"])
print("Data loaded.")
#data = np.load(data_paths, allow_pickle=True)

times = data["t"]
dt = times[1] - times[0]
x_full = data["U"]

mu = 0
sigma = 0.2 
scale_noise = np.exp(mu)
x_full_noise = x_full * (
    np.random.lognormal(mean=0, sigma=sigma, size=x_full.shape)
    * scale_noise
)
n_sims = x_full.shape[-1]
dxdt_full = np.array([np.gradient(x_full[:, :, :, i], dt, axis=2, edge_order=2) for i in range(n_sims)]
)

dxdt_full_noise = np.array([np.gradient(x_full_noise[:, :, :, i], dt, axis=2, edge_order=2) for i in range(n_sims)]
)

plt.figure()
plt.plot(x_full[25,25,:,0], label='state noise-free')
plt.plot(x_full_noise[25,25,:,0], label='state noisy')
plt.legend()
plt.show()

plt.figure()
plt.plot(dxdt_full[0,25,25,:], label='state derivative noise-free')
plt.plot(dxdt_full_noise[0,25,25,:], label='state derivative noisy')
plt.legend()
plt.show()
# %% signal noise ratio
rms = lambda x: np.sqrt(np.mean(x**2))

SNR_state = rms(x_full_noise) / rms(x_full - x_full_noise)
SNR_state_derivative = rms(dxdt_full_noise) / rms(dxdt_full - dxdt_full_noise)

SNR_db_state = 20 * np.log10(SNR_state)
SNR_db_state_derivative = 20 * np.log10(SNR_state_derivative)

print(f"SNR state: {SNR_db_state:.2f} dB")
print(f"SNR state derivative: {SNR_db_state_derivative:.2f} dB")

#%%
if noise:
    # for storage reasons we just load the pca components and not the full noisy data
    (
        t,
        x,
        dxdt,
        t_test,
        x_test,
        dxdt_test,
        V,
        n_sims,
        n_timesteps,
    ) = load_reactiondiffusion_data(
        config.reaction_diffusion["processed_data"],
        pca_order=pca_order,
    )

#noise_free
(t_,
 x_noise_free,
 dxdt_noise_free,
 t_test_,
 x_test_noise_free,
 dxdt_test_noise_free,
 V_,
 n_sims_,
 n_timesteps_,
 ) = load_reactiondiffusion_data(
    config.reaction_diffusion["processed_data"],
    pca_order=pca_order,
    noise=False,

)
n_timesteps_test = x_test.shape[0] // n_sims
n_dof = x.shape[1]
dt = t[1] - t[0]


 # %% Create Model
logging.info(
    "################################   2. Training    ################################"
)
logging.info("Creating model")

libraries = [PolynomialLibrary(3)]
param_libraries = []


# create sindy layer
layer_params = dict(
    state_dim=reduced_order,
    param_dim=0,
    feature_libraries=libraries,
    second_order=second_order,
    param_feature_libraries=param_libraries,
    x_mu_interaction=False,
    kernel_regularizer= tf.keras.regularizers.L1L2(l1=0, l2=0),#tf.keras.regularizers.L1L2(l1=1e-8, l2=0),
    mask=None,
    fixed_coeffs=None,
)
if identification_layer == "vindy":
    sindy_layer = VindyLayer(
        beta=beta_vindy,
        priors=Laplace(0.0, 1.0),
        **layer_params,
    )
elif identification_layer == "sindy":
    sindy_layer = SindyLayer(**layer_params)
else:
    raise ValueError('identification_layer must be either "vindy" or "sindy"')

veni = VENI(
    sindy_layer=sindy_layer,
    beta=beta_vae * reduced_order / n_dof,
    reduced_order=reduced_order,  # 1e-3
    x=x,
    mu=None,
    scaling="individual_sqrt",
    second_order=second_order,
    layer_sizes=[32, 16, 8],
    activation="elu",
    l_rec=l_rec,
    l_dz=l_dz,
    l_dx=l_dx,
    dt=dt,
)


# %% Scale data
veni.define_scaling(x)
x_train_scaled, dxdt_train_scaled = (
    veni.scale(x),
    veni.scale(dxdt),
)
x_test_scaled, dxdt_test_scaled = (
    veni.scale(x_test),
    veni.scale(dxdt_test),
)

x_input = [
    x_train_scaled[: 14 * n_timesteps],
    dxdt_train_scaled[: 14 * n_timesteps],
]
x_input_val = [
    x_train_scaled[14 * n_timesteps :],
    dxdt_train_scaled[14 * n_timesteps :],
]

# compile and build
veni.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    sindy_optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
    loss="mse",
)
veni.build(input_shape=([input.shape for input in x_input], None))


# %% Train model
logging.info("Training model")
load_model = False

log_dir = os.path.join(
    result_dir,
    f'{model_name}/log/{model_name}_{reduced_order}_{veni.__class__.__name__}_{datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")}',
)
weights_path = os.path.join(
    result_dir,
    f"{model_name}/{model_name}_{reduced_order}_{veni.__class__.__name__}_{identification_layer}/",#.weights.h5",
)
outdir = os.path.join(result_dir, f"{model_name}")
if not outdir:
    os.mkdir(outdir)

if load_model:
    veni.load_weights(
        os.path.join(weights_path)
    )  # kwargs_overwrite=dict(l_rec=1e0, l_dz=2e3, l_dx=1e-2, l1=0, l2=0, l_int=0))
    #
    # Coefficient Posterior Distributions
    # coefficient_distributions_to_csv(
    #     sindy_layer, outdir, var_names=["z", r"\stackrel{.}{z}"]
    # )
else:
    # save model config to file
    # veni.save(weights_path)
    callbacks = []
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
        )
    )
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    callbacks.append(SaveCoefficientsCallback())

    # veni.load_weights(os.path.join(weights_path))
    trainhist = veni.fit(
        x=x_input,
        validation_data=(x_input_val, None),
        callbacks=callbacks,
        y=None,
        epochs=2500,
        batch_size=int(n_timesteps / nth_time_step),
        verbose=2,
    )

    # save trainhist
    np.save(
        os.path.join(result_dir, f"{model_name}/trainhist_{identification_layer}.npy"),
        trainhist.history,
    )
    # load trainhist
    trainhist = np.load(
        os.path.join(result_dir, f"{model_name}/trainhist_{identification_layer}.npy"),
        allow_pickle=True,
    ).item()

    mean_over_epochs = np.array(trainhist["coeffs_mean"]).squeeze()
    scale_over_epochs = np.array(trainhist["coeffs_scale"]).squeeze()

    plot_coefficients_train_history(trainhist, outdir)

    plot_train_history(trainhist, outdir, validation=True)
    plot_train_history(trainhist, outdir, validation=False)

    veni.print(z=["z", "dz"], precision=4)

    # save model
    os.makedirs(os.path.join(outdir, f"{model_name}"), exist_ok=True)
    veni.save(weights_path)

# %% Plot training history

# reconstruction of PCA trajectories
veni.vis_modes(x_test_scaled, 4)
veni.vis_modes(x_train_scaled, 4)

sindy_layer.visualize_coefficients(x_range=[-0.5, 0.5])
plt.show()

# %% Sparsification of the identified model
sindy_layer.pdf_thresholding(threshold=.001)

# %% PUBLICATION

# reshape data for further processing
X_test = x_test_scaled.numpy().reshape(-1, n_timesteps_test, pca_order)
DXDT_test = dxdt_test_scaled.numpy().reshape(-1, n_timesteps_test, pca_order)
#DXDDT_test = dxddt_test_scaled.numpy().reshape(-1, n_timesteps_test, pca_order)
#PARAMS_test = params_test.reshape(-1, n_timesteps_test, params.shape[1])
t_test = t_test.reshape(-1, n_timesteps_test)
i_test = 0
n_traj = 10
test_ids = [1, 10]
n_test = len(test_ids)

# error in physical space
# %%
z, dzdt = veni.calc_latent_time_derivatives(
    x_test_scaled, dxdt_test_scaled
)
n_nodes = 1988
e_disps = []
e_disps_rel = []
for i_test in test_ids:
    # i_test = 10
    t_0 = i_test * int(n_timesteps_test)
    sol = veni.integrate(
        np.concatenate([z[t_0 : t_0 + 1]], axis=1).squeeze(),
        t_test[i_test].squeeze(),
    )
    z_pred = sol.y[0]
    dzdt_pred = sol.y[1]
    z_pca_pred = veni.decode(z_pred)
    X_pred = (
        (z_pca_pred @ V[:, :pca_order].T)
        .numpy()
        .reshape(n_timesteps_test - 1, n_nodes, n_dof)
    )
    X_ref = (X_test[i_test, : n_timesteps_test - 1] @ V[:, :pca_order].T).reshape(
        n_timesteps_test - 1, n_nodes, n_dof
    )
    e_disp = np.linalg.norm(X_pred - X_ref, axis=2)
    e_rel = np.linalg.norm(X_pred - X_ref, axis=2) / np.linalg.norm(X_ref, axis=2).max()

    e_disps.append(e_disp)
    e_disps_rel.append(e_rel)
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
        dzdt[t_0 : t_0 + n_timesteps_test - 1, 0],
        label=r"$\dot{z}$",
        color="blue",
    )
    plt.plot(sol.t, sol.y[1], label=r"$\dot{z}_s$", linestyle="--", color="red")
    plt.xlabel("time")
    plt.ylabel(r"$\dot{z}$")
    plt.legend()
    plt.show()

e_disps = np.array(e_disps)
e_disps_rel = np.array(e_disps_rel)
e_disp_mean = np.mean(e_disps, axis=2).mean(axis=1)
e_rel_mean = np.mean(e_disps_rel, axis=2).mean(axis=1)

for e_ in e_rel_mean:
    print(r"&$\numprint{" + str(e_) + r"}$")

# %% UQ
"""
Forward UQ:
- We sample SINDy coefficients from the predicted posterior distribution
- We integrate the ODE with the sampled coefficients and collect the trajectories
"""

z_test, dzdt_test, dzddt_test = veni.calc_latent_time_derivatives(
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
uq_means = []
for i_test in test_ids:
    logging.info(f"identification_layer trajectory {i_test+1}/{n_test}")
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
        z0, dzdt0 = veni.calc_latent_time_derivatives(
            X_test[i_test][0:1], DXDT_test[i_test][0:1]
        )
        sol = veni.integrate(
            np.concatenate([z0, dzdt0]).squeeze(),
            t_test[i_test].squeeze(),
            mu=PARAMS_test[i_test],
        )
        sol_list.append(sol.y)
        sol_list_t.append(sol.t)
    uq_ts.append(sol_list_t)
    uq_ys.append(sol_list)
    # mean simulation
    sindy_layer.kernel, sindy_layer.kernel_scale = kernel_orig, kernel_scale_orig
    z0, dzdt0 = veni.calc_latent_time_derivatives(
        X_test[i_test][0:1], DXDT_test[i_test][0:1]
    )
    sol = veni.integrate(
        np.concatenate([z0, dzdt0]).squeeze(),
        t_test[i_test].squeeze(),
        mu=PARAMS_test[i_test],
    )
    uq_means.append(sol.y)

# calculate mean and variance of the trajectories
uq_ys = np.array(uq_ys)
uq_ys_mean_sampled = np.mean(
    uq_ys, axis=1
)  # based on mean of the simulations using sampled coefficients
uq_ys_std = np.std(uq_ys, axis=1)
uq_ys_mean = np.array(uq_means)  # based on the simulation using the mean coefficients

# plot the mean and 3*std of the trajectories
fig, axs = plt.subplots(2, n_test, figsize=(12, 4), sharex=True)
axs = np.atleast_2d(axs)
fig.suptitle(f"Integrated Test Trajectories")
for i, i_test in enumerate(test_ids):
    axs[0, i].set_title(f"Test Trajectory {i_test+1}")
    # for i in range(2):
    axs[0, i].plot(t_test[i_test], z_test[i_test][:, 0], color="blue")
    axs[0, i].plot(uq_ts[i][0], uq_ys_mean[i][0], color="red", linestyle="--")
    axs[0, i].fill_between(
        uq_ts[i][0],
        uq_ys_mean_sampled[i][0] - 3 * uq_ys_std[i][0],
        uq_ys_mean_sampled[i][0] + 3 * uq_ys_std[i][0],
        color="red",
        alpha=0.3,
    )
    axs[0, i].set_xlabel("$t$")
    axs[0, i].set_ylabel("$z$")

    axs[1, i].plot(
        t_test[i_test],
        dzdt_test[i_test][:, 0],
        color="blue",
    )
    axs[1, i].plot(uq_ts[i][1], uq_ys_mean[i][1], color="red", linestyle="--")
    axs[1, i].fill_between(
        uq_ts[i][1],
        uq_ys_mean_sampled[i][1] - 3 * uq_ys_std[i][1],
        uq_ys_mean_sampled[i][1] + 3 * uq_ys_std[i][1],
        color="red",
        alpha=0.3,
    )
    axs[1, i].set_xlabel("$t$")
    axs[1, i].set_ylabel(r"$\dot{z}$")

# %% Decode the latent uncertainties
x_vindy = np.array([veni.decode(uq_ys_mean_[0]) for uq_ys_mean_ in uq_ys_mean])
x_vindy_ub = np.array(
    [
        veni.decode(uq_ys_mean_[0] + 3 * std_[0])
        for uq_ys_mean_, std_ in zip(uq_ys_mean_sampled, uq_ys_std)
    ]
)
x_vindy_lb = np.array(
    [
        veni.decode(uq_ys_mean_[0] - 3 * std_[0])
        for uq_ys_mean_, std_ in zip(uq_ys_mean_sampled, uq_ys_std)
    ]
)

# plot the mean and 3*std of the trajectories
fig, axs = plt.subplots(pca_order, n_test, figsize=(12, 4), sharex=True)
axs = np.atleast_2d(axs)
fig.suptitle(f"Integrated Test Trajectories")
for i, i_test in enumerate(test_ids):
    axs[0, i].set_title(f"Test Trajectory {i_test+1}")
    for j in range(pca_order):
        axs[j, i].plot(t_test[i_test], X_test[i_test][:, j], color="blue")
        axs[j, i].plot(uq_ts[i][0], x_vindy[i][:, j], color="red", linestyle="--")
        axs[j, i].fill_between(
            uq_ts[i][0],
            x_vindy_ub[i][:, j],
            x_vindy_lb[i][:, j],
            color="red",
            alpha=0.3,
        )
        axs[j, i].set_xlabel("$t$")
plt.show()

# %% decode to physical space
x_phys = x_vindy @ V[:, :pca_order].T
x_phys_ub = x_vindy_ub @ V[:, :pca_order].T
x_phys_lb = x_vindy_lb @ V[:, :pca_order].T
X_ref = X_test @ V[:, :pca_order].T

dofs = [789, 790, 791]  # node in the middle of the beam (263), y-dof = 263*3+1 = 790
n_dofs = len(dofs)

# plot the mean and 3*std of the trajectories
fig, axs = plt.subplots(n_dofs, n_test, figsize=(12, 4), sharex=True)
axs = np.atleast_2d(axs)
fig.suptitle(f"Integrated Test Trajectories")
for i, i_test in enumerate(test_ids):
    axs[0, i].set_title(f"Test Trajectory {i_test+1}")
    for j, dof in enumerate(dofs):
        axs[j, i].plot(t_test[i_test], X_ref[i_test][:, dof], color="blue")
        axs[j, i].plot(uq_ts[i][0], x_phys[i][:, dof], color="red", linestyle="--")
        axs[j, i].fill_between(
            uq_ts[i][0],
            x_phys_ub[i][:, dof],
            x_phys_lb[i][:, dof],
            color="red",
            alpha=0.3,
        )
        axs[j, i].set_xlabel("$t$")
plt.show()

save_data = dict()
# for i, i_test in enumerate(test_ids):
# scenario_info = f"test_{i_test}_omega_{PARAMS_test[i_test, 0][1]:.3f}_F_{PARAMS_test[i_test, 0][2]:.2f}"
save_data = dict()
# print(scenario_info)
save_data["uq_ts"] = uq_ts
save_data["z"] = np.concatenate(
    [z_test[test_ids], dzdt_test[test_ids]], axis=2
).transpose((0, 2, 1))
save_data["z_pred_mean"] = uq_ys_mean
save_data["z_pred_ub"] = uq_ys_mean_sampled + 3 * uq_ys_std
save_data["z_pred_lb"] = uq_ys_mean_sampled - 3 * uq_ys_std

save_data["x"] = X_test[test_ids]
save_data["x_pred_mean"] = x_vindy
save_data["x_pred_ub"] = x_vindy_ub
save_data["x_pred_lb"] = x_vindy_lb

# save_data["X"] = X_ref[test_ids, :][:, :, dofs]
# save_data["X_pred_mean"] = x_phys[:, :, dofs]
# save_data["X_pred_ub"] = x_phys_ub[:, :, dofs]
# save_data["X_pred_lb"] = x_phys_lb[:, :, dofs]
save_data["X"] = X_ref[test_ids]
save_data["X_pred_mean"] = x_phys
save_data["X_pred_ub"] = x_phys_ub
save_data["X_pred_lb"] = x_phys_lb

# save the data to a file
with open(os.path.join(outdir, f"save_data_{model_name}.pkl"), "wb") as f:
    pickle.dump(save_data, f)


# %% compare time derivatives
z = veni.encode(x_test_scaled).numpy()
# calc time derivatives
dzdt = np.gradient(z, t_test.squeeze(), axis=0)
dzddt = np.gradient(dzdt, t_test.squeeze(), axis=0)
z, dzdt2, dzddt2 = veni.calc_latent_time_derivatives(
    x_test_scaled, dxdt_test_scaled, dxddt_test_scaled
)

plt.figure()
plt.plot(dzdt[:500])
plt.plot(dzdt2[:500])
plt.xlabel("time")
plt.ylabel(r"$\dot{z}$")
plt.legend(["dzdt numerically", "dzdt chain rule"])
plt.show()

plt.figure()
plt.plot(dzddt[:500])
plt.plot(dzddt2[:500])
plt.xlabel("time")
plt.ylabel(r"$\ddot{z}$")
plt.legend(["dzdt numerically", "dzdt chain rule"])
plt.show()


# %% compare time derivatives with sindy model
if second_order:
    z, dzdt2, dzddt2 = veni.calc_latent_time_derivatives(
        x_test_scaled, dxdt_test_scaled, dxddt_test_scaled
    )
    s_dot_sindy = veni.sindy(tf.concat([z, dzdt2, params_test], axis=1)).numpy()
    dzdt_sindy, dzddt_sindy = (
        s_dot_sindy[:, :reduced_order],
        s_dot_sindy[:, reduced_order:],
    )

    plt.figure()
    plt.plot(
        dzddt2[: 3 * n_timesteps],
        label=r"$\ddot{z}$ chain rule",
        linestyle="-",
        color="C0",
    )
    # plt.plot(dzddt[:2*n_timesteps], label='$\ddot{z}$ numeric', linestyle='--', color='blue')
    plt.plot(
        dzddt_sindy[: 3 * n_timesteps],
        label=r"$\ddot{z}_s$ SINDy",
        linestyle="--",
        color="C1",
    )
    plt.xlabel("time")
    plt.ylabel(r"$\ddot{z}$")
    plt.legend()
    plt.show()

else:
    z, dzdt2 = veni.calc_latent_time_derivatives(x_train_scaled, dxdt_train_scaled)
    s_dot_sindy = veni.sindy(tf.concat([z, params], axis=1)).numpy()
    dzdt_sindy = s_dot_sindy[:, :reduced_order]

    plt.figure()
    plt.plot(
        dzdt2[: 2 * n_timesteps, 1],
        label=r"$\dot{z}$ chain rule",
        linestyle="-",
        color="C0",
    )
    # plt.plot(dzddt[:2*n_timesteps], label='$\dot{z}$ numeric', linestyle='--', color='blue')
    plt.plot(
        dzdt_sindy[: 2 * n_timesteps, 1],
        label=r"$\dot{z}_s$ SINDy",
        linestyle="--",
        color="C1",
    )
    plt.xlabel("time")
    plt.ylabel(r"$\ddot{z}$")
    plt.legend()
    plt.show()

# %%
