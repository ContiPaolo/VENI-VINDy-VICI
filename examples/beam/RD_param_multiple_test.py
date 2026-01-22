# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:36:23 2023

@author: paolo
"""
# %%
import matplotlib.pyplot as plt
import tensorflow as tf
import pysindy as ps
import os
import datetime
import numpy as np

# local imports
import logging
import sys

# Add the examples folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import config
import config

from keras import backend as K
from vindy import VAESindy, AutoencoderSindy, PCAESindy, config
from aesindy.libraries import (
    PolynomialLibrary,
    ForceLibrary,
    FourierLibrary,
    ExponentialLibrary,
)
from aesindy.layers import SindyLayer, VariationalSindyLayer
from aesindy.layers.distributions import Gaussian, Laplace
from aesindy.callbacks import SindyCallback, SaveCoefficientsCallback
from aesindy.utils import *

import mat73
import scipy.io
from sklearn.utils import extmath

save = False
load_model = False

from sklearn.model_selection import train_test_split

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# %%
def compute_randomized_SVD(S, N_POD, N_h, n_channels, name="", verbose=False):
    if verbose:
        print("Computing randomized POD...")
    U = np.zeros((n_channels * N_h, N_POD))
    start_time = time.time()
    for i in range(n_channels):
        U[i * N_h : (i + 1) * N_h], Sigma, Vh = extmath.randomized_svd(
            S[i * N_h : (i + 1) * N_h, :],
            n_components=N_POD,
            transpose=False,
            flip_sign=False,
            random_state=123,
        )
        if verbose:
            print("Done... Took: {0} seconds".format(time.time() - start_time))

    if verbose:
        I = 1.0 - np.cumsum(np.square(Sigma)) / np.sum(np.square(Sigma))
        print(I[-1])

    if name:
        sio.savemat(name, {"V": U[:, :N_POD]})

    return U, Sigma, Vh


# %% Script parameters
logging.info(
    "################################   1. Data    ################################"
)
# Script parameter
reduced_order = 2
test_size = 2
seed = 29  # 7

# AE SINDy parameters
poly_order = 3
use_pca = False
pod = True
pca_order = 32
convolutional = False
numerical_derivatives = False  # !
second_order = False
# Model parameters
model_name = "RD_noparam"
fifth_order = False  # use fifth order terms in normal form
preprocess = False
reduce_training = False
short = True
noise = True

# Mask
fixed_coeffs = np.array([[0]] * reduced_order)
n_terms = np.sum([i for i in range(poly_order + 2)])
mask = np.zeros((reduced_order, n_terms))
fixed_coeffs = np.array([[0]] * reduced_order)

logging.info(
    f"\tModel: Reaction-diffusion\n"
    f"\t\tReduced order: {reduced_order}\n"
    f"\t\tTest size: {test_size}\n"
    f"\t\tNumerical derivatives: {numerical_derivatives}\n"
)

# %% load data

# data = mat73.loadmat(r"C:\Users\paolo\PhD\Jonas\AE2ndOrderSINDy\aesindy\resources\RD\noparam\RD_noparam_ICs.mat")
data = mat73.loadmat(
    r"/Users/pconti/Desktop/VENI-VINDy-VICI/data/RD_noparam_ICs_20_test.mat"
)
# data = mat73.loadmat(r"C:\Users\paolo\PhD\Jonas\AE2ndOrderSINDy\aesindy\resources\RD\noparam\RD_noparam_ICs_9_test.mat")

x_full = data["U"]  # [:,:,:,-2:]
times = data["t"]
dt = times[1] - times[0]
Nx_hf, Ny_hf, n_timesteps, n_sims = x_full.shape

N = Nx_hf * Ny_hf

n_timesteps_train = int(n_timesteps / 2)

# Reshape for POD
if pod:
    x_pod_u_train = np.reshape(
        data["U"][:, :, :n_timesteps_train, :], (N, n_timesteps_train * n_sims), "F"
    )
    x_pod_v_train = np.reshape(
        data["V"][:, :, :n_timesteps_train, :], (N, n_timesteps_train * n_sims), "F"
    )
    x_pod_train = np.concatenate((x_pod_u_train, x_pod_v_train))
    if noise:
        mu = 0
        sigma = 0.2  # 0.4 #0.3
        scale_noise = np.exp(mu)
        np.random.seed(seed)
        x_pod_train *= (
            np.random.lognormal(mean=0, sigma=sigma, size=x_pod_train.shape)
            * scale_noise
        )

    U, S, _ = compute_randomized_SVD(x_pod_train, pca_order, N * 2, 1)
    x_pod_u = np.reshape(data["U"], (N, n_timesteps * n_sims), "F")
    x_pod_v = np.reshape(data["V"], (N, n_timesteps * n_sims), "F")
    x_pod = np.concatenate((x_pod_u, x_pod_v))
    x = x_pod.T @ U
    x = np.reshape(x, (n_sims, n_timesteps, pca_order))
else:
    x = data["U"].transpose(3, 2, 0, 1).reshape(n_sims, n_timesteps, N)

# POD on HF training data


dxdt = np.array(
    [np.gradient(x[i, :, :], dt, axis=0, edge_order=2) for i in range(n_sims)]
)
dxddt = np.array(
    [np.gradient(dxdt[i, :, :], dt, axis=0, edge_order=2) for i in range(n_sims)]
)

if short:
    n_timesteps = 800
    n_timesteps_train = int(n_timesteps / 2)

delay = 0
x = x[:, delay:n_timesteps, :]
dxdt = dxdt[:, delay:n_timesteps, :]
dxddt = dxddt[:, delay:n_timesteps, :]
times = times[delay:n_timesteps]

n_timesteps -= delay
n_timesteps_train -= delay
indexes = np.arange(n_sims)

x_train, x_test, dxdt_train, dxdt_test, dxddt_train, dxddt_test, ind_train, ind_test = (
    train_test_split(x, dxdt, dxddt, indexes, test_size=0.2, random_state=seed)
)

# Reduce training data to the training window
x_train = x_train[:, :n_timesteps_train, :]
dxdt_train = dxdt_train[:, :n_timesteps_train, :]
dxddt_train = dxddt_train[:, :n_timesteps_train, :]


times_train = np.tile(times, x_train.shape[0]).reshape(-1, 1)
times_test = np.tile(times, x_test.shape[0]).reshape(-1, 1)


def reshape_ae(data, reduce=False, n_timesteps=n_timesteps):
    if reduce:
        data = data[:, :n_timesteps, :]
    return data.reshape(-1, data.shape[2])


data_ = [x_train, dxdt_train, dxddt_train]  # , x_int_train, dx_int_train]

for i in range(len(data_)):
    data_[i] = reshape_ae(data_[i], reduce=reduce_training, n_timesteps=n_timesteps)
x_train, dxdt_train, dxddt_train = data_

data_ = [x_test, dxdt_test, dxddt_test]  # , x_int_test, dx_int_test]

for i in range(len(data_)):
    data_[i] = reshape_ae(data_[i])

x_test, dxdt_test, dxddt_test = data_


# %% Random preprocess
if preprocess:
    np.random.seed(seed)
    Sigma = np.random.rand(pca_order, pca_order)
    U_sigma, S_sigma, V_sigma = compute_randomized_SVD(Sigma, pca_order, pca_order, 1)
    Sigma = U_sigma @ V_sigma
    Sigma = Sigma * np.sqrt(S)

    x_train = x_train @ Sigma
    dxdt_train = dxdt_train @ Sigma
    x_test = x_test @ Sigma
    dxdt_test = dxdt_test @ Sigma
    # preprocess = False


# %% Create Model
logging.info(
    "################################   2. Training    ################################"
)
logging.info("Creating model")

if second_order:
    x_dim = reduced_order * 2
else:
    x_dim = reduced_order * 1

libraries = [
    PolynomialLibrary(poly_order, x_dim=x_dim),
]
param_libraries = [
    # PolynomialLibrary(3),
    # FourierLibrary([1]),
    # ForceLibrary()
]
# dt = times_train[1] - times_train[0]
# dt = dt[0]

n_tests = 1
trainhists_ae = []
trainhists_vae = []

# %% AE SINDY
# for n_test in range(n_tests):
#     tf.random.set_seed(n_test)
#     kernel_regularizer = tf.keras.regularizers.L1L2(l1=1e-2, l2=0)
#     sindy_layer = \
#         SindyLayer(
#         # SindyLayer(
#             state_dim=reduced_order, param_dim=0,
#             kernel_regularizer=kernel_regularizer,
#             feature_libraries=libraries, second_order=second_order,
#             param_feature_libraries=param_libraries,
#             x_mu_interaction=False,
#             mask=None, fixed_coeffs=None, dtype=tf.float32
#         )

#     aesindy = AutoencoderSindy(
#         sindy_layer=sindy_layer,
#         #beta=1e-3 * reduced_order / n_dof,
#         reduced_order=reduced_order,
#         x=x_train,
#         mu=None,
#         scaling="individual",
#         layer_sizes=[32, 16, 8], activation='elu',
#         l_rec=1e0, l_dz=1e0, l_dx=1e-2, l1=0, l2=0, l_int=0, dt=dt,
#         second_order=second_order
#     )

#     # aesindy.print()
#     optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)  # 1e-3
#     sindy_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)  # 1e-1 #5e-3

#     aesindy.compile(optimizer=optimizer,
#                     sindy_optimizer=sindy_optimizer,
#                     loss='mse',
#                     # run_eagerly=True,
#                     )


#     print('TEST NUM: ', n_test)
#     # test_num = str(n_test) + '_AE_ICs_nopcascaling_L1_nocallback' #2_VAE_ICs_yespcascaling_L1
#     # test_gs = f'{model_name}/{model_name}_{test_num}'  # 'gs' + '_1' #'gs_' + str(num_traj) + '_1'  # 30000 4
#     # # gs_1
#     # test = test_gs
#     # load_model = False
#     # logging.info('Training model')
#     # log_dir = os.path.join(config.outdir,
#     #                        rf'{model_name}/log/{model_name}_{reduced_order}_{aesindy.__class__.__name__}_' + test_gs)
#     # weights_path = os.path.join(config.outdir,
#     #                             rf'{model_name}/{model_name}_{reduced_order}_{aesindy.__class__.__name__}_' + test_gs)
#     logging.info('Training model')
#     load_model = False
#     test = 'AE_0'  # Arch_GS_9_constrained' #Arch_GS_9_0_NF #Arch_GS_9_0 #Arch_GS_9_0_5order
#     #AE_0: preprocess true, pod true
#     log_dir = os.path.join(config.outdir,
#                            rf'{model_name}/log/{model_name}_{reduced_order}_{aesindy.__class__.__name__}_' + test)
#     weights_path = os.path.join(config.outdir,
#                                 rf'{model_name}/{model_name}_{reduced_order}_{aesindy.__class__.__name__}_' + test)
#     # log_dir = os.path.join(config.outdir,
#     #                        f'{model_name}/log/{model_name}_{reduced_order}_{aesindy.__class__.__name__}_{datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")}')
#     # weights_path = os.path.join(config.outdir,
#     #                             f'{model_name}/{model_name}_{reduced_order}_{aesindy.__class__.__name__}_{test}')

#     if load_model:
#         pass
#         # aesindy = VAESindy.load(aesindy.__class__, x=x_train, mu=None, mask=None, fixed_coeffs=None,
#         #                         path=weights_path,
#         #                         kwargs_overwrite=dict(l_rec=5e0, l_dz=5e-1, l_dx=1e-3, l1=5e-1, l2=1e-3, l_int=0))
#         # aesindy.load_weights(os.path.join(
#         #    config.outdir + f'/{model_name}/{model_name}_{reduced_order}' + test))  # beam_1_test_NF_minimal beam_1_test_NF #_test_NF_minimal_longtraining #_test_NF_minimal_mixed
#     else:
#         # callback
#         # aesindy.save(weights_path)
#         callbacks = []
#         callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_path, 'weights/'),
#                                                             # os.path.join(config.outdir + f'/{model_name}_{reduced_order}' + test),
#                                                             # test_NF_minimal_mixed #_test_NF_cubic_paper #_test_NF_arch_cubic
#                                                             save_weights_only=True,
#                                                             save_best_only=True,
#                                                             monitor='loss',
#                                                             verbose=0))
#         # tensorboard callback to visualize training
#         callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
#         # callback to update the sindy coefficients during training using pysindy
#         x_input = [x_train, dxdt_train]  # [x_train, dxdt_train, params_train]#[x_train, dxdt_train, x_int_train,  params_train, param_int_train]

#         # callbacks.append(
#         #     SindyCallback(x=x_train, dxdt=dxdt_train, dxddt=None, thresholder="l1", threshold=5e-3, print_precision=4,
#         #                   # 2.5
#         #                   mu=None, t=times_train[:n_timesteps], freq=200, train_end=True))
#         tf.random.set_seed(n_test)
#         trainhist_ae = aesindy.fit(x=x_input,
#                                 callbacks=callbacks,
#                                 y=None,
#                                 epochs=500,
#                                 batch_size=int(n_timesteps/10),  # n_timesteps,
#                                 verbose=1
#                                 )
#         print('AESINDy')
#         aesindy.print(
#             z=['z', 'dz'], precision=5
#         )
#         coeffs = aesindy.sindy_coeffs()

#         # sindy_coeffs, sindy_mean, sindy_log_var = aesindy.sindy_layer._coeffs
#         sindy_coeffs = aesindy.sindy_layer._coeffs.numpy()
#         # sindy_coeffs = sindy_coeffs.numpy()
#         # sindy_mean = sindy_mean.numpy()
#         # sindy_log_var = sindy_log_var.numpy()

#         # save model
#         if not os.path.isdir(os.path.join(config.outdir, f'{model_name}')):
#             os.mkdir(os.path.join(config.outdir, f'{model_name}'))
#         # aesindy.save(weights_path)
#         vaesindy.save_weights(weights_path)

# %% VAE SINDY
n_dof = x_train.shape[1]
for n_test in range(n_tests):
    kernel_regularizer = tf.keras.regularizers.L1L2(l1=0, l2=0)  # 1e-2 #5e-4
    vsindy_layer = VariationalSindyLayer(
        # SindyLayer(
        beta=2e-5,
        priors=Laplace(0.0, 1.0),
        state_dim=reduced_order,
        param_dim=0,
        kernel_regularizer=kernel_regularizer,
        feature_libraries=libraries,
        second_order=second_order,
        param_feature_libraries=param_libraries,
        x_mu_interaction=False,
        mask=None,
        fixed_coeffs=None,
        dtype=tf.float32,
    )
    # sindy_coeffs, sindy_mean, sindy_log_var = sindy_layer._coeffs

    vaesindy = VAESindy(
        sindy_layer=vsindy_layer,
        beta=1e-4 * reduced_order / n_dof,  #!!! 1e-3
        reduced_order=reduced_order,
        x=x_train,
        mu=None,
        scaling="none",
        layer_sizes=[32, 16, 8],
        activation="elu",
        l_rec=1e-2,
        l_dz=4e0,
        l_dx=1e-2,
        l1=0,
        l2=0,
        l_int=0,
        dt=dt,  #!!! TODO
        second_order=second_order,
    )

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)  # 3e-3)
    sindy_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=8e-4)  # 1e-3
    vaesindy.compile(
        optimizer=optimizer,
        sindy_optimizer=sindy_optimizer,  #!!!
        loss="mse",
        # run_eagerly=True,
    )

    logging.info("Training model")

    # test = 'VAE_9ICs_9'
    # test = 'VAE_9ICs_10' #!!!
    test = "VAE_9ICs_11"
    test = "VAE_15ICs_0"
    test = "VAE_20ICs_0"
    test = "VAE_20ICs_short_0"  # 2e-5
    test = "VAE_20ICs_short_0_noise"
    test = "VAE_20ICs_short_0_noise1"  # 2e-5
    # test = 'VAE_20ICs_short_1' #more sparsity 3e-5, bigger batch size
    # test = 'VAE_15ICS_1'
    # 0: individual, pod, gaussian, 1e-2
    # 1: individual, pod, laplacian, no reg
    # 1: individual, pod, gaussian, 1e-3
    # 2: indiivdual, pod, gaussian, (l1=3e-3, l2=0) l_rec=1e0, l_dz=5e0, l_dx=5e-2, l1=0, l2=0, l_int=0, GOOODDD
    # 9ICs_2 indiivudal, pod, gaussina, (5e-4), ....
    #'VAE_9ICs_3' none, pod, gaussian, 1e-3
    #'VAE_9ICs_4'none, pod, gaussian, 5e-4 l_rec=5e-2, l_dz=1e0, l_dx=1e-2,
    # Arch_GS_9_constrained' #Arch_GS_9_0_NF #Arch_GS_9_0 #Arch_GS_9_0_5order
    log_dir = os.path.join(
        config.outdir,
        f"{model_name}/log/{model_name}_{reduced_order}_{vaesindy.__class__.__name__}",
    )
    weights_path = os.path.join(
        config.outdir,
        f"{model_name}/{model_name}_{reduced_order}_{vaesindy.__class__.__name__}_{test}",
    )

    if load_model:
        vaesindy.load_weights(os.path.join(weights_path, "weights/"))
        # aesindy.load_weights(os.path.join(
        #    config.outdir + f'/{model_name}/{model_name}_{reduced_order}' + test))  # beam_1_test_NF_minimal beam_1_test_NF #_test_NF_minimal_longtraining #_test_NF_minimal_mixed
        vaesindy.define_scaling(x_train)
        x_train_scaled, dxdt_train_scaled = vaesindy.scale(x_train), vaesindy.scale(
            dxdt_train
        )
        x_test_scaled, dxdt_test_scaled = vaesindy.scale(x_test), vaesindy.scale(
            dxdt_test
        )

        vaesindy.print(z=["z1", "z2"], precision=5)
        coeffs = vaesindy.sindy_coeffs()

        sindy_coeffs, sindy_mean, sindy_log_var = vaesindy.sindy_layer._coeffs
        sindy_coeffs = sindy_coeffs.numpy()
        sindy_mean = sindy_mean.numpy()
        sindy_log_var = sindy_log_var.numpy()

    else:
        # callback
        # vaesindy.save(weights_path)
        callbacks = []
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(weights_path, "weights/"),
                # os.path.join(config.outdir + f'/{model_name}_{reduced_order}' + test),
                # test_NF_minimal_mixed #_test_NF_cubic_paper #_test_NF_arch_cubic
                save_weights_only=True,
                save_best_only=True,
                monitor="loss",
                verbose=0,
            )
        )
        # tensorboard callback to visualize training
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        )
        # callback to update the sindy coefficients during training using pysindy
        callbacks.append(SaveCoefficientsCallback())
        vaesindy.define_scaling(x_train)
        x_train_scaled, dxdt_train_scaled = vaesindy.scale(x_train), vaesindy.scale(
            dxdt_train
        )
        x_test_scaled, dxdt_test_scaled = vaesindy.scale(x_test), vaesindy.scale(
            dxdt_test
        )

        x_input = [
            x_train_scaled,
            dxdt_train_scaled,
        ]  # [x_train, dxdt_train, params_train]#[x_train, dxdt_train, x_int_train,  params_train, param_int_train]

        # callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
        # callbacks.append(SaveCoefficientsCallback())
        # callbacks.append(SindyCallback(x=x_train, dxdt=dxdt_train, dxddt=None, thresholder="l1", threshold=1e-4, print_precision=4,
        #                                 mu=None, t=times_train[:n_timesteps], freq=1500, train_end=False, ensemble=True,
        #                                 n_subsets=500))
        tf.random.set_seed(n_test)
        trainhist_vae = vaesindy.fit(
            x=x_input,
            callbacks=callbacks,
            y=None,
            epochs=2000,
            batch_size=int(n_timesteps / 10),  # n_timesteps,
            verbose=1,
        )
        # save model
        if save:
            if not os.path.isdir(os.path.join(config.outdir, f"{model_name}")):
                os.mkdir(os.path.join(config.outdir, f"{model_name}"))
            # aesindy.save(weights_path)
            vaesindy.save_weights(weights_path)
            #'C:\\Users\\paolo\\PhD\\Jonas\\AE2ndOrderSINDy\\aesindy\\resources\\arch_new\\output\\RD_noparam/RD_noparam_2_VAESindy_VAE_20ICs_short_0' !!!

vaesindy.print(z=["z1", "z2"], precision=3)
coeffs = vaesindy.sindy_coeffs()

sindy_coeffs, sindy_mean, sindy_log_var = vaesindy.sindy_layer._coeffs
sindy_coeffs = sindy_coeffs.numpy()
sindy_mean = sindy_mean.numpy()
sindy_log_var = sindy_log_var.numpy()
# save model

# Plot training history
# trainhists_ae.append(trainhist_ae)
# trainhists_vae.append(trainhist_vae)
# plot_train_hist([trainhist_ae,trainhist_vae])
# aesindy.vis_modes(x_test, 4)
vaesindy.vis_modes(x_test_scaled, 6)

# aesindy.vis_modes(x_train, 4)
vaesindy.vis_modes(x_train_scaled, 6)

# %%
# visualize coefficients in the range [-0.01, 0.01]
vsindy_layer._visualize_coefficients(sindy_mean, sindy_log_var)

# %% compare time derivatives
z = vaesindy.encoder(x_test_scaled).numpy()
# %%
# calc time derivatives
dzdt = np.gradient(z, times_test.squeeze(), axis=0)
dzddt = np.gradient(dzdt, times_test.squeeze(), axis=0)
z, dzdt2, dzddt2 = vaesindy.calc_latent_time_derivatives(
    x_test_scaled, dxdt_test_scaled, dxdt_test_scaled
)

plt.figure()
plt.plot(dzdt[:, 0])
plt.plot(dzdt2[:, 0])
plt.xlabel("time")
plt.ylabel("$\dot{z}_1$")
plt.legend(["dz1dt numerically", "dz1dt chain rule"])
plt.show()

plt.figure()
plt.plot(dzdt[:, 1])
plt.plot(dzdt2[:, 1])
plt.xlabel("time")
plt.ylabel("$\dot{z}_2$")
plt.legend(["dz2dt numerically", "dz2dt chain rule"])
plt.show()

# Comparison of the derivative of the first variable wrt to the second variable
plt.figure()
plt.plot(z[:, 0])
plt.plot(dzdt2[:, 1])
plt.xlabel("time")
plt.legend(["z2", "dz1dt chain rule"])
plt.show()

# %% compare time derivatives with sindy model
if second_order:
    z, dzdt2, dzddt2 = vaesindy.calc_latent_time_derivatives(
        x_test_scaled, dxdt_test_scaled, dxdt_test_scaled
    )
    s_dot_sindy = vaesindy.sindy(tf.concat([z, dzdt2], axis=1)).numpy()
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
    z, dzdt2 = vaesindy.calc_latent_time_derivatives(x_test_scaled, dxdt_test_scaled)
    s_dot_sindy = vaesindy.sindy(tf.concat([z], axis=1)).numpy()
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
errs = []
max_shift = 10
for i in range(len(ind_test)):
    for shift in range(max_shift):
        t_0 = i * int(n_timesteps) + shift
        t_end = t_0 + n_timesteps - shift
        sol = vaesindy.integrate(
            np.concatenate([z[t_0 : t_0 + 1]], axis=1).squeeze(),
            times_test[: n_timesteps - shift].squeeze(),
        )
        errs.append(np.mean(np.square(sol.y[0] - z[t_0:t_end, 0])))

plt.plot(errs)
plt.yscale("log")
print(errs.index(min(errs)))
ind = errs.index(min(errs))
start, shift = int(np.floor(ind / max_shift)), ind % max_shift
# %%
start, shift = int(np.floor(ind / max_shift)), ind % max_shift
# start, shift = 2, 8
t_0 = start * int(n_timesteps) + shift
t_end = t_0 + n_timesteps - shift
sol = vaesindy.integrate(
    np.concatenate([z[t_0 : t_0 + 1]], axis=1).squeeze(),
    times_test[: n_timesteps - shift].squeeze(),
)
plt.figure(figsize=[10, 6])
# subplot 1 for z
plt.subplot(2, 1, 1)
plt.plot(sol.t, z[t_0:t_end, 0], label="$z_1$ real", color="blue")
plt.plot(sol.t, sol.y[0], label="$z_1$ pred", linestyle="--", color="red")
plt.xlabel("time")
plt.ylabel("$z$")
plt.legend()
# subplot 2 for z_dot
plt.subplot(2, 1, 2)
plt.plot(sol.t, z[t_0:t_end, 1], label="$z_2$ real", color="blue")
plt.plot(sol.t, sol.y[1], label="$z_2$ pred", linestyle="--", color="red")
plt.xlabel("time")
plt.ylabel("$\dot{z}$")
plt.legend()
plt.show()


# %%
if not load_model:
    plt.figure(figsize=[8, 8])
    i = 0
    keys_to_ignore = ["coeffs_mean", "coeffs_scale"]

    plt.figure(figsize=(12, 12))
    i = 1
    for key in trainhist_vae.history.keys():
        if key not in keys_to_ignore:
            plt.subplot(3, 3, i)
            plt.semilogy(trainhist_vae.history[key])
            plt.title(key)
            i += 1
    plt.show()

# %%
x_rec = vaesindy.rescale(vaesindy.decoder(sol.y.T))

plt.figure(figsize=[10, 15])
for n_POD in range(4):
    plt.subplot(511 + n_POD)
    plt.plot(x_rec[:, n_POD], "b-", label="real")
    plt.plot(x_test[t_0:t_end, n_POD], "r--", label="pred")
plt.show(9)


plt.figure(figsize=[5, 5])
plt.plot(sol.y[0], sol.y[1])
plt.show()

# %%Physical reconsturction

x_physic = x_rec @ U.T
x_real_physic = x_test[t_0:t_end, :] @ U.T
_, n_dof_uv = x_physic.shape
U_physic = np.reshape(
    x_physic[:, : int(n_dof_uv / 2)], (n_timesteps - shift, Nx_hf, Ny_hf)
)
# V_physic = x_physic[:,int(n_dof_uv):].reshape(n_timesteps_test, Nx_hf, Ny_hf).T
U_real_physic = np.reshape(
    x_real_physic[:, : int(n_dof_uv / 2)], (n_timesteps - shift, Nx_hf, Ny_hf)
)
# V_real_physic = x_real_physic[:,int(n_dof_u2):].reshape(n_timesteps_test, Nx_hf, Ny_hf).T

for t in [300, -10]:
    fig = plt.figure(figsize=(17, 4))
    # plt.title('$t=' + str(t))
    # t = int(time/dt)
    # plt.title('$\mu = $' + str(round(mu_test[mu],3)) + '$, t = $' + str(round(t*dt,1)) +'\n' , fontsize = 24)
    # plt.axis('off')

    ax = fig.add_subplot(131)
    surf = ax.imshow(U_physic[t, :, :], origin="lower", vmin=-1, vmax=1)
    # plt.xticks(loc, tick,fontsize=20)
    # plt.yticks(loc, tick,fontsize=20)
    plt.xlabel("x", rotation=0, fontsize=22)
    plt.ylabel("y", rotation=0, fontsize=22, labelpad=13)
    title = "Pred"
    ax.set_title(title, fontsize=22)
    cbar = plt.colorbar(surf)
    cbar.ax.tick_params(labelsize=20, pad=1)

    ax = fig.add_subplot(132)
    surf = ax.imshow(U_real_physic[t, :, :], origin="lower", vmin=-1, vmax=1)
    # plt.xticks(loc, tick,fontsize=20)
    # plt.yticks(loc, tick,fontsize=20)
    plt.xlabel("x", rotation=0, fontsize=22)
    plt.ylabel("y", rotation=0, fontsize=22, labelpad=13)
    title = "Real"
    ax.set_title(title, fontsize=22)
    cbar = plt.colorbar(surf)
    cbar.ax.tick_params(labelsize=20, pad=1)

    ax = fig.add_subplot(133)
    surf = ax.imshow(
        np.abs(U_physic[t, :, :] - U_real_physic[t, :, :]), origin="lower", cmap="bwr"
    )
    # plt.xticks(loc, tick,fontsize=20)
    # plt.yticks(loc, tick,fontsize=20)
    plt.xlabel("x", rotation=0, fontsize=22)
    plt.ylabel("y", rotation=0, fontsize=22, labelpad=13)
    title = "Abs. error"
    ax.set_title(title, fontsize=22)
    cbar = plt.colorbar(surf)
    cbar.ax.tick_params(labelsize=20, pad=1)
    plt.show()

# %% Figure plots

# 3 time instances
t_test = [10, 25, 40]
ind_t_test = [int(t / dt - shift - 1) for t in t_test]

# Define the range for the x and y axes
x_range = np.linspace(-10, 10, 5, dtype=int)
x_loc = np.linspace(0, 49, 5, dtype=int)
fontsize = 18

fig = plt.figure(figsize=(19, 16))
for i, t in enumerate(ind_t_test):
    j = i * 3
    ax = fig.add_subplot(331 + j)
    surf = ax.imshow(
        U_physic[t, :, :],
        extent=[x_loc[0], x_loc[-1], x_loc[0], x_loc[-1]],
        origin="lower",
        vmin=-1,
        vmax=1,
        interpolation="bilinear",
    )
    plt.xlabel("x", rotation=0, fontsize=fontsize)
    plt.ylabel("y", rotation=0, fontsize=fontsize, labelpad=3)
    if i == 0:
        ax.set_title("VAE+VINDy pred.", fontsize=25, pad=14)
    cbar = plt.colorbar(surf)
    cbar.ax.tick_params(labelsize=fontsize, pad=1)
    plt.xticks(x_loc, x_range, fontsize=fontsize)
    plt.yticks(x_loc, x_range, fontsize=fontsize)
    ax.text(-28, 24, f"$t = {t_test[i]}$", fontsize=25, rotation=0)

    ax = fig.add_subplot(332 + j)
    surf = ax.imshow(
        U_real_physic[t, :, :],
        extent=[x_loc[0], x_loc[-1], x_loc[0], x_loc[-1]],
        origin="lower",
        vmin=-1,
        vmax=1,
        interpolation="bilinear",
    )
    plt.xlabel("x", rotation=0, fontsize=fontsize)
    plt.ylabel("y", rotation=0, fontsize=fontsize, labelpad=3)
    if i == 0:
        ax.set_title("Reference sol.", fontsize=25, pad=14)
    cbar = plt.colorbar(surf)
    cbar.ax.tick_params(labelsize=fontsize, pad=1)
    plt.xticks(x_loc, x_range, fontsize=fontsize)
    plt.yticks(x_loc, x_range, fontsize=fontsize)

    ax = fig.add_subplot(333 + j)
    surf = ax.imshow(
        np.abs(U_physic[t, :, :] - U_real_physic[t, :, :]),
        extent=[x_loc[0], x_loc[-1], x_loc[0], x_loc[-1]],
        origin="lower",
        cmap="bwr",
        interpolation="bilinear",
    )
    plt.xlabel("x", rotation=0, fontsize=fontsize)
    plt.ylabel("y", rotation=0, fontsize=fontsize, labelpad=3)
    if i == 0:
        ax.set_title("Abs. error", fontsize=25, pad=14)
    cbar = plt.colorbar(surf)
    cbar.ax.tick_params(labelsize=fontsize, pad=1)
    plt.xticks(x_loc, x_range, fontsize=fontsize)
    plt.yticks(x_loc, x_range, fontsize=fontsize)
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    ax.text(63, 24, f"       ", fontsize=25, rotation=0)

plt.show()


# %%
"""
Forward UQ:
- We sample SINDy coefficients from the predicted posterior distribution
- We integrate the ODE with the sampled coefficients and collect the trajectories
- Decode the trajectory and perform UQ in the original space
"""
n_traj = 100

# Store the original coefficients
kernel_orig, kernel_scale_orig = vsindy_layer.kernel, vsindy_layer.kernel_scale

# List to store the solution trajectories in latent space
sol_list = []

for traj in range(n_traj):
    # Sample from the posterior distribution of the coefficients
    sampled_coeff, _, _ = vaesindy.sindy_layer._coeffs

    # Assign the sampled coefficients to the SINDy layer
    vaesindy.sindy_layer.kernel = sampled_coeff

    # Integrate the ODE and store the solution
    # t_0 = 1 * int(n_timesteps)
    # z = vaesindy.encoder(x_test_scaled).numpy()
    _, _, z = vaesindy.variational_encoder(x_test_scaled)
    print(z[0, :])
    sol = vaesindy.integrate(
        np.concatenate([z[t_0 : t_0 + 1]], axis=1).squeeze(),
        times_test[: n_timesteps - shift].squeeze(),
    )
    sol_list.append(sol.y)

    # Restore the original coefficients
    vaesindy.sindy_layer.kernel, vaesindy.sindy_layer.kernel_scale = (
        kernel_orig,
        kernel_scale_orig,
    )

z_mean = np.mean(sol_list, axis=0).T
z_std = np.std(sol_list, axis=0).T
# %%
plt.figure(figsize=[10, 6])
# subplot 1 for z
plt.subplot(2, 1, 1)
plt.plot(sol.t, z[t_0:t_end, 0], label="$z_1$ real", color="red")
plt.plot(sol.t, z_mean[:, 0], label="$z_1$ pred mean", color="blue")
# do uncertainty bounds
plt.fill_between(
    sol.t,
    z_mean[:, 0] - 3 * z_std[:, 0],
    z_mean[:, 0] + 3 * z_std[:, 0],
    alpha=0.3,
    color="blue",
)

# for traj in range(n_traj):
#    plt.plot(sol.t, sol_list[traj][0], linestyle='--', color='red') #label='$z_1$ pred',

plt.xlabel("time")
plt.ylabel("$z$")
plt.legend()
# subplot 2 for z_dot
plt.subplot(2, 1, 2)
plt.plot(sol.t, z[t_0:t_end, 1], label="$z_1$ real", color="red")
plt.plot(sol.t, z_mean[:, 1], label="$z_1$ pred mean", color="blue")
# do uncertainty bounds
plt.fill_between(
    sol.t,
    z_mean[:, 1] - 3 * z_std[:, 1],
    z_mean[:, 1] + 3 * z_std[:, 1],
    alpha=0.3,
    color="blue",
)
plt.xlabel("time")
plt.ylabel("$\dot{z}$")
plt.legend()
plt.show()

# %% Decode all the trajectoreis
x_rec_list = [
    vaesindy.rescale(vaesindy.decoder(sol_list[traj].T)) for traj in range(n_traj)
]
x_rec_mean = np.mean(x_rec_list, axis=0)
x_rec_std = np.std(x_rec_list, axis=0)

# %%
plt.figure(figsize=[10, 8])
for n_POD in range(4):
    plt.subplot(411 + n_POD)
    # for traj in range(n_traj):
    #    plt.plot(sol.t,x_rec_list[traj][:,n_POD], 'b-', label = 'real')
    plt.plot(sol.t, x_rec[:, n_POD], "b-", label="real")
    plt.plot(sol.t, x_rec_mean[:, n_POD], label="pred mean", color="blue")
    plt.fill_between(
        sol.t,
        x_rec_mean[:, n_POD] - 3 * x_rec_std[:, n_POD],
        x_rec_mean[:, n_POD] + 3 * x_rec_std[:, n_POD],
        alpha=0.3,
        color="blue",
    )
    plt.plot(sol.t, x_test[t_0:t_end, n_POD], "r--", label="pred")
plt.show(9)


# %% Figure plots

# 3 time instances
t_test = [10, 25, 40]
ind_t_test = [int(t / dt - shift - 1) for t in t_test]

# Define the range for the x and y axes
x_range = np.linspace(-10, 10, 5, dtype=int)
x_loc = np.linspace(0, 49, 5, dtype=int)
fontsize = 18

fig = plt.figure(figsize=(19, 6))
for i, t in enumerate([ind_t_test[2]]):
    j = i * 4
    # For the time instant t reconstruct the full field solution for all trajectories
    x_rec_full_list = [x_rec_list[traj][t : t + 1, :] @ U.T for traj in range(n_traj)]
    x_rec_full_mean = np.mean(x_rec_full_list, axis=0)[0, : int(n_dof_uv / 2)].reshape(
        Nx_hf, Ny_hf
    )
    x_rec_full_std = np.std(x_rec_full_list, axis=0)[0, : int(n_dof_uv / 2)].reshape(
        Nx_hf, Ny_hf
    )

    ax = fig.add_subplot(141 + j)
    surf = ax.imshow(
        x_rec_full_mean,
        extent=[x_loc[0], x_loc[-1], x_loc[0], x_loc[-1]],
        origin="lower",
        vmin=-1,
        vmax=1,
        interpolation="bilinear",
    )
    plt.xlabel("x", rotation=0, fontsize=fontsize)
    plt.ylabel("y", rotation=0, fontsize=fontsize, labelpad=3)
    if i == 0:
        ax.set_title("VAE+VINDy mean.", fontsize=25, pad=14)
    cbar = plt.colorbar(surf)
    cbar.ax.tick_params(labelsize=fontsize, pad=1)
    plt.xticks(x_loc, x_range, fontsize=fontsize)
    plt.yticks(x_loc, x_range, fontsize=fontsize)
    ax.text(-28, 24, f"$t = {t_test[i]}$", fontsize=25, rotation=0)

    ax = fig.add_subplot(142 + j)
    surf = ax.imshow(
        U_real_physic[t, :, :],
        extent=[x_loc[0], x_loc[-1], x_loc[0], x_loc[-1]],
        origin="lower",
        vmin=-1,
        vmax=1,
        interpolation="bilinear",
    )
    plt.xlabel("x", rotation=0, fontsize=fontsize)
    plt.ylabel("y", rotation=0, fontsize=fontsize, labelpad=3)
    if i == 0:
        ax.set_title("Reference sol.", fontsize=25, pad=14)
    cbar = plt.colorbar(surf)
    cbar.ax.tick_params(labelsize=fontsize, pad=1)
    plt.xticks(x_loc, x_range, fontsize=fontsize)
    plt.yticks(x_loc, x_range, fontsize=fontsize)

    ax = fig.add_subplot(143 + j)
    surf = ax.imshow(
        np.abs(U_physic[t, :, :] - U_real_physic[t, :, :]),
        extent=[x_loc[0], x_loc[-1], x_loc[0], x_loc[-1]],
        origin="lower",
        cmap="bwr",
        interpolation="bilinear",
    )
    plt.xlabel("x", rotation=0, fontsize=fontsize)
    plt.ylabel("y", rotation=0, fontsize=fontsize, labelpad=3)
    if i == 0:
        ax.set_title("Abs. error", fontsize=25, pad=14)
    cbar = plt.colorbar(surf)
    cbar.ax.tick_params(labelsize=fontsize, pad=1)
    plt.xticks(x_loc, x_range, fontsize=fontsize)
    plt.yticks(x_loc, x_range, fontsize=fontsize)
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    ax.text(63, 24, f"       ", fontsize=25, rotation=0)

    ax = fig.add_subplot(144 + j)
    surf = ax.imshow(
        x_rec_full_std,
        extent=[x_loc[0], x_loc[-1], x_loc[0], x_loc[-1]],
        origin="lower",
        cmap="bwr",
        interpolation="bilinear",
    )
    plt.xlabel("x", rotation=0, fontsize=fontsize)
    plt.ylabel("y", rotation=0, fontsize=fontsize, labelpad=3)
    if i == 0:
        ax.set_title("VAE+VINDy std", fontsize=25, pad=14)
    cbar = plt.colorbar(surf)
    cbar.ax.tick_params(labelsize=fontsize, pad=1)
    plt.xticks(x_loc, x_range, fontsize=fontsize)
    plt.yticks(x_loc, x_range, fontsize=fontsize)
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    ax.text(63, 24, f"       ", fontsize=25, rotation=0)

plt.show()

# %%
x_rec_full_list = [x_rec_list[traj] @ U.T for traj in range(n_traj)]
x_rec_full_mean = np.mean(x_rec_full_list, axis=0)
x_rec_full_std = np.std(x_rec_full_list, axis=0)
x_rec_full_abserr = np.abs(x_rec_full_mean - x_test[t_0:t_end, :] @ U.T)
# %%
# Plot std and abserror
fig = plt.figure(figsize=(19, 6))
plt.plot(x_rec_full_std[:, 0] * 10, label="std")
plt.plot(x_rec_full_abserr[:, 0], label="abs error")
plt.yscale("log")
plt.legend()
plt.show()
# %% Modes
U_plot = np.reshape(U[: int(n_dof_uv / 2)].T, (pca_order, Nx_hf, Ny_hf))
fig = plt.figure()
for n_POD in range(9):
    ax = fig.add_subplot(331 + n_POD)
    surf = ax.imshow(U_plot[n_POD, :, :], origin="lower")  # , vmin=-1, vmax=1)
    plt.xlabel("x", rotation=0, fontsize=22)
    plt.ylabel("y", rotation=0, fontsize=22, labelpad=13)
plt.show()

# %%
aesindy = vaesindy
for n_test in range(1):
    sindy_loss = True
    zs = []
    dzdts = []
    dzddts = []
    n_ics = int(x_train_scaled.shape[0] / n_timesteps)
    for i in range(n_ics):
        print(i)
        ids = np.arange(i * n_timesteps, (i + 1) * n_timesteps)
        if sindy_loss:
            z, dzdt = aesindy.calc_latent_time_derivatives(
                x_train_scaled[ids], dxdt_train_scaled[ids]
            )
            zs.append(z)
            dzdts.append(dzdt)
            # dzddts.append(dzddt)
        else:
            z = np.array(aesindy.variational_encoding(x_train_scaled[ids]))
            zs.append(z)

    z = np.concatenate(zs, axis=0)
    dzdt = np.concatenate(dzdts, axis=0)
    # dzddt =  np.concatenate(dzddts, axis=0)

    zs_test = []
    dzdts_test = []
    dzddts_test = []
    n_ics_test = 2
    for i in range(n_ics_test):
        print(i)
        ids = np.arange(i * n_timesteps, (i + 1) * n_timesteps)
        if sindy_loss:
            z_test, dzdt_test = aesindy.calc_latent_time_derivatives(
                x_test_scaled[ids], dxdt_test_scaled[ids]
            )
            zs_test.append(z_test)
            dzdts_test.append(dzdt_test)
            # dzddts_test.append(dzddt_test)
        else:
            z_test = np.array(aesindy.variational_encoding(x_test_scaled[ids]))
            zs_test.append(z_test)

    z_test = np.concatenate(zs_test, axis=0)
    dzdt_test = np.concatenate(dzdts_test, axis=0)
    # dzddt_test  =  np.concatenate(dzddts_test, axis=0)

    import pysindy as ps

    X = []
    X_dot = []
    X_dot_num = []
    U_ = []
    for i in range(len(zs)):
        X.append(zs[i])
        X_dot_num.append(np.gradient(zs[i], dt)[0])
        X_dot.append(dzdts[i])
        # X.append(np.stack((zs_test[i][:,0], dzdts_test[i][:,0]), axis = 1))
        # X_dot.append(np.stack((dzdts_test[i][:,0],dzddts_test[i][:,0]), axis = 1))
        # U_.append((params_train[i * n_timesteps:(i + 1) * n_timesteps]).reshape(-1, 1))

    X_test = []
    X_dot_test_num = []
    X_dot_test = []
    U_test = []
    for i in range(len(zs_test)):
        X_test.append(zs_test[i])
        X_dot_test_num.append(np.gradient(zs_test[i], dt)[0])
        X_dot_test.append(dzdts_test[i])
        # U_test.append((params_test[i * n_timesteps:(i + 1) * n_timesteps]).reshape(-1, 1))

    coeffs = aesindy.sindy_coeffs()[0]
    model = ps.SINDy(
        feature_names=["z1", "z2"],
        feature_library=ps.PolynomialLibrary(degree=3),
        optimizer=ps.STLSQ(threshold=1e-4),
    )  # , initial_guess =coeffs))
    model.fit(X, t=dt, multiple_trajectories=True, x_dot=X_dot)

    model.print(precision=4)

z, z_dot = aesindy.calc_latent_time_derivatives(x_test_scaled, dxdt_test_scaled)
t_0 = 0 * int(n_timesteps)
sol = aesindy.integrate(
    np.concatenate([z[t_0 : t_0 + 1]], axis=1).squeeze(),
    times_train[:n_timesteps].squeeze(),
    mu=None,
)
plt.plot(sol.t, sol.y[0])
plt.plot(sol.t, z[t_0 : t_0 + n_timesteps, 0])
plt.legend(["integrated", "true"])

n_timesteps_test = n_timesteps
for traj in range(len(X_test)):
    pred = model.simulate(X_test[traj][0, :], t=times)
    pred_dt = model.predict(X_test[traj])

    plt.figure(figsize=[20, 8])
    plt.subplot(221)
    plt.plot(X_dot_test[traj][:, 0], "b-", label="real dz1/dt")
    plt.plot(pred_dt[:, 0], "r--", label="pred dz1/dt")
    plt.legend(loc="upper right")

    plt.subplot(222)
    plt.plot(X_dot_test[traj][:, 1], "b-", label="real dz2/dt")
    plt.plot(pred_dt[:, 1], "r--", label="pred dz2/dt")
    plt.legend(loc="upper right")

    plt.subplot(223)
    plt.plot(X_test[traj][:, 0], "b-", label="real z1")
    plt.plot(pred[:, 0], "r--", label="pred z1")
    plt.legend()

    plt.subplot(224)
    plt.plot(X_test[traj][:, 1], "b-", label="real z2")
    plt.plot(pred[:, 1], "r--", label="pred z2")
    plt.legend(loc="upper right")

    plt.show()

    plt.figure()
    plt.plot(X_test[traj][:, 0], X_test[traj][:, 1], "b-", label="real")
    plt.plot(pred[:, 0], pred[:, 1], "r--", label="pred")
    plt.legend(loc="upper right")
    plt.show()

    x_rec = aesindy.decoder(pred)

    dofs = [0, 1]

    plt.figure(figsize=[20, 4])
    plt.subplot(121)
    plt.plot(
        x_test[traj * n_timesteps_test : (traj + 1) * n_timesteps_test, dofs[0]],
        "b-",
        label="real",
    )
    plt.plot(x_rec[:, dofs[0]], "r--", label="pred")
    plt.title("dof " + str(dofs[0] + 1))
    plt.legend(loc="upper right")

    plt.subplot(122)
    plt.plot(
        x_test[traj * n_timesteps_test : (traj + 1) * n_timesteps_test, dofs[1]],
        "b-",
        label="real",
    )
    plt.plot(x_rec[:, dofs[1]], "r--", label="pred")
    plt.title("dof " + str(dofs[1] + 1))
    plt.legend(loc="upper right")

    plt.show()

    plt.figure(figsize=[20, 12])
    for k in range(6):
        plt.subplot(231 + k)
        plt.plot(
            x_test[traj * n_timesteps_test : (traj + 1) * n_timesteps_test, k],
            "b-",
            label="real",
        )
        plt.plot(x_rec[:, k], "r--", label="pred")
        plt.title("dof " + str(k + 1))
        plt.legend(loc="upper right")

    plt.show()

    # Physical reconstruction
    x_rec = x_rec.numpy()
    x_real = x_test[traj * n_timesteps_test : (traj + 1) * n_timesteps_test, :]
    if preprocess:
        x_rec = x_rec @ (U_sigma @ V_sigma / np.sqrt(S)).T
        x_real = x_real @ (U_sigma @ V_sigma / np.sqrt(S)).T

    x_physic = x_rec @ U.T
    x_real_physic = x_real @ U.T
    _, n_dof_uv = x_physic.shape
    U_physic = (
        x_physic[:, : int(n_dof_uv / 2)].reshape(n_timesteps_test, Nx_hf, Ny_hf).T
    )
    V_physic = (
        x_physic[:, int(n_dof_uv / 2) :].reshape(n_timesteps_test, Nx_hf, Ny_hf).T
    )
    U_real_physic = (
        x_real_physic[:, : int(n_dof_uv / 2)].reshape(n_timesteps_test, Nx_hf, Ny_hf).T
    )
    V_real_physic = (
        x_real_physic[:, int(n_dof_uv / 2) :].reshape(n_timesteps_test, Nx_hf, Ny_hf).T
    )
    # x_real_physic = (x_real @ U.T).reshape(n_timesteps_test, Nx_hf, Ny_hf).T

    # # uMF_LSTM_test_rec = (uMF_LSTM_test @ POM_u.T).reshape(N_mu_test, Nt_test, Nx_hf, Ny_hf).T

    for t in [300, -1]:
        fig = plt.figure(figsize=(17, 4))
        # plt.title('$t=' + str(t))
        # t = int(time/dt)
        # plt.title('$\mu = $' + str(round(mu_test[mu],3)) + '$, t = $' + str(round(t*dt,1)) +'\n' , fontsize = 24)
        # plt.axis('off')

        ax = fig.add_subplot(131)
        surf = ax.imshow(U_physic[:, :, t], origin="lower", vmin=-1, vmax=1)
        # plt.xticks(loc, tick,fontsize=20)
        # plt.yticks(loc, tick,fontsize=20)
        plt.xlabel("x", rotation=0, fontsize=22)
        plt.ylabel("y", rotation=0, fontsize=22, labelpad=13)
        title = "Pred"
        ax.set_title(title, fontsize=22)
        cbar = plt.colorbar(surf)
        cbar.ax.tick_params(labelsize=20, pad=1)

        ax = fig.add_subplot(132)
        surf = ax.imshow(U_real_physic[:, :, t], origin="lower", vmin=-1, vmax=1)
        # plt.xticks(loc, tick,fontsize=20)
        # plt.yticks(loc, tick,fontsize=20)
        plt.xlabel("x", rotation=0, fontsize=22)
        plt.ylabel("y", rotation=0, fontsize=22, labelpad=13)
        title = "Real"
        ax.set_title(title, fontsize=22)
        cbar = plt.colorbar(surf)
        cbar.ax.tick_params(labelsize=20, pad=1)

        ax = fig.add_subplot(133)
        surf = ax.imshow(
            np.abs(U_physic[:, :, t] - U_real_physic[:, :, t]),
            origin="lower",
            cmap="bwr",
        )
        # plt.xticks(loc, tick,fontsize=20)
        # plt.yticks(loc, tick,fontsize=20)
        plt.xlabel("x", rotation=0, fontsize=22)
        plt.ylabel("y", rotation=0, fontsize=22, labelpad=13)
        title = "Abs. error"
        ax.set_title(title, fontsize=22)
        cbar = plt.colorbar(surf)
        cbar.ax.tick_params(labelsize=20, pad=1)
        plt.show()

# %%
n_timesteps_test = n_timesteps
for traj in range(len(X)):
    pred = model.simulate(X[traj][0, :], t=times)
    pred_dt = model.predict(X[traj])

    plt.figure(figsize=[20, 8])
    plt.subplot(221)
    plt.plot(X_dot_num[traj][:, 0], "b-", label="real dz1/dt")
    plt.plot(pred_dt[:, 0], "r--", label="pred dz1/dt")
    plt.legend(loc="upper right")

    plt.subplot(222)
    plt.plot(X_dot_num[traj][:, 1], "b-", label="real dz2/dt")
    plt.plot(pred_dt[:, 1], "r--", label="pred dz2/dt")
    plt.legend(loc="upper right")

    plt.subplot(223)
    plt.plot(X[traj][:, 0], "b-", label="real z1")
    plt.plot(pred[:, 0], "r--", label="pred z1")
    plt.legend()

    plt.subplot(224)
    plt.plot(X[traj][:, 1], "b-", label="real z2")
    plt.plot(pred[:, 1], "r--", label="pred z2")
    plt.legend(loc="upper right")

    plt.show()

    plt.figure()
    plt.plot(X[traj][:, 0], X[traj][:, 1], "b-", label="real")
    plt.plot(pred[:, 0], pred[:, 1], "r--", label="pred")
    plt.legend(loc="upper right")
    plt.show()

    x_rec = aesindy.decoder(pred)

    dofs = [0, 1]

    plt.figure(figsize=[20, 4])
    plt.subplot(121)
    plt.plot(
        x_train[traj * n_timesteps_test : (traj + 1) * n_timesteps_test, dofs[0]],
        "b-",
        label="real",
    )
    plt.plot(x_rec[:, dofs[0]], "r--", label="pred")
    plt.title("dof " + str(dofs[0] + 1))
    plt.legend(loc="upper right")

    plt.subplot(122)
    plt.plot(
        x_train[traj * n_timesteps_test : (traj + 1) * n_timesteps_test, dofs[1]],
        "b-",
        label="real",
    )
    plt.plot(x_rec[:, dofs[1]], "r--", label="pred")
    plt.title("dof " + str(dofs[1] + 1))
    plt.legend(loc="upper right")

    plt.show()

    plt.figure(figsize=[20, 12])
    for k in range(6):
        plt.subplot(231 + k)
        plt.plot(
            x_train[traj * n_timesteps_test : (traj + 1) * n_timesteps_test, k],
            "b-",
            label="real",
        )
        plt.plot(x_rec[:, k], "r--", label="pred")
        plt.title("dof " + str(k + 1))
        plt.legend(loc="upper right")

    plt.show()

    # Physical reconstruction
    x_rec = x_rec.numpy()
    x_real = x_train[traj * n_timesteps_test : (traj + 1) * n_timesteps_test, :]
    if preprocess:
        x_rec = x_rec @ (U_sigma @ V_sigma / np.sqrt(S)).T
        x_real = x_real @ (U_sigma @ V_sigma / np.sqrt(S)).T

    x_physic = x_rec @ U.T
    x_real_physic = x_real @ U.T
    _, n_dof_uv = x_physic.shape
    U_physic = (
        x_physic[:, : int(n_dof_uv / 2)].reshape(n_timesteps_test, Nx_hf, Ny_hf).T
    )
    V_physic = (
        x_physic[:, int(n_dof_uv / 2) :].reshape(n_timesteps_test, Nx_hf, Ny_hf).T
    )
    U_real_physic = (
        x_real_physic[:, : int(n_dof_uv / 2)].reshape(n_timesteps_test, Nx_hf, Ny_hf).T
    )
    V_real_physic = (
        x_real_physic[:, int(n_dof_uv / 2) :].reshape(n_timesteps_test, Nx_hf, Ny_hf).T
    )
    # x_real_physic = (x_real @ U.T).reshape(n_timesteps_test, Nx_hf, Ny_hf).T

    # # uMF_LSTM_test_rec = (uMF_LSTM_test @ POM_u.T).reshape(N_mu_test, Nt_test, Nx_hf, Ny_hf).T

    for t in [300, 600]:
        fig = plt.figure(figsize=(17, 4))
        # plt.title('$t=' + str(t))
        # t = int(time/dt)
        # plt.title('$\mu = $' + str(round(mu_test[mu],3)) + '$, t = $' + str(round(t*dt,1)) +'\n' , fontsize = 24)
        # plt.axis('off')

        ax = fig.add_subplot(131)
        surf = ax.imshow(U_physic[:, :, t], origin="lower", vmin=-1, vmax=1)
        # plt.xticks(loc, tick,fontsize=20)
        # plt.yticks(loc, tick,fontsize=20)
        plt.xlabel("x", rotation=0, fontsize=22)
        plt.ylabel("y", rotation=0, fontsize=22, labelpad=13)
        title = "Pred"
        ax.set_title(title, fontsize=22)
        cbar = plt.colorbar(surf)
        cbar.ax.tick_params(labelsize=20, pad=1)

        ax = fig.add_subplot(132)
        surf = ax.imshow(U_real_physic[:, :, t], origin="lower", vmin=-1, vmax=1)
        # plt.xticks(loc, tick,fontsize=20)
        # plt.yticks(loc, tick,fontsize=20)
        plt.xlabel("x", rotation=0, fontsize=22)
        plt.ylabel("y", rotation=0, fontsize=22, labelpad=13)
        title = "Real"
        ax.set_title(title, fontsize=22)
        cbar = plt.colorbar(surf)
        cbar.ax.tick_params(labelsize=20, pad=1)

        ax = fig.add_subplot(133)
        surf = ax.imshow(
            np.abs(U_physic[:, :, t] - U_real_physic[:, :, t]),
            origin="lower",
            cmap="bwr",
        )
        # plt.xticks(loc, tick,fontsize=20)
        # plt.yticks(loc, tick,fontsize=20)
        plt.xlabel("x", rotation=0, fontsize=22)
        plt.ylabel("y", rotation=0, fontsize=22, labelpad=13)
        title = "Abs. error"
        ax.set_title(title, fontsize=22)
        cbar = plt.colorbar(surf)
        cbar.ax.tick_params(labelsize=20, pad=1)
        plt.show()
    # #########################   CREATE VIDEO    ##########################

    # fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))

    # camera = Camera(fig)
    # #mu = mus[0]
    # for i in range(0, 1600, 5):
    #     surf = ax[0].imshow(x_physic[:,:,i], vmin=-1, vmax=1)
    #     surf = ax[1].imshow(x_real_physic[:,:,i], vmin=-1, vmax=1)
    #     surf = ax[2].imshow(np.abs(x_real_physic[:,:,i]-x_physic[:,:,i]), origin = 'lower', cmap='bwr')

    #     ax[0].set_title("Pred.", fontsize = 20)
    #     ax[1].set_title("Real", fontsize = 20)
    #     ax[2].set_title("Abs. error", fontsize = 20)

    #     camera.snap()
    # #fig.suptitle("$\mu$ = " + str(round(mu_test[mu],3)), fontsize = 22)
    # animation = camera.animate()

    # # Close the figure to prevent it from being displayed
    # #plt.close(fig)

    # # Save the animation as a GIF using PillowWriter
    # name = f"{model_name}_{round(U_test[traj][0][0],3)}.gif"
    # animation.save(name, writer=PillowWriter(fps=10))

    # # Display the saved GIF
    # HTML("<img src='" + name + "'>")
