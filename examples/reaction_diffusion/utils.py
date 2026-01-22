import logging
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from vindy.utils import *
import pickle

import mat73
import scipy.io
from sklearn.utils import extmath
import datetime
from sklearn.model_selection import train_test_split
import time

# Add the examples folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


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


def reshape_ae(data, n_timesteps, reduce=False):
    if reduce:
        data = data[:, :n_timesteps, :]
    return data.reshape(-1, data.shape[2])


def load_reactiondiffusion_data(
    data_paths,
    nth_time_step=1,
    end_time_step=None,
    n_int=0,
    pca_order=64,
    short=True,
    pod=True,
    seed=123,
    preprocess=False,
    noise=True,
):

    print("Loading data from ", data_paths)
    data = mat73.loadmat(data_paths)
    print("Data loaded.")
    # data = np.load(data_paths, allow_pickle=True)

    times = data["t"]
    dt = times[1] - times[0]
    x_full = data["U"]
    Nx_hf, Ny_hf, n_timesteps, n_sims = x_full.shape
    N = Nx_hf * Ny_hf
    n_timesteps_train = int(n_timesteps / 2)

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
            sigma = 0.2
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

    (
        x_train,
        x_test,
        dxdt_train,
        dxdt_test,
        dxddt_train,
        dxddt_test,
        ind_train,
        ind_test,
    ) = train_test_split(x, dxdt, dxddt, indexes, test_size=0.2, random_state=seed)

    times_train = np.tile(times, x_train.shape[0]).reshape(-1, 1)
    times_test = np.tile(times, x_test.shape[0]).reshape(-1, 1)

    def reshape_ae(
        data,
        reduce=False,
        end_time_step=end_time_step,
        nth_time_step=nth_time_step,
        n_timesteps=n_timesteps,
    ):
        if end_time_step is None:
            end_time_step = n_timesteps

        if reduce:
            data = data[:, 0:end_time_step:nth_time_step, :]
        return data.reshape(-1, data.shape[2])

    data_ = [x_train, dxdt_train, dxddt_train]  # , x_int_train, dx_int_train]

    for i in range(len(data_)):
        data_[i] = reshape_ae(data_[i], n_timesteps=n_timesteps)
    x_train, dxdt_train, dxddt_train = data_

    data_ = [x_test, dxdt_test, dxddt_test]  # , x_int_test, dx_int_test]

    for i in range(len(data_)):
        data_[i] = reshape_ae(data_[i])

    x_test, dxdt_test, dxddt_test = data_

    if preprocess:
        np.random.seed(seed)
        Sigma = np.random.rand(pca_order, pca_order)
        U_sigma, S_sigma, V_sigma = compute_randomized_SVD(
            Sigma, pca_order, pca_order, 1
        )
        Sigma = U_sigma @ V_sigma
        Sigma = Sigma * np.sqrt(S)

        x_train = x_train @ Sigma
        dxdt_train = dxdt_train @ Sigma
        x_test = x_test @ Sigma
        dxdt_test = dxdt_test @ Sigma
        # preprocess = False

    return (
        times_train,
        x_train,
        dxdt_train,
        times_test,
        x_test,
        dxdt_test,
        U,
        n_sims,
        n_timesteps,
    )
