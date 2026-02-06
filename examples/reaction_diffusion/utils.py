import logging
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from vindy.utils import *
import pickle

import mat73
import scipy.io as sio
from sklearn.utils import extmath
import datetime
from sklearn.model_selection import train_test_split
import time

# Add the examples folder to the Python path (kept for compatibility if callers need it)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import shared utilities
try:
    from utils import validate_data_path
except ImportError:
    # Fallback if examples/utils.py not found
    def validate_data_path(data_path, zenodo_doi="10.5281/zenodo.18313843"):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"Data file {data_path} not found. "
                f"Please download the file from Zenodo (http://doi.org/{zenodo_doi}) and "
                f"specify the correct path in the examples/config.py file."
            )


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


def load_reaction_diffusion_data(
    data_paths,
    nth_time_step=1,
    end_time_step=None,
    pca_order=64,
    short=True,
    pod=True,
    seed=123,
    preprocess=False,
    noise=True,
):
    """
    Load and preprocess reaction-diffusion data from a .mat file.
    Args:
        data_paths (str): Path to the .mat file containing the data.
        nth_time_step (int): Step size for time reduction.
        end_time_step (int or None): End time step for data slicing.
        pca_order (int): Number of principal components for PCA.
        short (bool): Whether to use a shorter time series.
        pod (bool): Whether to apply POD for dimensionality reduction.
        seed (int): Random seed for reproducibility.
        preprocess (bool): Whether to apply preprocessing to the data.
        noise (bool): Whether to add noise to the training data.
    Returns:
        times_train (np.ndarray): Training time data.
    """

    # Validate data path
    validate_data_path(data_paths)

    with open(data_paths, "rb") as f:
        data = pickle.load(f)
        x = data["x"]
        time = data["t"]

    n_sims, n_timesteps, Nx_hf, Ny_hf, n_channels = x.shape
    n_timesteps_train = int(n_timesteps / 2)
    dt = time[1] - time[0]

    x_train, x_test = train_test_split(x, test_size=0.2, random_state=seed)
    n_sims_train, n_sims_test = x_train.shape[0], x_test.shape[0]

    # reduce time steps for training data
    x_train = x_train[:, :n_timesteps_train]

    # apply noise
    mu = 0
    sigma = 0.2
    scale_noise = np.exp(mu)
    x_train = x_train * (
        np.random.lognormal(mean=0, sigma=sigma, size=x_train.shape) * scale_noise
    )

    # compute time derivatives of noisy data
    dxdt_train = np.gradient(x_train, dt, axis=1, edge_order=2)
    # compute time derivatives of clean test data
    dxdt_test = np.gradient(x_test, dt, axis=1, edge_order=2)

    # create data matrix S with shape (channels * N, n_sims_train * n_timesteps_train)
    x_train_2d = switch_data_format(
        x_train, n_sims_train, n_timesteps_train, target_format="2d"
    )  # (samples, features == channels * N)

    # perform pca on the noisy data
    logging.info("Performing PCA on the noisy data")
    pca = PCA(n_components=pca_order)
    pca.fit(x_train_2d)
    V = pca.components_.T

    x_pca_2d = pca.transform(x_train_2d)
    x_rec_2d = pca.inverse_transform(x_pca_2d)
    x_rec = switch_data_format(
        x_rec_2d,
        n_sims_train,
        n_timesteps_train,
        spatial_shape=(Nx_hf, Ny_hf, n_channels),
        target_format="5d",
    )

    dxdt_2d = switch_data_format(
        dxdt_train, n_sims_train, n_timesteps_train, target_format="2d"
    )
    dxdt_rec_2d = pca.inverse_transform(pca.transform(dxdt_2d))
    dxdt_rec = switch_data_format(
        dxdt_rec_2d,
        n_sims_train,
        n_timesteps_train,
        spatial_shape=(Nx_hf, Ny_hf, n_channels),
        target_format="5d",
    )
    plt.title("Noisy sample")
    plt.imshow(dxdt_train[0, 0, :, :, 0])
    plt.show()
    plt.title("Rec sample")
    plt.imshow(dxdt_rec[0, 0, :, :, 0])
    plt.show()

    plt.title("Noisy sample")
    plt.imshow(x_train[0, 0, :, :, 0])
    plt.show()
    plt.title("Rec sample")
    plt.imshow(x_rec[0, 0, :, :, 0])
    plt.show()

    # reconstruct and visualize a sample to verify reshapes
    # original training data reshaped to (n_sims_train, n_timesteps_train, Nx, Ny, channels)
    time_train = np.tile(time, x.shape[0]).reshape(-1, 1)
    time_test = np.tile(time, x_test.shape[0]).reshape(-1, 1)

    # reduce data to pca modes
    dxdt_pca_2d = pca.transform(dxdt_2d)
    x_pca_test_2d = pca.transform(
        switch_data_format(x_test, n_sims_test, n_timesteps, target_format="2d")
    )
    dxdt_pca_test_2d = pca.transform(
        switch_data_format(dxdt_test, n_sims_test, n_timesteps, target_format="2d")
    )

    return (
        time_train,
        x_pca_2d,
        dxdt_pca_2d,
        time_test,
        x_pca_test_2d,
        dxdt_pca_test_2d,
        pca,
        (Nx_hf, Ny_hf, n_channels),
        x_test,
        n_sims_train,
        n_timesteps_train,
        n_sims_test,
        n_timesteps,
    )
