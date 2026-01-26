import logging
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
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

    if not os.path.isfile(data_paths):
        raise FileNotFoundError(
            f"Data file {data_paths} not found. "
            f"Please download the file from Zenodo (http://doi.org/10.5281/zenodo.18313843) and "
            f"specify the correct path in the examples/config.py file."
        )

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
        V,
        n_sims_train,
        n_timesteps_train,
        n_sims_test,
        n_timesteps,
    )


def switch_data_format(
    data, n_sims, n_timesteps, spatial_shape=None, target_format="auto"
):
    """
    Convert between vectorized (2D), simulation-wise flattened (3D), and full spatial (5D) data formats.

    Parameters
    - data: np.ndarray. One of:
        * 2D: (n_sims * n_timesteps, features)
        * 3D: (n_sims, n_timesteps, features)
        * 5D: (n_sims, n_timesteps, Nx, Ny, channels)
    - n_sims: int, number of simulations
    - n_timesteps: int, timesteps per simulation
    - spatial_shape: optional tuple describing spatial dims. Accepts (Nx, Ny, channels) or (N, channels) or (Nx, Ny).
      When converting to/from 5D, this is required unless it can be inferred unambiguously from feature size.
    - target_format: 'auto' (default), '2d', '3d', or '5d'. When 'auto', the function chooses a sensible target based on input.

    Returns
    - Converted np.ndarray in requested format.

    Examples
    - 2D -> 5D: provide spatial_shape=(Nx,Ny,channels) and target_format='5d'
    - 5D -> 2D: target_format='2d' or rely on 'auto' to get 3D flattened by default
    """
    if data is None:
        return None

    if target_format not in ("auto", "2d", "3d", "5d"):
        raise ValueError("target_format must be one of 'auto','2d','3d','5d'")

    # helpers to interpret spatial_shape
    def infer_spatial_from_features(features):
        # try to infer (Nx, Ny, channels) if possible
        if spatial_shape is not None:
            if len(spatial_shape) == 3:
                return tuple(spatial_shape)
            if len(spatial_shape) == 2:
                # could be (N, channels) or (Nx, Ny)
                return tuple(spatial_shape)
        return None

    # Input is vectorized 2D: (n_sims * n_timesteps, features)
    if data.ndim == 2 and data.shape[0] == n_sims * n_timesteps:
        if target_format == "2d" or (target_format == "auto" and data.ndim == 2):
            return data
        features = data.shape[1]
        if target_format == "5d":
            if spatial_shape is None:
                raise ValueError(
                    "spatial_shape (Nx,Ny,channels) is required to reshape to 5D"
                )
            # accept (Nx,Ny,channels) or (N,channels)
            if len(spatial_shape) == 3:
                Nx, Ny, channels = spatial_shape
            elif len(spatial_shape) == 2:
                # (N, channels)
                N, channels = spatial_shape
                # try to factor N into Nx,Ny by assuming square grid
                Nx = int(np.sqrt(N))
                if Nx * Nx != N:
                    raise ValueError(
                        "Cannot infer Nx,Ny from N; provide (Nx,Ny,channels)"
                    )
                Ny = Nx
            else:
                raise ValueError("spatial_shape must be length 2 or 3")
            if features != Nx * Ny * channels:
                raise ValueError(
                    f"Feature size {features} does not match provided spatial_shape {spatial_shape}"
                )
            return data.reshape(n_sims, n_timesteps, Nx, Ny, channels)
        # default: to 3D flattened features
        return data.reshape(n_sims, n_timesteps, -1)

    # Input is simulation-wise flattened 3D: (n_sims, n_timesteps, features)
    if data.ndim == 3 and data.shape[0] == n_sims and data.shape[1] == n_timesteps:
        if target_format == "3d" or (
            target_format == "auto" and data.ndim == 3 and spatial_shape is None
        ):
            return data
        if target_format == "2d" or (
            target_format == "auto" and data.ndim == 3 and spatial_shape is None
        ):
            return data.reshape(-1, data.shape[-1])
        # convert to 5D
        features = data.shape[2]
        if spatial_shape is None:
            # try infer square grid and single channel
            Nx = int(np.sqrt(features))
            if Nx * Nx == features:
                Ny = Nx
                channels = 1
            else:
                raise ValueError("spatial_shape required to reshape 3D to 5D")
        else:
            if len(spatial_shape) == 3:
                Nx, Ny, channels = spatial_shape
            elif len(spatial_shape) == 2:
                N, channels = spatial_shape
                Nx = int(np.sqrt(N))
                if Nx * Nx != N:
                    raise ValueError(
                        "Cannot infer Nx,Ny from N; provide (Nx,Ny,channels)"
                    )
                Ny = Nx
            else:
                raise ValueError("spatial_shape must be length 2 or 3")
            if features != Nx * Ny * channels:
                raise ValueError(
                    f"Feature size {features} does not match provided spatial_shape {spatial_shape}"
                )
        return data.reshape(n_sims, n_timesteps, Nx, Ny, channels)

    # Input is full spatial 5D: (n_sims, n_timesteps, Nx, Ny, channels)
    if data.ndim == 5 and data.shape[0] == n_sims and data.shape[1] == n_timesteps:
        if target_format == "5d" or (target_format == "auto" and data.ndim == 5):
            return data
        # flatten to 3D
        flat3 = data.reshape(n_sims, n_timesteps, -1)
        if target_format == "3d" or (target_format == "auto"):
            return flat3
        # flatten to 2D
        return flat3.reshape(-1, flat3.shape[-1])

    # If none matched, raise
    raise ValueError(
        f'Data shape {getattr(data, "shape", None)} not compatible with n_sims={n_sims}, n_timesteps={n_timesteps}'
    )
