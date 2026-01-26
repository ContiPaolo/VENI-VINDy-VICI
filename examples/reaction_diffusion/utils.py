import logging
import sys
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

    # reduce dimensionality with POD
    # if pod:
    #     x_pod_u_train = np.reshape(
    #         data["U"][:, :, :n_timesteps_train, :], (N, n_timesteps_train * n_sims), "F"
    #     )
    #     x_pod_v_train = np.reshape(
    #         data["V"][:, :, :n_timesteps_train, :], (N, n_timesteps_train * n_sims), "F"
    #     )
    #     x_pod_train = np.concatenate((x_pod_u_train, x_pod_v_train))
    #     # apply noise to the high-dimensional training data
    #     if noise:
    #         mu = 0
    #         sigma = 0.2
    #         scale_noise = np.exp(mu)
    #         np.random.seed(seed)
    #         x_pod_train *= (
    #             np.random.lognormal(mean=0, sigma=sigma, size=x_pod_train.shape)
    #             * scale_noise
    #         )
    #
    #     U, S, _ = compute_randomized_SVD(x_pod_train, pca_order, N * 2, 1)
    #     x_pod_u = np.reshape(data["U"], (N, n_timesteps * n_sims), "F")
    #     x_pod_v = np.reshape(data["V"], (N, n_timesteps * n_sims), "F")
    #     x_pod = np.concatenate((x_pod_u, x_pod_v))
    #     x = x_pod.T @ U
    #     x = np.reshape(x, (n_sims, n_timesteps, pca_order))
    # else:
    #     # ensure x has shape (n_sims, n_timesteps, Nx, Ny, channels)
    #     x = data["U"].transpose(3, 2, 0, 1).reshape(n_sims, n_timesteps, N)
    #     U = None

    # build full state with both channels in the last dimension: (n_sims, n_timesteps, Nx, Ny, 2)
    x = np.concatenate(
        (
            data["U"].transpose(3, 2, 0, 1)[..., np.newaxis],
            data["V"].transpose(3, 2, 0, 1)[..., np.newaxis],
        ),
        axis=-1,
    )
    # reduce time steps for faster training to 20 seconds for training and 40 seconds for testing
    if short:
        n_timesteps = 800
        n_timesteps_train = int(n_timesteps / 2)
    x = x[:, :n_timesteps:nth_time_step, :, :]
    times = times[:n_timesteps:nth_time_step]
    n_timesteps = x.shape[1]
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=seed)
    dxdt_test = np.array(
        [
            np.gradient(x_test[i, :, :, :], dt, axis=0, edge_order=2)
            for i in range(x_test.shape[0])
        ]
    )

    # perform SVD
    # x_train has shape (n_sims, n_timesteps, Nx, Ny, channels)
    x_train = x_train[:, :n_timesteps_train]
    n_sims_train = x_train.shape[0]
    channels = x_train.shape[-1]

    # apply noise
    mu = 0
    sigma = 0.2
    scale_noise = np.exp(mu)
    x_train = x_train * (
        np.random.lognormal(mean=0, sigma=sigma, size=x_train.shape) * scale_noise
    )

    # compute time derivatives of noisy data
    # Use a vectorized gradient call along the time axis (axis=1 for shape (n_sims, n_timesteps, ...)).
    # This ensures the same dt is used for all simulations and avoids scaling mismatches
    # versus calling np.gradient on a single 1D trace without specifying dt.
    # Example: for a single trajectory `x0 = x_train[0, :, 0, 0, 0]` compute `np.gradient(x0, dt, edge_order=2)`.
    dxdt = np.gradient(x_train, dt, axis=1, edge_order=2)

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
        spatial_shape=(Nx_hf, Ny_hf, channels),
        target_format="5d",
    )

    dxdt_2d = switch_data_format(
        dxdt, n_sims_train, n_timesteps_train, target_format="2d"
    )
    dxdt_rec_2d = pca.inverse_transform(pca.transform(dxdt_2d))
    dxdt_rec = switch_data_format(
        dxdt_rec_2d,
        n_sims_train,
        n_timesteps_train,
        spatial_shape=(Nx_hf, Ny_hf, channels),
        target_format="5d",
    )
    plt.title("Noisy sample")
    plt.imshow(dxdt[0, 0, :, :, 0])
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

    # project onto POD basis: x_train_2d (samples, features) @ U2 (features, pca)
    x_train_pod = (x_train_2d @ U2).reshape(n_sims_train, n_timesteps_train, -1)

    # reconstruct from POD: (samples, pca) @ U2.T -> (samples, features) then reshape back to spatial shape
    x_rec_from_pod = (x_train_pod.reshape(-1, x_train_pod.shape[-1]) @ U2.T).reshape(
        n_sims_train, n_timesteps_train, Nx_hf, Ny_hf, channels
    )
    plt.imshow(x_rec_from_pod[0, 0, :, :, 1])
    plt.show()

    times_train = np.tile(times, x.shape[0]).reshape(-1, 1)
    times_test = np.tile(times, x_test.shape[0]).reshape(-1, 1)

    return (
        times_train,
        x,
        dxdt,
        times_test,
        x_test,
        dxdt_test,
        V,
        n_sims,
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
