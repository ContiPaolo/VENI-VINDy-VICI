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


logging.info(
    "################################   1. Loading    ################################"
)


# %% script parameters
def preprocess_data(noise_level=0.02, reduced_order=32, plots=False):

    # Load data
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
        config.beam["data"],
        nth_time_step=1,
        pca_order=4,
    )
    """
    Preprocess the MEMS data by adding noise, performing PCA, and saving the processed data.

    Args:
        n_samples (int): Number of samples to preprocess. If -1, preprocess all samples.
        noise_level (float): Noise level to add to the data (as a percentage of mean displacement).
        reduced_order (int): Number of principal components to retain during PCA.
        plots (bool): Whether to generate and display plots for debugging.

    Returns:
        None
    """

    # scale params so that the coefficients grow
    params[:, 2] = 1e-1 * params[:, 2]
    params_test[:, 2] = 1e-1 * params_test[:, 2]

    # project pod coordinates back to physical space to apply noise
    # remove all zero rows from V
    zero_rows = np.all(V == 0, axis=1)
    ref_coords2 = ref_coords.reshape(-1)[~zero_rows].reshape(-1, 3)
    V = V[~zero_rows]

    # # for debugging purposes only use first two simulations
    # x = x[: 2 * n_timesteps]
    # dxdt = dxdt[: 2 * n_timesteps]
    # dxddt = dxddt[: 2 * n_timesteps]
    # params = params[: 2 * n_timesteps]
    # t = t[: 2 * n_timesteps]

    n_train = int(x.shape[0] / n_timesteps)
    shape = (n_train, n_timesteps, ref_coords2.shape[0], ref_coords2.shape[1])

    X = (x @ V.T).reshape(shape)
    DXDT = (dxdt @ V.T).reshape(shape)
    DXDDT = (dxddt @ V.T).reshape(shape)
    X_test = x_test @ V.T
    DXDT_test = dxdt_test @ V.T
    DXDDT_test = dxddt_test @ V.T

    # calculate absolute displacement of each node
    disp_glob = np.linalg.norm(X, axis=3)
    # noise_level percent of the mean displacement as noise
    noise = np.random.normal(loc=0, scale=noise_level * disp_glob.mean(), size=X.shape)
    # add noise to the state
    X_ = X + noise
    DXDT_ = np.gradient(X_, (t[1] - t[0])[0], axis=1)
    DXDDT_ = np.gradient(DXDT_, (t[1] - t[0])[0], axis=1)

    # vectorize data again
    X_ = X_.reshape(n_train * n_timesteps, -1)
    DXDT_ = DXDT_.reshape(n_train * n_timesteps, -1)
    DXDDT_ = DXDDT_.reshape(n_train * n_timesteps, -1)

    # perform pca on the noisy data
    logging.info("Performing PCA on the noisy data")
    pca = PCA(n_components=reduced_order)
    pca.fit(X_[::4])
    x_pca = pca.transform(X_)
    dxdt_pca = pca.transform(DXDT_)
    dxddt_pca = pca.transform(DXDDT_)

    x_test_pca = pca.transform(X_test)
    dxdt_test_pca = pca.transform(DXDT_test)
    dxddt_test_pca = pca.transform(DXDDT_test)

    V = pca.components_.T

    # save data
    logging.info("Saving processed data")
    dir = os.path.split(config.beam_data[0])[0]
    save_data = dict(
        x=x_pca,
        dxdt=dxdt_pca,
        dxddt=dxddt_pca,
        x_test=x_test_pca,
        dxdt_test=dxdt_test_pca,
        dxddt_test=dxddt_test_pca,
        V=V,
        ref_coords=ref_coords2,
        noise=noise,
        params=params,
        params_test=params_test,
        t=t,
        t_test=t_test,
        n_sims=n_sims,
        n_timesteps=n_timesteps,
    )
    with open(os.path.join(dir, "data.npy"), "wb") as out_file:
        pickle.dump(save_data, out_file, pickle.HIGHEST_PROTOCOL)

    if plots:
        X_pca = pca.inverse_transform(pca.transform(X_))
        DXDT_pca = pca.inverse_transform(pca.transform(DXDT_))
        DXDDT_pca = pca.inverse_transform(pca.transform(DXDDT_))

        X_pca_test = pca.inverse_transform(pca.transform(X_test))

        # relative error norm
        rel_error = np.linalg.norm(X - X_pca) / np.linalg.norm(X)
        rel_error_n = np.linalg.norm(X_ - X_pca) / np.linalg.norm(X_)
        rel_error_test = np.linalg.norm(X_test - X_pca_test) / np.linalg.norm(
            X_pca_test
        )

        plt.plot(X[:, 100])
        plt.plot(X_[:, 100])
        plt.plot(X_pca[:, 100])
        plt.legend(["original", "noisy", "pca"])

        dof = 502
        fig, ax = plt.subplots(3, 3)
        ax[0, 0].plot(X[:1000, dof])
        ax[1, 0].plot(DXDT[:1000, dof])
        ax[2, 0].plot(DXDDT[:1000, dof])
        ax[0, 1].plot(X_[:1000, dof])
        ax[1, 1].plot(DXDT_[:1000, dof])
        ax[2, 1].plot(DXDDT_[:1000, dof])
        ax[0, 2].plot(X_pca[:1000, dof])
        ax[1, 2].plot(DXDT_pca[:1000, dof])
        ax[2, 2].plot(DXDDT_pca[:1000, dof])

        plt.semilogy(pca.singular_values_)


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
    short = True,
    pod = True,
    seed = 123,
    preprocess = False,
    noise = True
):

    print("Loading data from ", data_paths)
    data = mat73.loadmat(data_paths)     
    print("Data loaded.")
    #data = np.load(data_paths, allow_pickle=True)

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

    x_train, x_test, dxdt_train, dxdt_test, dxddt_train, dxddt_test, ind_train, ind_test = (
        train_test_split(x, dxdt, dxddt, indexes, test_size=0.2, random_state=seed)
    )

    times_train = np.tile(times, x_train.shape[0]).reshape(-1, 1)
    times_test = np.tile(times, x_test.shape[0]).reshape(-1, 1)

    def reshape_ae(data, reduce=False, end_time_step=end_time_step, nth_time_step=nth_time_step, n_timesteps=n_timesteps):
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
        U_sigma, S_sigma, V_sigma = compute_randomized_SVD(Sigma, pca_order, pca_order, 1)
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
        n_timesteps
    )

def load_beam_data(
    data_paths,
    nth_time_step=1,
    end_time_step=None,
    pca_order=64,
):
    """
    Load and preprocess the MEMS data from the specified file.

    Args:
        data_paths (str): Path to the data file.
        nth_time_step (int): Step size for time subsampling.
        end_time_step (int, optional): Last time step to include. If None, include all.
        pca_order (int): Number of principal components to retain.

    Returns:
        tuple: Processed data including time, parameters, states, derivatives, and PCA components.
    """

    # check if data path is valid
    # todo: put correct Zenodo link here
    if not os.path.isfile(data_paths):
        raise FileNotFoundError(
            f"Data file {data_paths} not found. "
            f"Please download the file from Zenodo (http://doi.org/10.5281/zenodo.18313843) and "
            f"specify the correct path in the examples/config.py file."
        )

    with open(data_paths, "rb") as f:
        data = pickle.load(f)
        V = data["V"]
        time = data["t"]
        time_test = data["t_test"]
        param = data["params"]
        params_test = data["params_test"]
        ref_coords = data["ref_coords"]

        x = data["x"][:, :pca_order]
        dx_dt = data["dxdt"][:, :pca_order]
        dx_ddt = data["dxddt"][:, :pca_order]
        x_test = data["x_test"][:, :pca_order]
        dx_dt_test = data["dxdt_test"][:, :pca_order]
        dx_ddt_test = data["dxddt_test"][:, :pca_order]

        n_timesteps = data["n_timesteps"]
        n_sims = data["n_sims"]

    if end_time_step is None:
        end_time_step = n_timesteps

    n_test = int(x_test.shape[0] / n_timesteps)
    n_sims = n_sims - n_test
    manipulate_dicts = dict(
        time=time,
        param=param,
        x=x,
        dx_dt=dx_dt,
        dx_ddt=dx_ddt,
    )

    for key, array in manipulate_dicts.items():
        logging.info(f"Manipulating {key}")
        manipulate_dicts[key] = array.reshape(n_sims, n_timesteps, array.shape[1])[
            :, :end_time_step:nth_time_step
        ].reshape(-1, array.shape[1])

    manipulate_dicts_test = dict(
        time=time_test,
        param=params_test,
        x=x_test,
        dx_dt=dx_dt_test,
        dx_ddt=dx_ddt_test,
    )
    for key, array in manipulate_dicts_test.items():
        logging.info(f"Manipulating {key}")
        # we don't want to skip time steps for the identification_layer data, and we don't want to trim time
        manipulate_dicts_test[key] = array.reshape(n_test, n_timesteps, array.shape[1])[
            :, ::nth_time_step
        ].reshape(-1, array.shape[1])

    # acknowledge for new n_timesteps
    n_timesteps = int((end_time_step + 1) // nth_time_step)

    return (
        manipulate_dicts["time"],
        manipulate_dicts["param"],
        manipulate_dicts["x"],
        manipulate_dicts["dx_dt"],
        manipulate_dicts["dx_ddt"],
        manipulate_dicts_test["time"],
        manipulate_dicts_test["param"],
        manipulate_dicts_test["x"],
        manipulate_dicts_test["dx_dt"],
        manipulate_dicts_test["dx_ddt"],
        ref_coords,
        V,
        n_sims,
        n_timesteps,
    )


if __name__ == "__main__":
    preprocess_data(noise_level=0.01, reduced_order=32)
