import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from vindy.utils import *
import pickle


logging.info(
    "################################   1. Loading    ################################"
)


# %% script parameters
def preprocess_data(n_samples=-1, noise_level=0.02, reduced_order=32, plots=False):

    use_pca = False
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
        nth_time_step=1,
        end_time_step=-1,
        n_int=0,
    )
    # reference coordinates
    ref_coords_path = config.beam["ref_coords"]
    # load reference coordinates csv file
    ref_coords = np.loadtxt(ref_coords_path)

    # scale params so that the coefficients grow
    params[:, 2] = 1e-1 * params[:, 2]
    params_test[:, 2] = 1e-1 * params_test[:, 2]

    # project pod coordinates back to physical space to apply noise
    # remove all zero rows from V
    zero_rows = np.all(V == 0, axis=1)
    ref_coords2 = ref_coords.reshape(-1)[~zero_rows].reshape(-1, 3)
    V = V[~zero_rows]

    # for debugging purposes only use first 10000 samples
    x = x[: 2 * n_timesteps]
    dxdt = dxdt[: 2 * n_timesteps]
    dxddt = dxddt[: 2 * n_timesteps]
    params = params[: 2 * n_timesteps]
    t = t[: 2 * n_timesteps]

    # x = x[: 2 * n_timesteps]

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
    # 20 percent of the mean displacement as noise
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
        f=f,
        f_test=f_test,
        t=t,
        t_test=t_test,
        n_sims=n_sims,
        n_timesteps=n_timesteps,
    )
    with open(os.path.join(dir, "processed_data.npy"), "wb") as out_file:
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

        # 3d plot of the beam
        # visualizer = Visualizer()
        # # disps = add_lognormal_noise(disps, 0.2)
        # visualizer.animate(
        #     20 * X_.reshape((-1, ref_coords2.shape[0], ref_coords2.shape[1])) + ref_coords2,
        #     range(10000),
        #     color="magenta",
        #     point_size=3,
        # )


def load_beam_data(
    data_paths,
    nth_time_step=1,
    end_time_step=None,
    n_int=0,
    pca_order=64,
):
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

    n_test = int(n_sims / 2)
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
            :, ::1
        ].reshape(-1, array.shape[1])

    # acknowledge for new n_timesteps
    n_timesteps = int((end_time_step + 1) // nth_time_step)

    # x_successors = []
    # if n_int > 0:
    #     x_int = np.zeros([x.shape[0], x.shape[1] - n_int, n_int, x.shape[2]])
    #     for i_sample in range(0, x.shape[1] - n_int):
    #         x_int[:, i_sample] = x[:, i_sample : i_sample + n_int]
    #     dx_int = np.zeros([x.shape[0], x.shape[1] - n_int, n_int, x.shape[2]])
    #     for i_sample in range(0, dx_dt.shape[1] - n_int):
    #         dx_int[:, i_sample] = dx_dt[:, i_sample : i_sample + n_int]
    #     param_int = np.zeros(
    #         [param.shape[0], param.shape[1] - n_int, n_int, param.shape[2]]
    #     )
    #     for i_sample in range(0, dx_dt.shape[1] - n_int):
    #         param_int[:, i_sample] = param[:, i_sample : i_sample + n_int]
    #     # remove n_int samples from all data
    #     x = x[:, :-n_int]
    #     dx_dt = dx_dt[:, :-n_int]
    #     dx_ddt = dx_ddt[:, :-n_int]
    #     time = time[:, :-n_int]
    #     param = param[:, :-n_int]
    # else:
    #     x_int = None
    #
    # if n_int > 0:
    #     x_int_train = np.concatenate(x_int, axis=0)
    #     dx_int_train = np.concatenate(x_int, axis=0)
    #     param_int_train = np.concatenate(param_int, axis=0)
    # else:
    # x_int_train, x_int_test = None, None
    # dx_int_train, dx_int_test = None, None
    # param_int_train, param_int_test = None, None

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
    preprocess_data(n_samples=-0, noise_level=0.01, reduced_order=32)
