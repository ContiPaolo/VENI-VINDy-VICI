"""
BEAM MODEL SCRIPT

This script trains and evaluates a beam model using the VENI framework.
The model identifies the dynamics of a beam system based on input data.
The data is assumed to be preprocessed and available in the specified path (as per config.py).

Model:
    dzdt    = z
    dzddt   = -w0^2 * z - 2 * xi * w0 * z_dot - gamma * z^3 + u
            = - 0.29975625 * z - 0.01095 z_dot - gamma * z^3 + u
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

from vindy import VENI
from vindy.libraries import PolynomialLibrary, ForceLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Laplace
from vindy.callbacks import SaveCoefficientsCallback
from utils import load_beam_data, plot_train_history, plot_coefficients_train_history

# Add the examples folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Constants
LOAD_MODEL = False
BETA_VINDY = 1e-8
BETA_VAE = 1e-8
L_REC = 1e-3
L_DZ = 1e0
L_DX = 1e-5
END_TIME_STEP = 14000
MODEL_NAME = "beam"
IDENTIFICATION_LAYER = "vindy"  # 'vindy' or 'sindy'
REDUCED_ORDER = 1
PCA_ORDER = 3
NTH_TIME_STEP = 6
EPOCHS = 10000
LEARNING_RATE = 2e-3
SECOND_ORDER = True
PDF_THRESHOLD = 5


def load_data():
    """
    Load and preprocess the beam data.

    Returns:
        Tuple containing training and test data, PCA components, and other
        parameters.
    """
    logging.info("Loading beam data...")
    return load_beam_data(
        config.beam["processed_data"],
        end_time_step=END_TIME_STEP,
        nth_time_step=NTH_TIME_STEP,
        pca_order=PCA_ORDER,
    )


def create_model(x, params, dt, n_dof):
    """
    Create the VENI model.

    Args:
        params (np.ndarray): Parameters for the model.
        dt (float): Time step size.
        n_dof (int): Number of degrees of freedom.

    Returns:
        VENI: The initialized VENI model.
    """
    logging.info("Creating model...")
    libraries = [PolynomialLibrary(3)]
    param_libraries = [ForceLibrary(functions=[tf.cos])]

    layer_params = dict(
        state_dim=REDUCED_ORDER,
        param_dim=params.shape[1],
        feature_libraries=libraries,
        second_order=SECOND_ORDER,
        param_feature_libraries=param_libraries,
        x_mu_interaction=False,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-8, l2=0),
        mask=None,
        fixed_coeffs=None,
    )

    if IDENTIFICATION_LAYER == "vindy":
        sindy_layer = VindyLayer(
            beta=BETA_VINDY,
            priors=Laplace(0.0, 1.0),
            **layer_params,
        )
    elif IDENTIFICATION_LAYER == "sindy":
        sindy_layer = SindyLayer(**layer_params)
    else:
        raise ValueError('IDENTIFICATION_LAYER must be either "vindy" or "sindy"')

    return VENI(
        sindy_layer=sindy_layer,
        beta=BETA_VAE * REDUCED_ORDER / n_dof,
        reduced_order=REDUCED_ORDER,
        x=x,
        mu=params,
        scaling="individual_sqrt",
        second_order=SECOND_ORDER,
        layer_sizes=[32, 32, 32],
        activation="elu",
        l_rec=L_REC,
        l_dz=L_DZ,
        l_dx=L_DX,
        dt=dt,
    )


def train_model(veni, x_input, x_input_val, weights_path, log_dir, train_histdir):
    """
    Train the VENI model.

    Args:
        veni (VENI): The VENI model.
        x_input (list): Training data.
        x_input_val (list): Validation data.
        weights_path (str): Path to save the model weights.
        log_dir (str): Directory for TensorBoard logs.
    """

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if LOAD_MODEL:
        logging.info("Loading model...")
        veni.load_weights(os.path.join(weights_path))
    else:
        logging.info("Training model...")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=weights_path,
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
                verbose=0,
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            SaveCoefficientsCallback(),
        ]

        trainhist = veni.fit(
            x=x_input,
            validation_data=(x_input_val, None),
            callbacks=callbacks,
            y=None,
            epochs=EPOCHS,
            batch_size=int(x_input[0].shape[0] / NTH_TIME_STEP),
            verbose=2,
        )
        # Save training history
        np.save(
            train_histdir,
            trainhist.history,
        )

        veni.print(precision=4)

        # save model
        veni.load_weights(weights_path)

    # load trainhist
    trainhist = np.load(
        train_histdir,
        allow_pickle=True,
    ).item()

    return trainhist


def perform_forward_uq(
    veni,
    x_test,
    dxdt_test,
    dxddt_test,
    params_test,
    t_test,
    test_ids,
    n_traj,
    n_test,
    n_timesteps_test,
    sigma=3,
):
    """
    Perform forward uncertainty quantification (UQ) by sampling SINDy coefficients
    from the posterior distribution, integrating the ODE, and collecting trajectories.

    Args:
        veni: The trained VENI model.
        X_test: Test data in the physical space.
        DXDT_test: Test data derivatives.
        PARAMS_test: Test parameters.
        t_test: Test time steps.
        test_ids: List of test trajectory indices.
        n_traj: Number of trajectories to sample for UQ.
        n_test: Number of test trajectories.
        n_timesteps_test: Number of timesteps in each test trajectory.
        sigma: Standard deviation multiplier for uncertainty bounds.

    Returns:
        dict: A dictionary containing UQ results including mean, std, and trajectories.
    """
    X_test = switch_data_format(x_test, n_test, n_timesteps_test)
    DXDT_test = switch_data_format(dxdt_test, n_test, n_timesteps_test)
    PARAMS_test = switch_data_format(params_test, n_test, n_timesteps_test)
    t_test = switch_data_format(t_test, n_test, n_timesteps_test)

    # Calculate latent time derivatives
    z_test, dzdt_test, dzddt_test = veni.calc_latent_time_derivatives(
        x_test, dxdt_test, dxddt_test
    )
    z_test = switch_data_format(z_test, n_test, n_timesteps_test)
    dzdt_test = switch_data_format(dzdt_test, n_test, n_timesteps_test)

    # Store the original coefficients
    kernel_orig, kernel_scale_orig = (
        veni.sindy_layer.kernel,
        veni.sindy_layer.kernel_scale,
    )

    uq_ts = []
    uq_ys = []
    uq_means = []
    for i_test in test_ids:
        logging.info(f"Processing trajectory {i_test+1}/{n_test}")
        z_preds = []
        t_preds = []
        z0, dzdt0 = veni.calc_latent_time_derivatives(
            X_test[i_test][0:1], DXDT_test[i_test][0:1]
        )
        for traj in range(n_traj):
            logging.info(f"\tSampling trajectory {traj+1}/{n_traj}")

            # Sample from the posterior distribution of the coefficients and integrate the model
            sol, coeffs = veni.sindy_layer.integrate_uq(
                np.concatenate([z0, dzdt0]).squeeze(),
                t_test[i_test].squeeze(),
                mu=PARAMS_test[i_test],
            )

            z_preds.append(sol.y)
            t_preds.append(sol.t)
        uq_ts.append(t_preds)
        uq_ys.append(z_preds)

        # Mean simulation
        veni.sindy_layer.kernel, veni.sindy_layer.kernel_scale = (
            kernel_orig,
            kernel_scale_orig,
        )
        z0, dzdt0 = veni.calc_latent_time_derivatives(
            X_test[i_test][0:1], DXDT_test[i_test][0:1]
        )
        sol = veni.integrate(
            np.concatenate([z0, dzdt0]).squeeze(),
            t_test[i_test].squeeze(),
            mu=PARAMS_test[i_test],
        )
        uq_means.append(sol.y)

    # Calculate mean and variance of the trajectories
    uq_ys = np.array(uq_ys)
    uq_ys_mean_sampled = np.mean(uq_ys, axis=1)
    uq_ys_std = np.std(uq_ys, axis=1)
    uq_ys_mean = np.array(uq_means)

    # Calculate sigma bounds
    uq_ys_lb = uq_ys_mean - sigma * uq_ys_std
    uq_ys_ub = uq_ys_mean + sigma * uq_ys_std

    # Return results
    return {
        "uq_ts": uq_ts,
        "uq_ys": uq_ys,
        "uq_ys_mean_sampled": uq_ys_mean_sampled,
        "uq_ys_std": uq_ys_std,
        "uq_ys_mean": uq_ys_mean,
        "uq_ys_lb": uq_ys_lb,
        "uq_ys_ub": uq_ys_ub,
        "z_test": z_test,
        "dzdt_test": dzdt_test,
    }


def switch_data_format(data, n_sims, n_timesteps):
    """
    Switch between vectorized data and simulation-wise data.

    Args:
        data (np.ndarray): The input data, either vectorized or simulation-wise.
        n_sims (int): Number of simulations.
        n_timesteps (int): Number of time steps per simulation.

    Returns:
        np.ndarray: The data in the switched format.
    """
    if data.ndim == 2 and data.shape[0] == n_sims * n_timesteps:
        # Convert from vectorized to simulation-wise
        return data.reshape(n_sims, n_timesteps, -1)
    elif data.ndim == 3 and data.shape[0] == n_sims and data.shape[1] == n_timesteps:
        # Convert from simulation-wise to vectorized
        return data.reshape(-1, data.shape[-1])
    else:
        raise ValueError("Data shape does not match the expected dimensions.")


def training_plots(trainhist, result_dir, x_train_scaled, x_test_scaled, veni):
    """
    Generate training plots and visualizations.
    Args:
        trainhist (dict): Training history.
        result_dir (str): Directory to save results.
        x_train_scaled (np.ndarray): Scaled training data.
        x_test_scaled (np.ndarray): Scaled test data.
        veni (VENI): The trained VENI model.
    """
    # Plot training history
    plot_train_history(trainhist, result_dir, validation=True)
    plot_train_history(trainhist, result_dir, validation=False)
    plot_coefficients_train_history(trainhist, result_dir)

    # reconstruction of PCA trajectories
    veni.vis_modes(x_test_scaled, 4)
    veni.vis_modes(x_train_scaled, 4)

    # visualize identified coefficients
    veni.sindy_layer.visualize_coefficients(x_range=[-1.5, 1.5])
    plt.show()


def perform_inference(
    veni,
    x_test_scaled,
    dxdt_test_scaled,
    t_test,
    params_test,
    test_ids,
    n_sims,
    n_timesteps_test,
):
    """
    Perform inference on test trajectories and plot the results.

    Args:
        veni: The trained VENI model.
        x_test_scaled: Scaled test data.
        dxdt_test_scaled: Scaled test data derivatives.
        t_test: Test time steps.
        params_test: Test parameters.
        test_ids: List of test trajectory indices.
        n_sims: Number of simulations.
        n_timesteps_test: Number of timesteps in each test trajectory.

    Returns:
        Tuple: Predicted trajectories and their corresponding time steps.
    """
    # Reshape data into simulation-wise format
    T_test = switch_data_format(t_test, n_sims, n_timesteps_test)
    z_test, dzdt_test = veni.calc_latent_time_derivatives(
        x_test_scaled, dxdt_test_scaled
    )
    z_test = switch_data_format(z_test, n_sims, n_timesteps_test)
    dzdt_test = switch_data_format(dzdt_test, n_sims, n_timesteps_test)
    Params_test = switch_data_format(params_test, n_sims, n_timesteps_test)

    z_preds = []
    t_preds = []
    for i_test in test_ids:
        logging.info(f"Processing trajectory {i_test+1}/{len(test_ids)}")
        # Perform integration
        sol = veni.integrate(
            np.concatenate([z_test[i_test, 0], dzdt_test[i_test, 0]]).squeeze(),
            T_test[i_test].squeeze(),
            mu=Params_test[i_test],
        )
        z_preds.append(sol.y)
        t_preds.append(sol.t)

    # Convert predictions to arrays
    z_preds = np.array(z_preds)
    t_preds = np.array(t_preds)

    # Plot inference results
    fig, axs = plt.subplots(len(test_ids), 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Inference of Test Trajectories")
    for i, i_test in enumerate(test_ids):
        axs[i].set_title(f"Test Trajectory {i_test + 1}")
        axs[i].plot(T_test[i_test], z_test[i_test][:, 0], color="blue", label="True")
        axs[i].plot(
            t_preds[i], z_preds[i][0], color="red", linestyle="--", label="Predicted"
        )
        axs[i].set_xlabel("$t$")
        axs[i].set_ylabel("$z$")
        axs[i].legend()
    plt.show()

    return z_preds, t_preds


def uq_plots(
    uq_ts,
    uq_ys_mean,
    uq_ys_mean_sampled,
    uq_ys_std,
    t_test,
    z_test,
    test_ids,
):
    """
    Generate UQ plots.

    Args:
        uq_ts (list): Time points for UQ trajectories.
        uq_ys_mean (list): Mean trajectories from UQ.
        uq_ys_mean_sampled (list): Mean of sampled trajectories.
        uq_ys_std (list): Standard deviation of sampled trajectories.
        t_test (np.ndarray): Test time steps.
        z_test (np.ndarray): Latent states for test data.
        test_ids (list): List of test trajectory indices to plot.
    """
    n_test = len(test_ids)
    # plot the mean and 3*std of the trajectories
    fig, axs = plt.subplots(n_test, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Integrated Test Trajectories")
    for i, i_test in enumerate(test_ids):
        axs[i].set_title(f"Test Trajectory {i_test + 1}")
        # for i in range(2):
        axs[i].plot(t_test[i_test], z_test[i_test][:, 0], color="blue")
        axs[i].plot(uq_ts[i][0], uq_ys_mean[i][0], color="red", linestyle="--")
        axs[i].fill_between(
            uq_ts[i][0],
            uq_ys_mean_sampled[i][0] - 3 * uq_ys_std[i][0],
            uq_ys_mean_sampled[i][0] + 3 * uq_ys_std[i][0],
            color="red",
            alpha=0.3,
        )
        axs[i].set_xlabel("$t$")
        axs[i].set_ylabel("$z$")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to load data, create the model, train, and evaluate.
    """
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
    ) = load_data()

    n_timesteps_test = x_test.shape[0] // n_sims
    n_dof = x.shape[1]
    dt = t[1] - t[0]

    # Create model
    veni = create_model(x, params, dt, n_dof)

    # Scale data
    veni.define_scaling(x)
    x_train_scaled, dxdt_train_scaled, dxddt_train_scaled = (
        veni.scale(x).numpy(),
        veni.scale(dxdt).numpy(),
        veni.scale(dxddt).numpy(),
    )
    x_test_scaled, dxdt_test_scaled, dxddt_test_scaled = (
        veni.scale(x_test).numpy(),
        veni.scale(dxdt_test).numpy(),
        veni.scale(dxddt_test).numpy(),
    )

    x_input = [
        x_train_scaled[: 24 * n_timesteps],
        dxdt_train_scaled[: 24 * n_timesteps],
        dxddt_train_scaled[: 24 * n_timesteps],
        params[: 24 * n_timesteps],
    ]
    x_input_val = [
        x_train_scaled[24 * n_timesteps :],
        dxdt_train_scaled[24 * n_timesteps :],
        dxddt_train_scaled[24 * n_timesteps :],
        params[24 * n_timesteps :],
    ]

    # Compile and build model
    veni.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    veni.build(input_shape=([input.shape for input in x_input], None))

    # Train model
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    log_dir = os.path.join(
        result_dir,
        f'{MODEL_NAME}/log/{MODEL_NAME}_{REDUCED_ORDER}_{datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")}',
    )
    weights_path = os.path.join(
        result_dir,
        f"{MODEL_NAME}/{MODEL_NAME}_{REDUCED_ORDER}_{veni.__class__.__name__}_{IDENTIFICATION_LAYER}.weights.h5",
    )
    train_hist_dir = os.path.join(
        result_dir, f"{MODEL_NAME}/trainhist_{IDENTIFICATION_LAYER}.npy"
    )
    trainhist = train_model(
        veni, x_input, x_input_val, weights_path, log_dir, train_hist_dir
    )

    training_plots(trainhist, result_dir, x_train_scaled, x_test_scaled, veni)

    # Sparsification of the identified model
    veni.sindy_layer.pdf_thresholding(threshold=PDF_THRESHOLD)

    # Inference and forward UQ
    logging.info("Performing inference and forward UQ...")

    # Predict latent states and uncertainty bounds
    n_traj = 10
    test_ids = [1, 10]

    # Inference
    z_preds, t_preds = perform_inference(
        veni,
        x_test_scaled,
        dxdt_test_scaled,
        t_test,
        params_test,
        test_ids,
        n_sims,
        n_timesteps_test,
    )

    # UQ
    uq_results = perform_forward_uq(
        veni,
        x_test_scaled,
        dxdt_test_scaled,
        dxddt_test_scaled,
        params_test,
        t_test,
        test_ids,
        n_traj,
        n_sims,
        n_timesteps_test,
        REDUCED_ORDER,
    )

    # Plot results
    uq_plots(
        uq_results["uq_ts"],
        uq_results["uq_ys_mean"],
        uq_results["uq_ys_mean_sampled"],
        uq_results["uq_ys_std"],
        switch_data_format(t_test, n_sims, n_timesteps_test),
        uq_results["z_test"],
        test_ids,
    )


if __name__ == "__main__":
    main()
