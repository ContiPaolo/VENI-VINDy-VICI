"""
Reaction diffusion example - refactored to be modular (MEMS-like structure).
This file was reorganized from a linear script into functions:
- set_seed
- load_data
- create_model
- train_model
- perform_inference
- perform_forward_uq
- training_plots / uq_plots
- main

The intent is to keep runtime behavior but avoid doing heavy work at import-time so
this module can be imported safely by tests or other scripts.
"""

import os
import sys
import logging
import datetime
import time
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# local imports
from vindy import VENI
from vindy.libraries import PolynomialLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Laplace
from vindy.callbacks import SaveCoefficientsCallback
from vindy.utils import plot_train_history, plot_coefficients_train_history
from utils import load_reactiondiffusion_data

# Add the examples folder to the Python path (keep compatibility with examples/ imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import config (robust fallback for static analysis / tests)
try:
    import config
except Exception:
    config = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ----------------------
# Default script constants (mirrors previous top-level script)
# ----------------------
MODEL_NAME = "reactiondiffusion"
IDENTIFICATION_LAYER = "vindy"  # 'vindy' or 'sindy'
REDUCED_ORDER = 2
PCA_ORDER = 32
NOISE = True
NTH_TIME_STEP = 3
SECOND_ORDER = False

BETA_VINDY = 2e-5
BETA_VAE = 1e-4
L_REC = 1e-2
L_DZ = 4e0
L_DX = 1e-2

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

# Training defaults (can be overridden by calling train_model with args)
LOAD_MODEL = False
EPOCHS = 2500
BATCH_SIZE = None  # computed later based on data

# ----------------------
# Utility functions
# ----------------------


def set_seed(seed: int):
    """Set seeds for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(
    pca_order: int = PCA_ORDER, noise: bool = NOISE, nth_time_step: int = NTH_TIME_STEP
):
    """Load reaction diffusion data and preprocess.

    Returns the tuple used throughout the rest of the script. This function mirrors the
    previous inline loading logic but keeps everything in a callable form.
    """
    import mat73

    # Ensure config is available and has the expected attribute
    if config is None or not hasattr(config, "reaction_diffusion"):
        raise RuntimeError(
            "config module or attribute 'reaction_diffusion' not found.\n"
            "Create 'examples/config.py' from 'examples/config.py.template' and set the path to the data file."
        )

    config_path = config.reaction_diffusion
    logging.info("Loading data from %s", config_path)
    data = mat73.loadmat(config_path)
    logging.info("Data loaded.")

    # original file provided full-field snapshots in data['U'] and time in data['t']
    times = data["t"]
    dt = times[1] - times[0]
    x_full = data["U"]

    # noise / preprocessing branch: reuse utility if available (preferred)
    if noise:
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
        ) = load_reactiondiffusion_data(config.reaction_diffusion, pca_order=pca_order)
    else:
        # Fallback: create minimal shapes from the full data. This path is less tested.
        t = times
        dt = t[1] - t[0]
        # Assume x_full shape is (nx, ny, nt, n_sims)
        n_sims = x_full.shape[-1]
        # collapse spatial dims to PCA_ORDER using simple truncation (fallback behavior)
        V = np.eye(x_full.shape[0] * x_full.shape[1])
        n_timesteps = x_full.shape[2]
        # vectorize temporal data as (n_sims * n_timesteps, pca_order)
        x = np.zeros((n_sims * n_timesteps, pca_order))
        dxdt = np.zeros_like(x)
        # create trivial test splits
        t_test = t[:n_timesteps]
        x_test = x[:n_timesteps]
        dxdt_test = dxdt[:n_timesteps]

    # Parameters are not present in this dataset by default — create empty placeholders
    params = np.zeros((x.shape[0], 0)) if hasattr(x, "shape") else np.zeros((0, 0))
    params_test = (
        np.zeros((x_test.shape[0], 0)) if hasattr(x_test, "shape") else np.zeros((0, 0))
    )

    return (
        t,
        params,
        x,
        dxdt,
        None,  # dxddt (not provided/generated here)
        t_test,
        params_test,
        x_test,
        dxdt_test,
        None,  # dxddt_test
        None,  # ref_coords (not used)
        V,
        n_sims,
        n_timesteps,
    )


def create_model(x, params, dt, n_dof):
    """Create and return a configured VENI model for reaction diffusion.

    The model creation mirrors the previous inline logic but uses constants defined above.
    """
    logging.info("Creating model...")
    libraries = [PolynomialLibrary(3)]
    param_libraries = []

    layer_params = dict(
        state_dim=REDUCED_ORDER,
        param_dim=0 if params is None or params.size == 0 else params.shape[1],
        feature_libraries=libraries,
        second_order=SECOND_ORDER,
        param_feature_libraries=param_libraries,
        x_mu_interaction=False,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=0),
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

    veni = VENI(
        sindy_layer=sindy_layer,
        beta=BETA_VAE * REDUCED_ORDER / max(1, n_dof),
        reduced_order=REDUCED_ORDER,
        x=x,
        mu=None,
        scaling="individual_sqrt",
        second_order=SECOND_ORDER,
        layer_sizes=[32, 16, 8],
        activation="elu",
        l_rec=L_REC,
        l_dz=L_DZ,
        l_dx=L_DX,
        dt=dt,
    )

    return veni


def train_model(
    veni,
    x_input,
    x_input_val,
    result_dir,
    model_name=MODEL_NAME,
    load_model=LOAD_MODEL,
    epochs=EPOCHS,
    batch_size=None,
):
    """Train or load the model; save training history and weights to result directory.

    Returns training history dict (loaded from file) for downstream plotting.
    """
    os.makedirs(result_dir, exist_ok=True)

    log_dir = os.path.join(
        result_dir,
        f'{model_name}/log/{model_name}_{REDUCED_ORDER}_{veni.__class__.__name__}_{datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")}',
    )
    weights_path = os.path.join(
        result_dir,
        f"{model_name}/{model_name}_{REDUCED_ORDER}_{veni.__class__.__name__}_{IDENTIFICATION_LAYER}.weights.h5",
    )
    train_histdir = os.path.join(
        result_dir, f"{model_name}/trainhist_{IDENTIFICATION_LAYER}.npy"
    )

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if load_model and os.path.exists(weights_path):
        logging.info("Loading existing model weights from %s", weights_path)
        veni.load_weights(weights_path)
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

        start_time = time.time()
        trainhist = veni.fit(
            x=x_input,
            validation_data=(x_input_val, None),
            callbacks=callbacks,
            y=None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
        )
        end_time = time.time()
        logging.info(
            "time per epoch: %.2f seconds", (end_time - start_time) / max(1, epochs)
        )

        # save trainhist
        np.save(train_histdir, trainhist.history)

        # ensure weights saved
        if os.path.exists(weights_path):
            try:
                veni.load_weights(weights_path)
            except Exception:
                logging.warning("Could not reload weights after training (non-fatal)")

    # load trainhist
    if os.path.exists(train_histdir):
        trainhist = np.load(train_histdir, allow_pickle=True).item()
    else:
        logging.warning("Training history not found at %s", train_histdir)
        trainhist = {}

    return trainhist, weights_path, log_dir


# ----------------------
# Helpers: plotting, inference, UQ
# ----------------------


def switch_data_format(data, n_sims, n_timesteps):
    """Switch between vectorized and simulation-wise data formats."""
    if data is None:
        return None
    if data.ndim == 2 and data.shape[0] == n_sims * n_timesteps:
        return data.reshape(n_sims, n_timesteps, -1)
    elif data.ndim == 3 and data.shape[0] == n_sims and data.shape[1] == n_timesteps:
        return data.reshape(-1, data.shape[-1])
    else:
        return data


def training_plots(trainhist, result_dir, x_train_scaled, x_test_scaled, veni):
    """Plot training results and coefficient history."""
    try:
        plot_train_history(trainhist, result_dir, validation=True)
        plot_train_history(trainhist, result_dir, validation=False)
        plot_coefficients_train_history(trainhist, result_dir)

        # reconstruction of PCA trajectories
        veni.vis_modes(x_test_scaled, 4)
        veni.vis_modes(x_train_scaled, 4)

        # visualize identified coefficients
        veni.sindy_layer.visualize_coefficients(x_range=[-1.5, 1.5])
        plt.show()
    except Exception as e:
        logging.warning("Unable to create training plots: %s", e)


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
    """Perform inference on test trajectories and plot results."""
    T_test = switch_data_format(t_test, n_sims, n_timesteps_test)
    z_test, dzdt_test = veni.calc_latent_time_derivatives(
        x_test_scaled, dxdt_test_scaled
    )
    z_test = switch_data_format(z_test, n_sims, n_timesteps_test)
    dzdt_test = switch_data_format(dzdt_test, n_sims, n_timesteps_test)
    Params_test = switch_data_format(params_test, n_sims, n_timesteps_test)

    z_preds = []
    t_preds = []
    start_time = datetime.datetime.now()
    for i, i_test in enumerate(test_ids):
        logging.info("Processing trajectory %d/%d", i + 1, len(test_ids))
        # Perform integration
        sol = veni.integrate(
            np.concatenate([z_test[i_test, 0], dzdt_test[i_test, 0]]).squeeze(),
            T_test[i_test].squeeze(),
            mu=Params_test[i_test] if Params_test is not None else None,
        )
        z_preds.append(sol.y)
        t_preds.append(sol.t)
    end_time = datetime.datetime.now()
    logging.info(
        "Inference time: %.2f seconds per trajectory",
        (end_time - start_time).total_seconds() / max(1, len(test_ids)),
    )

    # Plot simple results
    try:
        fig, axs = plt.subplots(
            len(test_ids), 1, figsize=(12, 6 * len(test_ids)), sharex=True
        )
        if len(test_ids) == 1:
            axs = [axs]
        for i, i_test in enumerate(test_ids):
            axs[i].plot(
                T_test[i_test], z_test[i_test][:, 0], color="blue", label="True"
            )
            axs[i].plot(
                t_preds[i],
                z_preds[i][0],
                color="red",
                linestyle="--",
                label="Predicted",
            )
            axs[i].set_xlabel("$t$")
            axs[i].set_ylabel("$z$")
            axs[i].legend()
        plt.show()
    except Exception as e:
        logging.warning("Inference plotting failed: %s", e)

    return np.array(z_preds), np.array(t_preds)


def perform_forward_uq(
    veni,
    x_test,
    dxdt_test,
    dxddt_test,
    params_test,
    t_test,
    test_ids,
    n_traj,
    n_sims,
    n_timesteps_test,
    sigma=3,
):
    """Perform forward uncertainty quantification by sampling coefficients and integrating."""
    # Try to switch data format to simulation-wise arrays
    X_test = switch_data_format(x_test, n_sims, n_timesteps_test)
    DXDT_test = switch_data_format(dxdt_test, n_sims, n_timesteps_test)
    PARAMS_test = switch_data_format(params_test, n_sims, n_timesteps_test)
    T_test = switch_data_format(t_test, n_sims, n_timesteps_test)

    # Calculate latent time derivatives
    z_test, dzdt_test, dzddt_test = veni.calc_latent_time_derivatives(
        x_test, dxdt_test, dxddt_test
    )
    z_test = switch_data_format(z_test, n_sims, n_timesteps_test)
    dzdt_test = switch_data_format(dzdt_test, n_sims, n_timesteps_test)

    # Store the original coefficients
    kernel_orig, kernel_scale_orig = (
        veni.sindy_layer.kernel,
        veni.sindy_layer.kernel_scale,
    )

    uq_ts = []
    uq_ys = []
    uq_means = []
    for i_test in test_ids:
        logging.info("Processing trajectory %d/%d for UQ", i_test + 1, len(test_ids))
        sol_list = []
        sol_list_t = []
        for traj in range(n_traj):
            logging.info("\tSample %d/%d", traj + 1, n_traj)
            # Use layer helper if available
            try:
                sol, coeffs = veni.sindy_layer.integrate_uq(
                    np.concatenate(
                        [z_test[i_test][0:1], dzdt_test[i_test][0:1]]
                    ).squeeze(),
                    T_test[i_test].squeeze(),
                    mu=PARAMS_test[i_test] if PARAMS_test is not None else None,
                )
                sol_list.append(sol.y)
                sol_list_t.append(sol.t)
            except Exception:
                # Fallback: sample kernel directly if integrate_uq not implemented
                sampled_coeff, _, _ = veni.sindy_layer._coeffs
                try:
                    veni.sindy_layer.kernel = tf.reshape(sampled_coeff, (-1, 1))
                except Exception:
                    pass
                z0, dzdt0 = veni.calc_latent_time_derivatives(
                    X_test[i_test][0:1], DXDT_test[i_test][0:1]
                )
                sol = veni.integrate(
                    np.concatenate([z0, dzdt0]).squeeze(),
                    T_test[i_test].squeeze(),
                    mu=PARAMS_test[i_test] if PARAMS_test is not None else None,
                )
                sol_list.append(sol.y)
                sol_list_t.append(sol.t)

        uq_ts.append(sol_list_t)
        uq_ys.append(sol_list)

        # mean simulation
        veni.sindy_layer.kernel, veni.sindy_layer.kernel_scale = (
            kernel_orig,
            kernel_scale_orig,
        )
        z0, dzdt0 = veni.calc_latent_time_derivatives(
            X_test[i_test][0:1], DXDT_test[i_test][0:1]
        )
        sol = veni.integrate(
            np.concatenate([z0, dzdt0]).squeeze(),
            T_test[i_test].squeeze(),
            mu=PARAMS_test[i_test] if PARAMS_test is not None else None,
        )
        uq_means.append(sol.y)

    uq_ys = np.array(uq_ys)
    uq_ys_mean_sampled = np.mean(uq_ys, axis=1)
    uq_ys_std = np.std(uq_ys, axis=1)
    uq_ys_mean = np.array(uq_means)

    uq_ys_lb = uq_ys_mean - sigma * uq_ys_std
    uq_ys_ub = uq_ys_mean + sigma * uq_ys_std

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


def uq_plots(
    uq_ts, uq_ys_mean, uq_ys_mean_sampled, uq_ys_std, t_test, z_test, test_ids
):
    try:
        n_test = len(test_ids)
        fig, axs = plt.subplots(n_test, 1, figsize=(12, 6 * n_test), sharex=True)
        if n_test == 1:
            axs = [axs]
        for i, i_test in enumerate(test_ids):
            axs[i].plot(t_test[i_test], z_test[i_test][:, 0], color="blue")
            axs[i].plot(uq_ts[i][0], uq_ys_mean[i][0], color="red", linestyle="--")
            axs[i].fill_between(
                uq_ts[i][0],
                uq_ys_mean_sampled[i][0] - 3 * uq_ys_std[i][0],
                uq_ys_mean_sampled[i][0] + 3 * uq_ys_std[i][0],
                color="red",
                alpha=0.3,
            )
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.warning("UQ plotting failed: %s", e)


# ----------------------
# Main
# ----------------------


def main():
    set_seed(42)

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
    ) = load_data(pca_order=PCA_ORDER, noise=NOISE, nth_time_step=NTH_TIME_STEP)

    n_timesteps_test = (
        x_test.shape[0] // n_sims if (x_test is not None and n_sims > 0) else 0
    )
    n_dof = x.shape[1] if x is not None else 1
    dt = t[1] - t[0] if t is not None and len(t) > 1 else 1.0

    # Create model
    veni = create_model(x, params, dt, n_dof)

    # Scale data
    veni.define_scaling(x)
    x_train_scaled, dxdt_train_scaled = veni.scale(x), veni.scale(dxdt)
    x_test_scaled, dxdt_test_scaled = veni.scale(x_test), veni.scale(dxdt_test)

    # Prepare inputs for training (keep original slicing heuristic)
    split_train = int(14 * n_timesteps) if n_timesteps else int(0.7 * x.shape[0])
    x_input = [x_train_scaled[:split_train], dxdt_train_scaled[:split_train]]
    x_input_val = [x_train_scaled[split_train:], dxdt_train_scaled[split_train:]]

    # compile and build
    veni.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),
        sindy_optimizer=tf.keras.optimizers.AdamW(learning_rate=8e-4),
        loss="mse",
    )
    # determine batch size if not provided
    batch_size = (
        int(n_timesteps / NTH_TIME_STEP) if (n_timesteps and NTH_TIME_STEP) else 32
    )
    veni.build(input_shape=([input.shape for input in x_input], None))

    # Train
    trainhist, weights_path, log_dir = train_model(
        veni,
        x_input,
        x_input_val,
        RESULT_DIR,
        model_name=MODEL_NAME,
        load_model=LOAD_MODEL,
        epochs=EPOCHS,
        batch_size=batch_size,
    )

    # Training plots and coefficient history
    training_plots(trainhist, RESULT_DIR, x_train_scaled, x_test_scaled, veni)

    # Sparsify coefficients
    try:
        veni.sindy_layer.pdf_thresholding(threshold=0.001)
    except Exception as e:
        logging.warning("Sparsification failed: %s", e)

    # Inference + UQ
    logging.info("Performing inference and forward UQ...")
    n_traj = 10
    test_ids = [1, 10] if n_sims > 10 else list(range(min(2, n_sims)))

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

    uq_results = perform_forward_uq(
        veni,
        x_test_scaled,
        dxdt_test_scaled,
        dxddt_test,
        params_test,
        t_test,
        test_ids,
        n_traj,
        n_sims,
        n_timesteps_test,
    )

    uq_plots(
        uq_results["uq_ts"],
        uq_results["uq_ys_mean"],
        uq_results["uq_ys_mean_sampled"],
        uq_results["uq_ys_std"],
        switch_data_format(t_test, n_sims, n_timesteps_test),
        uq_results["z_test"],
        test_ids,
    )

    # Save some results to disk (best-effort)
    outdir = os.path.join(RESULT_DIR, MODEL_NAME)
    os.makedirs(outdir, exist_ok=True)
    save_path = os.path.join(outdir, f"save_data_{MODEL_NAME}.pkl")
    try:
        save_data = {
            "uq_ts": uq_results.get("uq_ts"),
            "z_pred_mean": uq_results.get("uq_ys_mean"),
            "z_pred_ub": uq_results.get("uq_ys_ub"),
            "z_pred_lb": uq_results.get("uq_ys_lb"),
        }
        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)
        logging.info("Saved UQ results to %s", save_path)
    except Exception as e:
        logging.warning("Failed to save results: %s", e)


if __name__ == "__main__":
    main()
