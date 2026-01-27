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
import mat73

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
from utils import load_reaction_diffusion_data, switch_data_format

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

BETA_VINDY = 1e-4
BETA_VAE = 2e-5
L_REC = 1e-2
L_DZ = 4e0
L_DX = 1e-2

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

# Training defaults (can be overridden by calling train_model with args)
LOAD_MODEL = True  # False
EPOCHS = 5000
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

    # Ensure config is available and has the expected attribute
    if config is None or not hasattr(config, "reaction_diffusion"):
        raise RuntimeError(
            "config module or attribute 'reaction_diffusion' not found.\n"
            "Create 'examples/config.py' from 'examples/config.py.template' and set the path to the data file."
        )

    config_path = config.reaction_diffusion
    logging.info("Loading data from %s", config_path)

    return load_reaction_diffusion_data(config_path, pca_order=pca_order)


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

    sindy_layer = VindyLayer(
        beta=BETA_VINDY,
        priors=Laplace(0.0, 1.0),
        **layer_params,
    )

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


# def switch_data_format(data, n_sims, n_timesteps):
#     """Switch between vectorized and simulation-wise data formats."""
#     if data is None:
#         return None
#     if data.ndim == 2 and data.shape[0] == n_sims * n_timesteps:
#         return data.reshape(n_sims, n_timesteps, -1)
#     elif data.ndim == 3 and data.shape[0] == n_sims and data.shape[1] == n_timesteps:
#         return data.reshape(-1, data.shape[-1])
#     else:
#         return data


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
    test_ids,
    n_sims,
    n_timesteps,
):
    """Perform inference on test trajectories and plot results."""
    T_test = switch_data_format(t_test, n_sims, n_timesteps, target_format="3d")
    z_test, dzdt_test = veni.calc_latent_time_derivatives(
        x_test_scaled, dxdt_test_scaled
    )
    z_test = switch_data_format(z_test, n_sims, n_timesteps, target_format="3d")

    z_preds = []
    t_preds = []
    start_time = datetime.datetime.now()
    for i, i_test in enumerate(test_ids):
        logging.info("Processing trajectory %d/%d", i + 1, len(test_ids))
        # Perform integration
        sol = veni.integrate(
            z_test[i_test, 0],
            T_test[i_test].squeeze(),
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
    t_test,
    test_ids,
    n_traj,
    n_sims,
    n_timesteps,
    sigma=3,
):
    """Perform forward uncertainty quantification by sampling coefficients and integrating."""
    # Try to switch data format to simulation-wise arrays
    X_test = switch_data_format(x_test.numpy(), n_sims, n_timesteps, target_format="3d")
    DXDT_test = switch_data_format(
        dxdt_test.numpy(), n_sims, n_timesteps, target_format="3d"
    )
    T_test = switch_data_format(t_test, n_sims, n_timesteps, target_format="3d")

    # Calculate latent time derivatives
    z_test, dzdt_test = veni.calc_latent_time_derivatives(x_test, dxdt_test)
    z_test = switch_data_format(z_test, n_sims, n_timesteps, target_format="3d")
    dzdt_test = switch_data_format(dzdt_test, n_sims, n_timesteps, target_format="3d")

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
            sol, coeffs = veni.sindy_layer.integrate_uq(
                z_test[i_test][0], T_test[i_test].squeeze()
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
            z0.squeeze(),
            T_test[i_test].squeeze(),
        )
        uq_means.append(sol.y)

    uq_ys = np.array(uq_ys)
    # uq_ys currently has shape (n_tests, n_traj, n_states, n_timesteps)
    # transpose to (n_tests, n_traj, n_timesteps, n_states)
    try:
        uq_ys = np.transpose(uq_ys, (0, 1, 3, 2))
    except Exception:
        # fallback: if shape is already (n_tests, n_traj, n_timesteps, n_states)
        pass

    # now compute statistics across trajectories -> shapes (n_tests, n_timesteps, n_states)
    uq_ys_mean_sampled = np.mean(uq_ys, axis=1)
    uq_ys_std = np.std(uq_ys, axis=1)

    # uq_means list contains mean simulations with shape (n_states, n_timesteps)
    uq_ys_mean = np.array(uq_means)
    try:
        uq_ys_mean = np.transpose(uq_ys_mean, (0, 2, 1))
    except Exception:
        # if already (n_tests, n_timesteps, n_states), keep as is
        pass

    # compute bounds in (n_tests, n_timesteps, n_states)
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
            # t_test[i_test] is (n_timesteps, ...) -> use flattened times
            tvals = np.array(t_test[i_test]).squeeze()
            # z_test is (n_sims, n_timesteps, n_states)
            axs[i].plot(tvals, z_test[i_test][:, 0], color="blue")
            # uq_ys_mean, uq_ys_mean_sampled, uq_ys_std have shape (n_tests, n_timesteps, n_states)
            axs[i].plot(tvals, uq_ys_mean[i][:, 0], color="red", linestyle="--")
            axs[i].fill_between(
                tvals,
                uq_ys_mean_sampled[i][:, 0] - 3 * uq_ys_std[i][:, 0],
                uq_ys_mean_sampled[i][:, 0] + 3 * uq_ys_std[i][:, 0],
                color="red",
                alpha=0.3,
            )
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.warning("UQ plotting failed: %s", e)


def plot_latent_phase(z_true, z_mean_preds, test_ids, dims=(0, 1), figsize=(8, 6)):
    """
    Phase plot of latent variables for selected test trajectories.
    z_true: (n_sims, n_timesteps, state_dim)
    z_mean_preds: (n_sims, state_dim, n_timesteps) or list of arrays
    """
    try:
        for idx in test_ids:
            zt = z_true[idx]  # time x state_dim
            zm = z_mean_preds[idx]
            # normalize shapes to (time, state_dim)
            if zm.ndim == 2 and zm.shape[0] == zt.shape[0]:
                zm_t = zm
            else:
                zm_t = zm.T
            plt.figure(figsize=figsize)
            plt.plot(
                zt[:, dims[0]], zt[:, dims[1]], "-o", ms=3, label="Reference", alpha=0.7
            )
            plt.plot(
                zm_t[:, dims[0]],
                zm_t[:, dims[1]],
                "--",
                lw=2,
                label="Mean pred",
                alpha=0.9,
            )
            plt.scatter(
                zt[0, dims[0]], zt[0, dims[1]], c="green", marker="s", label="start"
            )
            plt.xlabel(f"z[{dims[0]}]")
            plt.ylabel(f"z[{dims[1]}]")
            plt.title(f"Latent phase plot - sim {idx}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        logging.warning("plot_latent_phase failed: %s", e)


def plot_rd_uq_imshow(
    veni,
    uq_results,
    pca,
    x_test_original,
    test_ids,
    spatial_shape,
    channel=0,
    times_to_plot=None,
    cmap="viridis",
    figsize=(12, 4),
):
    """
    Show RD (2D) reference, mean prediction, and variance prediction using imshow.
    - uq_results: dict returned by perform_forward_uq
      * 'uq_ys' expected shape: (n_tests, n_traj, state_dim, time)
      * 'uq_ys_mean' expected shape: (n_tests, state_dim, time)
    - V: PCA components matrix (features x components)
    - pca_mean: PCA mean vector (features,)
    - x_test_original: original spatial test data (n_sims, n_timesteps, Nx, Ny, nch)
    - spatial_shape: (Nx, Ny, nch)
    """
    try:
        Nx, Ny, nch = spatial_shape
        n_sims, n_timesteps, _, _, _ = x_test_original.shape
        if times_to_plot is None:
            times_to_plot = [0, n_timesteps // 2, n_timesteps - 1]

        # helper: latent (time,state) -> physical (time,features)
        def latent_to_phys(traj_time_state, V, pca_mean):
            # traj_time_state: (time, state_dim)
            # V: features x components -> reconstruct features = traj @ V.T + mean
            if V.shape[1] != traj_time_state.shape[1]:
                # accept V transposed
                if V.shape[0] == traj_time_state.shape[1]:
                    return traj_time_state.dot(V).astype(np.float32)
                raise ValueError("Incompatible V shape vs latent dim")
            return traj_time_state.dot(V.T) + pca_mean

        for i_idx, idx in enumerate(test_ids):
            # true field
            x_true = x_test_original[idx]  # time x Nx x Ny x nch

            # mean latent
            z_mean = uq_results["uq_ys_mean"][i_idx]
            # ensure shape (time, state)
            if z_mean.ndim == 2 and z_mean.shape[0] == n_timesteps:
                z_mean_t = z_mean
            else:
                z_mean_t = z_mean.T

            # decode latent -> PCA-coordinates using the VENI decoder
            # veni.decode expects input shape (n_samples, state_dim)
            x_pca_mean = veni.decode(z_mean_t)
            x_pca_mean = np.asarray(x_pca_mean)

            # decoded outputs are in the model's scaled space -> rescale back to PCA coordinate space
            try:
                # use the model's rescale method (works with tensors/numpy)
                x_pca_mean = veni.rescale(tf.convert_to_tensor(x_pca_mean)).numpy()
            except Exception:
                # fallback: try dividing by scale_factor if available as numpy
                try:
                    scale = np.array(veni.scale_factor)
                    x_pca_mean = x_pca_mean / scale
                except Exception:
                    # last resort: leave as-is and hope scaling was not applied
                    pass

            # PCA inverse using provided PCA object
            x_mean_phys = pca.inverse_transform(x_pca_mean)
            x_mean_phys = x_mean_phys.reshape(n_timesteps, Nx, Ny, nch)

            # Diagnostics: compare decoded PCA coords (rescaled) to PCA coords of true field
            try:
                # compute true PCA coords from the reference full field
                x_true_flat = x_true.reshape(n_timesteps, -1)  # (time, features)
                pca_coords_true = pca.transform(x_true_flat)

                # ensure shapes align
                if pca_coords_true.shape == x_pca_mean.shape:
                    mae = np.mean(np.abs(pca_coords_true - x_pca_mean))
                    max_err = np.max(np.abs(pca_coords_true - x_pca_mean))
                    logging.info(
                        "PCA-coords reconstruction error (sim %d): MAE=%.6f, max=%.6f",
                        idx,
                        mae,
                        max_err,
                    )
                else:
                    logging.debug(
                        "PCA coords shape mismatch (true %s vs decoded %s)",
                        pca_coords_true.shape,
                        x_pca_mean.shape,
                    )

                # Compare physical fields statistics
                true_min, true_max = x_true.min(), x_true.max()
                rec_min, rec_max = x_mean_phys.min(), x_mean_phys.max()
                logging.info(
                    "Field ranges (sim %d): true[min, max]=[%.4f, %.4f] rec[min, max]=[%.4f, %.4f]",
                    idx,
                    true_min,
                    true_max,
                    rec_min,
                    rec_max,
                )
            except Exception as e:
                logging.debug("Diagnostics failed: %s", e)

            # samples -> phys
            samples = uq_results["uq_ys"][i_idx]  # (n_traj, time, state)

            # decode each sample from latent -> PCA coords, then PCA inverse to phys
            phys_samples_list = []
            for s in samples:
                # s should be (time, state)
                x_pca_s = veni.decode(s)
                x_pca_s = np.asarray(x_pca_s)
                try:
                    x_pca_s = veni.rescale(tf.convert_to_tensor(x_pca_s)).numpy()
                except Exception:
                    try:
                        scale = np.array(veni.scale_factor)
                        x_pca_s = x_pca_s / scale
                    except Exception:
                        pass
                x_full_s = pca.inverse_transform(x_pca_s)
                phys_samples_list.append(x_full_s.reshape(n_timesteps, Nx, Ny, nch))
            phys_samples = np.stack(phys_samples_list, axis=0)
            x_std_phys = np.std(phys_samples, axis=0)

            for t_idx in times_to_plot:
                if t_idx < 0 or t_idx >= n_timesteps:
                    continue
                fig, axs = plt.subplots(1, 3, figsize=figsize)
                vmin = min(
                    x_true[t_idx, :, :, channel].min(),
                    x_mean_phys[t_idx, :, :, channel].min(),
                )
                vmax = max(
                    x_true[t_idx, :, :, channel].max(),
                    x_mean_phys[t_idx, :, :, channel].max(),
                )

                im0 = axs[0].imshow(
                    x_true[t_idx, :, :, channel], cmap=cmap, vmin=vmin, vmax=vmax
                )
                axs[0].set_title(f"Reference (sim {idx}) t={t_idx}")
                plt.colorbar(im0, ax=axs[0])

                im1 = axs[1].imshow(
                    x_mean_phys[t_idx, :, :, channel], cmap=cmap, vmin=vmin, vmax=vmax
                )
                axs[1].set_title("Mean prediction")
                plt.colorbar(im1, ax=axs[1])

                im2 = axs[2].imshow(x_std_phys[t_idx, :, :, channel], cmap="magma")
                axs[2].set_title("Prediction std")
                plt.colorbar(im2, ax=axs[2])

                plt.suptitle(f"Simulation {idx} - time {t_idx}")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
    except Exception as e:
        logging.warning("plot_rd_uq_imshow failed: %s", e)


# ----------------------
# Main
# ----------------------


def main():
    set_seed(42)

    # Load data
    (
        t,
        x,
        dxdt,
        t_test,
        x_test,
        dxdt_test,
        pca,
        spatial_shape,
        x_test_original,
        n_sims,
        n_timesteps,
        n_sims_test,
        n_timesteps_test,
    ) = load_data(pca_order=PCA_ORDER, noise=NOISE, nth_time_step=NTH_TIME_STEP)

    # Create model
    veni = create_model(x, params=None, dt=t[1] - t[0], n_dof=x.shape[1])

    # Scale data
    veni.define_scaling(x)
    x_train_scaled, dxdt_train_scaled = veni.scale(x), veni.scale(dxdt)
    x_test_scaled, dxdt_test_scaled = veni.scale(x_test), veni.scale(dxdt_test)

    # Prepare inputs for training (keep original slicing heuristic)
    split_train = (
        int((n_sims - 2) * n_timesteps) if n_timesteps else int(0.7 * x.shape[0])
    )
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
        veni.sindy_layer.pdf_thresholding(threshold=0.1)
    except Exception as e:
        logging.warning("Sparsification failed: %s", e)

    # Inference + UQ
    logging.info("Performing inference and forward UQ...")
    n_traj = 10
    test_ids = list(range(n_sims_test))

    z_preds, t_preds = perform_inference(
        veni,
        x_test_scaled,
        dxdt_test_scaled,
        t_test,
        test_ids,
        n_sims_test,
        n_timesteps_test,
    )

    uq_results = perform_forward_uq(
        veni,
        x_test_scaled,
        dxdt_test_scaled,
        t_test,
        test_ids,
        n_traj,
        n_sims_test,
        n_timesteps_test,
    )

    uq_plots(
        uq_results["uq_ts"],
        uq_results["uq_ys_mean"],
        uq_results["uq_ys_mean_sampled"],
        uq_results["uq_ys_std"],
        switch_data_format(t_test, n_sims_test, n_timesteps_test, target_format="3d"),
        uq_results["z_test"],
        test_ids,
    )

    # New plots: latent phase and RD UQ images
    try:
        plot_latent_phase(uq_results["z_test"], uq_results["uq_ys_mean"], test_ids)
        plot_rd_uq_imshow(
            veni,
            uq_results,
            pca,
            x_test_original,
            test_ids,
            spatial_shape,
            channel=0,
        )
    except Exception as e:
        logging.warning("Additional plotting failed: %s", e)

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
