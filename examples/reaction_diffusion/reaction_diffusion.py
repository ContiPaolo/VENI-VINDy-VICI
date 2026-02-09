import os
import sys
import logging
import datetime
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# local imports
from vindy import VENI
from vindy.libraries import PolynomialLibrary
from vindy.layers import VindyLayer
from vindy.distributions import Laplace
from vindy.callbacks import SaveCoefficientsCallback
from vindy.utils import switch_data_format
from utils import load_reaction_diffusion_data

from examples.utils import (
    set_seed,
    plot_train_history,
    plot_coefficients_train_history,
    get_config,
    perform_inference,
    plot_inference_results,
    perform_forward_uq,
    uq_plots,
)

config = get_config()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ----------------------
# Default script constants (mirrors previous top-level script)
# ----------------------
MODEL_NAME = "reactiondiffusion"
REDUCED_ORDER = 2
PCA_ORDER = 32
NOISE = True
NTH_TIME_STEP = 3
SECOND_ORDER = False
PRETRAINING = (
    False  # pretrain for reconstruction only to stabilize training and avoid degene
)

BETA_VINDY = 1e-4
BETA_VAE = 2e-5
L_REC = 1e-2
L_DZ = 4e0
L_DX = 1e-2

# PRETRAINED
BETA_VINDY = 1e-4
BETA_VAE = 2e-5
L_REC = 1e-3
L_DZ = 4e0
L_DX = 1e-3

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

# Training defaults (can be overridden by calling train_model with args)
LOAD_MODEL = True  # False
EPOCHS = 5000
BATCH_SIZE = None  # computed later based on data

# ----------------------
# Utility functions
# ----------------------


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
        f"{model_name}/{model_name}_{REDUCED_ORDER}_{veni.__class__.__name__}.weights.h5",
    )
    train_histdir = os.path.join(result_dir, f"{model_name}/trainhist.npy")

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


def training_plots(trainhist, result_dir, x_train_scaled, x_test_scaled, veni):
    """Plot training results and coefficient history."""
    try:
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


def plot_latent_phase(z_true, z_mean_preds, test_ids, dims=(0, 1), figsize=(8, 6)):
    """
    Phase plot of latent variables for selected test trajectories.
    z_true: (n_sims, n_timesteps, state_dim)
    z_mean_preds: (n_sims, state_dim, n_timesteps) or list of arrays
    """
    for idx in test_ids:
        zt = z_true[idx]  # time x state_dim
        zm = z_mean_preds[idx]
        # normalize shapes to (time, state_dim)
        plt.figure(figsize=figsize)
        plt.plot(
            zt[:, dims[0]], zt[:, dims[1]], "-o", ms=3, label="Reference", alpha=0.7
        )
        plt.plot(
            zm[:, dims[0]],
            zm[:, dims[1]],
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
    Nx, Ny, nch = spatial_shape
    n_sims, n_timesteps, _, _, _ = x_test_original.shape
    if times_to_plot is None:
        times_to_plot = [0, n_timesteps // 2, n_timesteps - 1]

    for i_idx, idx in enumerate(test_ids):
        # true field
        x_true = x_test_original[idx]  # time x Nx x Ny x nch

        # mean latent
        z_mean = uq_results["mean_latent"][i_idx]

        # decode latent -> PCA-coordinates using the VENI decoder
        x_pca_mean = veni.decode(z_mean).numpy()

        # use the model's rescale method
        x_pca_mean = veni.rescale(x_pca_mean).numpy()

        # PCA inverse using provided PCA object
        x_mean_phys = pca.inverse_transform(x_pca_mean)
        x_mean_phys = x_mean_phys.reshape(n_timesteps, Nx, Ny, nch)

        # samples -> phys
        samples = uq_results["latent_trajectories_samples"][
            i_idx
        ]  # (n_traj, time, state)

        # decode each sample from latent -> PCA coords, then PCA inverse to phys
        phys_samples_list = []
        for s in samples:
            # s should be (time, state)
            x_pca_s = veni.decode(s).numpy()
            x_pca_s = veni.rescale(x_pca_s).numpy()
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


# ----------------------
# Main
# ----------------------


def main():
    set_seed(23)

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
        # sindy_optimizer=tf.keras.optimizers.AdamW(learning_rate=8e-4),
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
    veni.sindy_layer.pdf_thresholding(threshold=0.1)

    # Inference + UQ
    logging.info("Performing inference and forward UQ...")
    n_traj = 10
    test_ids = list(range(n_sims_test))

    Z, z_preds, t_preds = perform_inference(
        veni,
        test_ids,
        n_sims_test,
        n_timesteps_test,
        t_test,
        x_test_scaled,
        dxdt_test_scaled,
    )
    T = switch_data_format(t_test, n_sims_test, n_timesteps_test, target_format="3d")
    plot_inference_results(t_preds, z_preds, T, Z, test_ids)

    uq_results = perform_forward_uq(
        veni,
        test_ids,
        n_traj,
        n_sims_test,
        n_timesteps_test,
        t_test,
        x_test_scaled,
        dxdt_test_scaled,
    )

    uq_plots(
        uq_results["sampled_times"],
        uq_results["mean_latent"],
        uq_results["mean_latent_samples"],
        uq_results["std_latent_samples"],
        switch_data_format(t_test, n_sims_test, n_timesteps_test, target_format="3d"),
        uq_results["z"],
        test_ids,
    )

    # New plots: latent phase and RD UQ images
    plot_latent_phase(uq_results["z"], uq_results["mean_latent"], test_ids)
    plot_rd_uq_imshow(
        veni,
        uq_results,
        pca,
        x_test_original,
        test_ids,
        spatial_shape,
        channel=0,
    )


if __name__ == "__main__":
    main()
