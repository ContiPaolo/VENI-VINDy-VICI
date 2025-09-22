import numpy as np
import os
import matplotlib.pyplot as plt
import imageio


def add_lognormal_noise(trajectory, sigma):
    noise = np.random.lognormal(mean=0, sigma=sigma, size=trajectory.shape)
    return trajectory * noise, noise


def coefficient_distributions_to_csv(sindy_layer, outdir, var_names=[], param_names=[]):
    """
    Save the coefficient distributions of the SINDy layer to csv files
    :param sindy_layer:
    :param outdir:
    :param var_names:
    :return:
    """
    if not var_names:
        var_names = [f"z{i}" for i in range(1, sindy_layer.output_dim + 1)]
    if not param_names:
        param_names = [f"p_{i}" for i in range(1, sindy_layer.param_dim + 1)]
    feature_names = [
        name_.replace("*", "")
        for name_ in sindy_layer.get_feature_names(var_names, param_names)
    ]
    n_vars = sindy_layer.state_dim
    n_features = len(feature_names)
    _, mean, log_scale = sindy_layer._coeffs
    # reverse log_scale
    scale = sindy_layer.priors.reverse_log(log_scale.numpy())

    mean_values, scale_values = (
        mean.numpy().reshape(n_vars, n_features).T,
        scale.numpy().reshape(n_vars, n_features).T,
    )

    # minimum scale value for which Laplacian dist can be plotted in pgfplots is 1e-4
    scale_values = np.maximum(scale_values, 1e-4)

    for i in range(n_vars):
        save_value = np.concatenate(
            [
                np.array(feature_names)[:, np.newaxis],
                mean_values[:, i : i + 1],
                scale_values[:, i : i + 1],
            ],
            axis=1,
        ).T
        np.savetxt(
            os.path.join(outdir, f"vindy_{var_names[i]}_dot.csv"),
            save_value,
            delimiter=",",
            fmt="%s",
            comments="",
            header=",".join(np.array(range(len(feature_names))).astype(str)),
        )


def coefficient_distribution_gif(
    mean_over_epochs, scale_over_epochs, sindy_layer, outdir, model_name, config
):
    """
    Create a gif showing how the coefficient distributions evolve over time
    """
    os.makedirs(os.path.join(outdir, "coefficients"), exist_ok=True)
    # create gif showing how the coefficient distributions evolve over time
    for i, (mean_, scale_) in enumerate(zip(mean_over_epochs, scale_over_epochs)):
        # if i % 10 == 0:
        if i <= 400:
            x_range = 1.5  # - (1.5 * i / len(mean_over_epochs))
            # dont show figure
            fig = sindy_layer._visualize_coefficients(
                mean_, scale_, x_range=[-x_range, x_range], y_range=[0, 6]
            )
            # fig title
            fig.suptitle(f"Epoch {i}")
            # save fig as frame for gif
            fig.savefig(os.path.join(outdir, "coefficients", f"coeffs_{i}.png"))
            plt.close(fig)
    # make gif from frames
    images = []
    for i in range(0, 400, 1):
        images.append(
            imageio.imread(os.path.join(outdir, "coefficients", f"coeffs_{i}.png"))
        )
    imageio.mimsave(
        os.path.join(config.outdir, "coefficients", "coeffs.gif"),
        images,
        duration=100,
    )


def plot_train_history(history, outdir, validation: bool = True):
    """
    Plot the training history
    :param history:
    :param outdir:
    :return:
    """
    os.makedirs(outdir, exist_ok=True)
    # plot training history
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for loss_term, loss_values in history.items():
        if (
            (not validation and "val_" in loss_term)
            or (validation and "val_" not in loss_term)
        ) and "coeffs" not in loss_term:
            ax.plot(loss_values, label=loss_term)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()
    fig.savefig(os.path.join(outdir, "training_history.png"))
    plt.close(fig)


def plot_coefficients_train_history(history, outdir):
    """
    Plot the coefficient training history
    :param history:
    :param outdir:
    :return:
    """
    mean_over_epochs = np.array(history["coeffs_mean"]).squeeze()
    scale_over_epochs = np.array(history["coeffs_scale"]).squeeze()
    os.makedirs(outdir, exist_ok=True)
    # plot training history
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0].plot(mean_over_epochs)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Coefficient mean")
    ax[1].plot(scale_over_epochs)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Coefficient scale")
    plt.show()
    fig.savefig(os.path.join(outdir, "coefficients_history.png"))
    plt.close(fig)
