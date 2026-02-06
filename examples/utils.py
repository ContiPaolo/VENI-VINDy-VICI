"""
Shared utility functions for examples.

This module contains common functions used across different example scripts
(MEMS, reaction_diffusion, etc.) to avoid code duplication.
"""

import os
import random
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def set_seed(seed: int):
    """
    Set seed for reproducibility in TensorFlow, NumPy, and Python's random module.

    Args:
        seed (int): The seed value to set.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def validate_data_path(data_path: str, zenodo_doi: str = "10.5281/zenodo.18313843"):
    """
    Validate that a data file exists and provide a helpful error message if not.

    Args:
        data_path (str): Path to the data file.
        zenodo_doi (str): Zenodo DOI for downloading the data (default: 10.5281/zenodo.18313843).

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Data file {data_path} not found. "
            f"Please download the file from Zenodo (http://doi.org/{zenodo_doi}) and "
            f"specify the correct path in the examples/config.py file."
        )


def plot_train_history(trainhist, result_dir, validation=True):
    """
    Plot training history including loss curves.

    Args:
        trainhist (dict): Training history dictionary containing loss values.
        result_dir (str): Directory to save the plot.
        validation (bool): Whether to include validation loss in the plot.
    """
    try:
        os.makedirs(result_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot training loss
        if "loss" in trainhist:
            ax.plot(trainhist["loss"], label="Training Loss", linewidth=2)

        # Plot validation loss if requested and available
        if validation and "val_loss" in trainhist:
            ax.plot(trainhist["val_loss"], label="Validation Loss", linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training History", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()

        # Save figure
        suffix = "_val" if validation else "_train"
        save_path = os.path.join(result_dir, f"training_history{suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved training history plot to {save_path}")

        plt.close(fig)
    except Exception as e:
        logging.warning(f"Failed to plot training history: {e}")


def plot_coefficients_train_history(trainhist, result_dir):
    """
    Plot the evolution of SINDy coefficients during training.

    Args:
        trainhist (dict): Training history dictionary.
        result_dir (str): Directory to save the plot.
    """
    try:
        os.makedirs(result_dir, exist_ok=True)

        # Check if coefficient history is available
        if "sindy_coefficients" not in trainhist:
            logging.warning("No SINDy coefficients found in training history")
            return

        coeffs = np.array(trainhist["sindy_coefficients"])

        # Plot coefficient evolution
        fig, ax = plt.subplots(figsize=(12, 6))

        n_coeffs = coeffs.shape[-1] if coeffs.ndim > 1 else 1
        for i in range(n_coeffs):
            if coeffs.ndim > 1:
                ax.plot(coeffs[:, i], label=f"Coeff {i}", linewidth=1.5)
            else:
                ax.plot(coeffs, label=f"Coeff {i}", linewidth=1.5)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Coefficient Value", fontsize=12)
        ax.set_title("SINDy Coefficient Evolution", fontsize=14, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(result_dir, "coefficients_history.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved coefficient history plot to {save_path}")

        plt.close(fig)
    except Exception as e:
        logging.warning(f"Failed to plot coefficient history: {e}")


def create_result_directory(base_dir: str, model_name: str) -> str:
    """
    Create a result directory for saving model outputs.

    Args:
        base_dir (str): Base directory for results.
        model_name (str): Name of the model/experiment.

    Returns:
        str: Path to the created result directory.
    """
    result_dir = os.path.join(base_dir, model_name)
    os.makedirs(result_dir, exist_ok=True)
    logging.info(f"Result directory: {result_dir}")
    return result_dir


def log_model_summary(veni, result_dir: str = None):
    """
    Log a summary of the VENI model architecture.

    Args:
        veni: The VENI model instance.
        result_dir (str, optional): Directory to save the summary text file.
    """
    try:
        logging.info("=" * 60)
        logging.info("Model Summary:")
        logging.info("=" * 60)

        # Log key parameters
        logging.info(f"Reduced order: {veni.reduced_order}")
        logging.info(f"Second order: {veni.second_order}")
        logging.info(f"Scaling method: {veni.scaling}")

        # Get model summary
        summary_lines = []
        veni.summary(print_fn=lambda x: summary_lines.append(x))

        for line in summary_lines:
            logging.info(line)

        # Save to file if directory provided
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            summary_path = os.path.join(result_dir, "model_summary.txt")
            with open(summary_path, "w") as f:
                f.write("\n".join(summary_lines))
            logging.info(f"Model summary saved to {summary_path}")

        logging.info("=" * 60)
    except Exception as e:
        logging.warning(f"Failed to log model summary: {e}")
