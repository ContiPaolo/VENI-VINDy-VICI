import tensorflow as tf
import os
import numpy as np
from vindy import SindyNetwork
from vindy.libraries import PolynomialLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Gaussian, Laplace
from vindy.callbacks import (
    SaveCoefficientsCallback,
)
from Roessler_utils import (
    data_generation,
    generate_directories,
    data_plot,
    training_plot,
    trajectory_plot,
    uq_plot,
)


def roessler(t, x0, a=0.2, b=0.2, c=5.7):
    x, y, z = x0
    return [-y - z, x + a * y, b + z * (x - c)]


sindy_type = "vindy"  # "sindy" or "vindy"
model_name = "roessler"  # LV or roessler
seed = 29  # random seed
random_IC = True  # use random initial conditions
random_a = True  # use random parameters (a in the Roessler system)
measurement_noise_factor = 0.01  # measurement noise factor
model_noise_factor = 0.1  # model noise factor
n_train = 30  # number of training trajectories
n_test = 4  # number of identification_layer trajectories

# noise before derivative, model error, seed
scenario_info = f"{sindy_type}_nbd__me_{random_a}_{model_noise_factor}_seed_{seed}_noise_{measurement_noise_factor}"
_, _, _, weights_dir = generate_directories(
    model_name, sindy_type, scenario_info, "results"
)

(t, x_noise_free, dxdt_noise_free, _, _, _, _) = data_generation(
    roessler,
    n_train,
    n_test,
    random_IC,
    random_a,
    seed,
    0,
    0,
)
dxdt_noise_free = np.array(dxdt_noise_free)

(_, x_noise, dxdt_noise, _, _, _, _) = data_generation(
    roessler,
    n_train,
    n_test,
    random_IC,
    random_a,
    seed,
    0,
    0.05,
)
dxdt_noise = np.array(dxdt_noise)


# %% signal noise ratio
rms = lambda x: np.sqrt(np.mean(x**2))

SNR_state = rms(x_noise_free) / rms(x_noise - x_noise_free)
SNR_state_derivative = rms(dxdt_noise_free) / rms(dxdt_noise - dxdt_noise_free)

SNR_db_state = 20 * np.log10(SNR_state)
SNR_db_state_derivative = 20 * np.log10(SNR_state_derivative)

data_plot(t, x_noise_free, dxdt_noise_free, [])
data_plot(t, x_noise, dxdt_noise, [])
