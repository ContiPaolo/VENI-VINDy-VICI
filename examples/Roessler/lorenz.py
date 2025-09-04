# Perform all necessary imports
import tensorflow as tf
import os
import numpy as np
import matplotlib
from vindy import SindyNetwork
from vindy.libraries import PolynomialLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Laplace
from vindy.callbacks import (
    SaveCoefficientsCallback,
)

from utils import (
    data_generation,
    generate_directories,
    data_plot,
    training_plot,
    trajectory_plot,
    uq_plot,
)

matplotlib.use("TkAgg")

"""
# ## Roessler System
# The Roessler system is a system of three ordinary differential equations that describe a simple chaotic system. The equations are given by:
# $ \dot{z}_1 = -z_2 - z_3 $
# $ \dot{z}_2 = z_1 + 0.2*z_2 $
# $ \dot{z}_3 = 0.2 + z_3(z_1-5.7) = 0.2 + z_3 z_1 - 5.7z_3 $
"""



def roessler(t, x0, a=0.2, b=0.2, c=5.7):
    x, y, z = x0
    return [-y - z, x + a * y, b + z * (x - c)]


# Define the Lotka-Volterra ODE
# dxdt = 1.0*x - 0.1*x*y
# dydt = -1.5*y + 0.075*x*y
def lotka_volterra(t, x0, a=1.0, b=0.1, c=1.5, d=0.075):
    x, y = x0
    return [a * x - b * x * y, -c * y + d * x * y]


def lorenz(t, x0, a=10, b=28, c=8 / 3):
    x, y, z = x0
    return [a * (y - x), x * (b - z) - y, x * y - c * z]


# Let's define some general script parameters including the noise factors, the number of training and test trajectories, and the random seed. We will also define the type of SINDy model we want to use (VINDy or SINDy) and the model name (Roessler or LV). With the following setup we train our system on data with measurement noise and model noise. We will also use random initial conditions and parameters for the Roessler system.

# In[33]:
script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir, "results")

sindy_type = "vindy"  # "sindy" or "vindy"
model_name = "lorenz"  # lotka_volterra or roessler or lorenz
out_name = "lorenz_high_vindy"
seed = 69  # random seed
seed_ = 69
random_IC = True  # use random initial conditions
random_a = True  # use random parameters (a in the Roessler system)
measurement_noise_factor = 0  # 0.01  # measurement noise factor
model_noise_factor = 0  # 0.1  # model noise factor
n_train = 100  # number of training trajectories
n_test = 3  # number of test trajectories
include_bias = True  # include bias term in the library
epochs = 1500  # number of epochs for training
train = True
l_dz = 1
l_vindy = 1e-2
ode = dict(roessler=roessler, lotka_volterra=lotka_volterra, lorenz=lorenz)[model_name]
learning_rate = 0.0025
# Let's define directories for saving the results

# In[34]:
scenarios = [(0, 0), (0, 0.005), (0, 0.01), (0.005, 0.0), (0.01, 0.0), (0.01, 0.01)]
# scenarios = [(0, 0), (0, 0.005)]
save_data = dict()
for measurement_noise_factor, model_noise_factor in scenarios:

    # noise before derivative, model error, seed
    scenario_info = f"{sindy_type}_nbd__me_{random_a}_{model_noise_factor}_seed_{seed_}_noise_{measurement_noise_factor}"
    outdir, figdir, log_dir, weights_dir = generate_directories(
        out_name, sindy_type, scenario_info, "results"
    )

    # First, we will generate the data for the Roessler system that we can use to train the VINDy model. We will use the `scipy.integrate.solve_ivp` function to solve the ODEs and generate the data. Let's plot the data to see what it looks like
    #

    # In[35]:

    (t, x, dxdt, x_test, dxdt_test, var_names, dim) = data_generation(
        ode,
        n_train,
        n_test,
        random_IC,
        random_a,
        seed,
        model_noise_factor,
        measurement_noise_factor,
        model=model_name,
    )

    # data_plot(t, x, dxdt, x_test)

    save_data[scenario_info] = dict(x=x, x_test=x_test)
    # ## Model Generation
    #
    # Now, we will define the VINDy model and train it on the generated data to learn the Roessler system. We will use the `VariationalSindyLayer` to define the model. The `VariationalSindyLayer` is a Bayesian version of the SINDy model that uses a variational inference approach to learn the model coefficients. We will use the `Laplacian` priors for the coefficients and use a polynomial library of degree 2 to learn the model.

    # In[36]:

    # reshape data to fit the model
    x_train = np.concatenate(x, axis=0)
    dxdt_train = np.concatenate(dxdt, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    dxdt_test = np.concatenate(dxdt_test, axis=0)

    # model parameters
    libraries = [
        PolynomialLibrary(2, include_bias=include_bias),
    ]
    dt = t[1] - t[0]

    # create sindy layer
    layer_params = dict(
        state_dim=x_train.shape[1],
        param_dim=0,
        feature_libraries=libraries,
        second_order=False,
        mask=None,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=0),
    )
    if sindy_type == "vindy":
        sindy_layer = VindyLayer(
            beta=l_vindy,
            priors=Laplace(0.0, 1.0),
            **layer_params,
        )
    elif sindy_type == "sindy":
        sindy_layer = SindyLayer(
            **layer_params,
        )
    else:
        raise ValueError(f"Unknown SINDy type: {sindy_type}")

    # create autoencoder sindy model
    model = SindyNetwork(
        sindy_layer=sindy_layer,
        x=x_train,
        l_dz=l_dz,
        dt=dt,
        second_order=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="huber"
    )

    model.build(input_shape=([x_train.shape, dxdt_train.shape], None))

    # Let's train the VINDy model on the (noisy) Roessler data

    # In[37]:

    weights_path = os.path.join(weights_dir, ".weights.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(weights_path),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
        ),
        SaveCoefficientsCallback(),
    ]

    if train:
        train_val_id = int(0.7 * x_train.shape[0])
        input_train = [x_train[:train_val_id], dxdt_train[:train_val_id]]
        input_val = [x_train[train_val_id:], dxdt_train[train_val_id:]]
        trainhist = model.fit(
            x=input_train,
            validation_data=(input_val, None),
            callbacks=callbacks,
            y=None,
            epochs=epochs,
            batch_size=256,
            verbose=2,
        )

        mean_over_epochs = np.array(trainhist.history["coeffs_mean"]).squeeze()
        scale_over_epochs = np.array(trainhist.history["coeffs_scale"]).squeeze()
        np.savetxt(
            os.path.join(outdir, f"{scenario_info}_coeffs_mean.csv"),
            mean_over_epochs[-1],
            delimiter=",",
        )
        np.savetxt(
            os.path.join(outdir, f"{scenario_info}_coeffs_scale.csv"),
            scale_over_epochs[-1],
            delimiter=",",
        )
        save_data[scenario_info]["mean_over_epochs"] = mean_over_epochs
        save_data[scenario_info]["scale_over_epochs"] = scale_over_epochs
        save_data[scenario_info]["trainhist"] = trainhist

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(mean_over_epochs)
        plt.title("Mean Coefficients over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Coefficient")

        plt.figure()
        plt.plot(scale_over_epochs)
        plt.title("Scale Coefficients over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Scale Coefficient")

        # plot training history
        training_plot(sindy_layer, trainhist, var_names)

    # load best weights
    model.load_weights(os.path.join(weights_path))
    # apply pdf threshold
    sindy_layer.pdf_thresholding(threshold=0.01)

    # Let's plot the training history of the VINDy model and check how the coefficients evolved during training.
    # We plot the coefficients of the VINDy model to see which terms are learned by the model.

    equation = sindy_layer.model_equation_to_str(z=var_names, precision=3)
    save_data[scenario_info]["equation"] = equation
    dxdt_pred = model.sindy(x_test).numpy()
    save_data[scenario_info]["dxdt_pred"] = dxdt_pred

    # %% integration
    nt = t.shape[0]
    i_test = 1
    # integrate the model
    t_0 = i_test * int(nt)
    sol = sindy_layer.integrate(x_test[t_0 : t_0 + 1].squeeze(), t.squeeze(), mu=None)
    t_pred = sol.t
    x_pred = sol.y

    # trajectory_plot(t, x_test, t_pred, x_pred, dim, nt, i_test, var_names)

    # ## Uncertainty quantification
    # Instead of only using the mean coefficients for a single prediction, we can also sample various models and use them to predict the future states of the Roessler system. This will give us an idea of the uncertainty in the model predictions.
    # Forward UQ:
    # * We sample SINDy coefficients from the predicted posterior distribution
    # * We integrate the ODE with the sampled coefficients and collect the trajectories

    # In[ ]:

    n_traj = 20
    # Store the original coefficients
    kernel_orig, kernel_scale_orig = sindy_layer.kernel, sindy_layer.kernel_scale

    t_preds = []
    x_preds = []
    t_0 = i_test * int(nt)
    print(f"test trajectory {i_test}")
    # List to store the solution trajectories in latent space
    for traj in range(n_traj):
        print(f"\t sample {traj+1} out of {n_traj}")
        # Sample from the posterior distribution of the coefficients
        # sampled_coeff, _, _ = sindy_layer._coeffs
        # only take non-zero coefficients
        # sampled_coeff = sampled_coeff.numpy()
        # sampled_coeff = sampled_coeff[sampled_coeff != 0]
        #
        # # Assign the sampled coefficients to the SINDy layer
        # sindy_layer.kernel = tf.reshape(sampled_coeff, (-1, 1))
        # print(np.abs(sindy_layer.kernel.numpy()).sum())
        # sindy_layer.set_coeffs_for_uq(sampled_coeff)
        # sol = sindy_layer.integrate(x_test[t_0 : t_0 + 1].squeeze(), t.squeeze())
        sol, coeffs = sindy_layer.integrate_uq(
            x_test[t_0 : t_0 + 1].squeeze(), t.squeeze()
        )

        # print(np.abs(model.sindy.kernel.numpy()).sum())
        t_preds.append(sol.t)
        x_preds.append(sol.y.T)
    # restore original coefficients
    # calculate mean and variance of the trajectories
    x_uq = np.array(x_preds)
    x_uq_mean_sampled = np.mean(x_uq, axis=0)
    x_uq_std = np.std(x_uq, axis=0)

    save_data[scenario_info]["uq_ts"] = np.array(t_preds)
    save_data[scenario_info]["uq_ys"] = x_uq
    save_data[scenario_info]["uq_ys_mean"] = x_uq_mean_sampled
    save_data[scenario_info]["uq_ys_std"] = x_uq_std

    # UQ plot
    import matplotlib.pyplot as plt

    uq_plot(t, x_test, t_preds, x_preds, x_uq_mean_sampled, x_uq_std, dim, nt, i_test)

    #
    coeffs = np.array([sindy_layer._coeffs[0].numpy() for i in range(10)])
    std = np.std(coeffs, axis=0)


plt.show()

# save the data to a file
# if train:
#     with open(
#         os.path.join(outdir, f"save_data_{model_name}_{sindy_type}.pkl"), "wb"
#     ) as f:
#         pickle.dump(save_data, f)
# else:
#     with open(
#         os.path.join(outdir, f"save_data_{model_name}_{sindy_type}.pkl"), "rb"
#     ) as f:
#         save_data = pickle.load(f)


# %% save the data into csv's for tikz

nth_step = 1

j = 0

# trajectories
n_traj = i_test
n_states = 3
for key, value in save_data.items():
    t = value["uq_ts"][0, ::nth_step]
    n_t = t.shape[0]
    y_ref = value["x_test"].transpose([0, 2, 1])[:n_traj, :, ::nth_step]
    y_mean = value["uq_ys_mean"][np.newaxis].transpose([0, 2, 1])[:, :, ::nth_step]
    y_std = value["uq_ys_std"][np.newaxis].transpose([0, 2, 1])[:, :, ::nth_step]
    ub = y_mean + 3 * y_std
    lb = y_mean - 3 * y_std
    # Plot reference and mean predictions for each state and trajectory
    fig, ax = plt.subplots(n_traj, n_states, figsize=(10, 10))
    ax = np.atleast_2d(ax)
    for i in range(n_traj):
        for j in range(n_states):
            ax[i, j].plot(t, y_ref[i, j], label="Reference")
            ax[i, j].plot(t, y_mean[i, j], label="Mean")
            ax[i, j].fill_between(
                t,
                y_mean[i, j] - 3 * y_std[i, j],
                y_mean[i, j] + 3 * y_std[i, j],
                alpha=0.3,
            )
            ax[i, j].set_xlabel("Time")
            ax[i, j].set_ylabel(f"Traj {i+1}")
            ax[i, j].set_title(f"${var_names[i]}_{j+1}$")
            ax[i, j].legend()

    header = ["t"]
    header += [
        f"{var_names[j]}_traj_{i}"
        for i in range(1, n_traj + 1)
        for j in range(0, n_states)
    ]
    header += [
        f"{var_names[j]}_traj_{i}_mean"
        for i in range(1, n_traj + 1)
        for j in range(0, n_states)
    ]
    header += [
        f"{var_names[j]}_traj_{i}_lb"
        for i in range(1, n_traj + 1)
        for j in range(0, n_states)
    ]
    header += [
        f"{var_names[j]}_traj_{i}_ub"
        for i in range(1, n_traj + 1)
        for j in range(0, n_states)
    ]
    # save a csv which maps the scenario info to index
    np.savetxt(
        os.path.join(outdir, f"{key}_traj.csv"),
        np.concatenate(
            [
                t[np.newaxis],
                y_ref.reshape([-1, n_t]),
                y_mean.reshape([-1, n_t]),
                lb.reshape([-1, n_t]),
                ub.reshape([-1, n_t]),
            ]
        ).T,
        delimiter=",",
        fmt="%s",
        comments="",
        header=",".join(header),
    )

# distributions
save_values = dict()
for scenario_info, data in save_data.items():
    # training and test data
    n_ics = len(data["x"])
    states = np.concatenate(data["x"][:, ::nth_step], axis=1)
    state_headers = [f"{var_names[i]}_{j}" for j in range(n_ics) for i in range(dim)]
    test_states = np.concatenate(data["x_test"][:, ::nth_step], axis=1)
    test_state_headers = [
        f"{var_names[i]}_test_{j}" for j in range(n_test) for i in range(dim)
    ]
    # save csv with headers for states
    np.savetxt(
        # os.path.join(figdir, f'{scenario_info}_states.csv'),
        os.path.join(outdir, f"vindy_states_{j}.csv"),
        np.concatenate([states, test_states], axis=1),
        delimiter=",",
        comments="",
        header=",".join(state_headers + test_state_headers),
    )
    # save distributions
    feature_names = [
        name_.replace("*", "") for name_ in sindy_layer.get_feature_names(var_names)
    ]
    n_vars = len(var_names)
    n_features = len(feature_names)
    # mean_values, scale_values = (
    #     np.array(data["mean_over_epochs"][-1]).reshape(n_vars, n_features).T,
    #     np.array(np.exp(data["scale_over_epochs"])[-1]).reshape(n_vars, n_features).T,
    # )
    # for i in range(n_vars):
    #     if j == 0:
    #         save_value = np.concatenate(
    #             [
    #                 np.array(feature_names)[:, np.newaxis],
    #                 mean_values[:, i : i + 1],
    #                 scale_values[:, i : i + 1],
    #             ],
    #             axis=1,
    #         ).T
    #         # save_value = np.concatenate([save_value, np.array([scenario_info] * save_value.shape[0]).reshape(save_value.shape[0], 1)],
    #         #                             axis=1)
    #         save_values[i] = save_value
    #         print("j")
    #     else:
    #
    #         save_value = np.concatenate(
    #             [mean_values[:, i : i + 1], scale_values[:, i : i + 1]], axis=1
    #         ).T
    #         # save_value = np.concatenate([save_value, np.array([scenario_info] * save_value.shape[0]).reshape(save_value.shape[0], 1)],
    #         #                             axis=1)
    #         save_values[i] = np.concatenate([save_values[i], save_value], axis=0)
    # j += 1

# save a csv which maps the scenario info to index
np.savetxt(
    os.path.join(outdir, f"vindy_scenario_info.csv"),
    np.array(list(save_data.keys()))[:, np.newaxis].T,
    delimiter=",",
    fmt="%s",
    comments="",
    header=",".join(np.array(range(len(save_data.keys()))).astype(str)),
)

for i in range(n_vars):
    np.savetxt(
        os.path.join(outdir, f"vindy_{var_names[i]}_dot.csv"),
        # os.path.join(figdir, f'{scenario_info}_coeffs_{var_names[i]}_dot.csv'),
        save_values[i],
        delimiter=",",
        fmt="%s",
        comments="",
        header=",".join(np.array(range(len(feature_names))).astype(str)),
    )
