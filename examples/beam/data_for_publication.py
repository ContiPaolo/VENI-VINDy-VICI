import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from aesindy import SindyNetwork, config
from preprocess_data import load_beam_data
from visualizer import Visualizer

model_name = "beam"  # LV or roessler
sindy_type = "vindy"  # "sindy" or "vindy"
outdir = os.path.join(config.outdir, f"{model_name}")
# Load data
with open(os.path.join(outdir, f"save_data_{model_name}.pkl"), "rb") as f:
    data = pickle.load(f)

t = data["uq_ts"][0][0]
n_t = len(t)
n_th_timestep = 2

# %% 3d visualization
(
    _,
    _,
    _,
    _,
    _,
    _,
    _,
    _,
    _,
    _,
    ref_coords,
    _,
    _,
    _,
) = load_beam_data(config.beam["processed_data"])

step_size = 2.675520
ref_coords_adj = ref_coords
ref_coords_adj[:, 0] = ref_coords_adj[:, 0] / step_size
ref_coords_adj[:, 1] = ref_coords_adj[:, 1] / 10 * 5
ref_coords_adj[:, 2] = ref_coords_adj[:, 2] / 24 * 3

# find nodes on edges of the beam, i.e. with (*, 0, 0) or (*, 0, 3) (*, 5, 0) or (*, 5, 3)
edge1 = np.where((ref_coords_adj[:, 1] == 0) & (ref_coords_adj[:, 2] == 0))[0]
edge2 = np.where(
    (ref_coords_adj[:, 1] == 0)
    & (np.abs(ref_coords_adj[:, 2] - ref_coords_adj[:, 2].max()) < 0.001)
)[0]
edge3 = np.where(
    (np.abs(ref_coords_adj[:, 1] - ref_coords_adj[:, 1].max()) < 0.001)
    & (np.abs(ref_coords_adj[:, 2] - ref_coords_adj[:, 2].max()) < 0.001)
)[0]
edge4 = np.where(
    (np.abs(ref_coords_adj[:, 1] - ref_coords_adj[:, 1].max()) < 0.001)
    & (ref_coords_adj[:, 2] == 0)
)[0]

elements1 = np.array([edge1[:-1], edge1[1::]])
elements2 = np.array([edge2[:-1], edge2[1::]])
elements3 = np.array([edge3[:-1], edge3[1::]])
elements4 = np.array([edge4[:-1], edge4[1::]])

elements = np.concatenate([elements1, elements2], axis=0)[:, :86]
elements = np.concatenate(
    [elements, np.concatenate([elements2, elements3], axis=0)[:, :86]], axis=1
)
elements = np.concatenate(
    [elements, np.concatenate([elements3, elements4], axis=0)[:, :86]], axis=1
)
elements = np.concatenate(
    [elements, np.concatenate([elements4, elements1], axis=0)[:, :86]], axis=1
)

faces = []
for element in elements.T:
    # faces.append(element[[2, 0, 1]])
    # faces.append(element[[3, 2, 1]])
    faces.append(element[[1, 0, 2]])
    faces.append(element[[1, 2, 3]])
faces = np.unique(np.array(faces), axis=0)

n_sims = 2
shape = (n_sims, n_t, ref_coords.shape[0], ref_coords.shape[1])
X_ref = data["X"][:, :-1].reshape(shape)
X_pred = data["X_pred_mean"].reshape(shape)
X_uq = (data["X_pred_ub"] - data["X_pred_lb"]).reshape(shape)
error = np.linalg.norm(X_ref - X_pred, axis=3)
error2 = np.abs(X_ref - X_pred)
uq = np.linalg.norm(X_uq, axis=3) / 2  # half of the confidence interval (ub - lb) / 2

traj = 0
# 3d plot of the beam
disp_ampl = 50
# component_manipulation
comp_scaling = np.array([0.5, 1, 1])  # [0.2, 1, 1]
coords_ref = (disp_ampl * X_ref[traj, :10000:10] + ref_coords) * comp_scaling
coords_pred = (disp_ampl * X_pred[traj, :10000:10] + ref_coords) * comp_scaling
plt.plot(coords_ref[:, 1000, 1])
plt.plot(coords_pred[:, 1000, 1])

n_t = coords_ref.shape[0]
figdir = os.path.join(outdir, "figures")
if not os.path.exists(figdir):
    os.makedirs(figdir)

visualizer = Visualizer(background_color=[0, 0, 0, 0])

visualizer.animate(
    [coords_ref],
    range(n_t),
    color=["gray"],
    colormap="viridis",
    faces=faces,
    view=[-90, 90],
    animation_name=f"{figdir}/ref",
    save_animation=True,
    save_single_frames=False,
    camera_distance=200,
    close_on_end=True,
)
visualizer.animate(
    [coords_pred],
    range(n_t),
    color=[error[traj, :10000:10]],
    colormap="viridis",
    color_scale_limits=[error.min(), error.max()],
    faces=faces,
    view=[-90, 90],
    animation_name=f"{figdir}/error",
    save_animation=True,
    save_single_frames=False,
    camera_distance=200,
)
visualizer.animate(
    [coords_pred],
    range(n_t),
    color=[uq[traj, :10000:10]],
    colormap="plasma",
    color_scale_limits=[uq.min(), uq.max()],
    faces=faces,
    view=[-90, 90],
    animation_name=f"{figdir}/uq",
    save_animation=True,
    save_single_frames=False,
    camera_distance=200,
)


quantities = ["X"]  # z latent space, x pca space, X original space
X_dofs = [789, 790, 791]
for quantity in quantities:

    z_ref = data[f"{quantity}"][:, :-1, X_dofs]
    z_mean = data[f"{quantity}_pred_mean"][:, :, X_dofs]
    z_ub = data[f"{quantity}_pred_ub"][:, :, X_dofs]
    z_lb = data[f"{quantity}_pred_lb"][:, :, X_dofs]

    n_traj = z_mean.shape[0]
    n_states = z_mean.shape[2]

    header = ["t"]
    header += [
        f"{quantity}_{i}traj_{j}_"
        for j in range(1, n_traj + 1)
        for i in range(1, n_states + 1)
    ]
    header += [
        f"{quantity}_{i}_traj_{j}_mean"
        for j in range(1, n_traj + 1)
        for i in range(1, n_states + 1)
    ]
    header += [
        f"{quantity}_{i}_traj_{j}_lb"
        for j in range(1, n_traj + 1)
        for i in range(1, n_states + 1)
    ]
    header += [
        f"{quantity}_{i}_traj_{j}_ub"
        for j in range(1, n_traj + 1)
        for i in range(1, n_states + 1)
    ]
    # save a csv which maps the scenario info to index
    np.savetxt(
        os.path.join(outdir, f"{model_name}_{quantity}_traj.csv"),
        np.concatenate(
            [
                t[np.newaxis][:, ::n_th_timestep],
                z_ref.reshape([-1, n_t])[:, ::n_th_timestep],
                z_mean.reshape([-1, n_t])[:, ::n_th_timestep],
                z_lb.reshape([-1, n_t])[:, ::n_th_timestep],
                z_ub.reshape([-1, n_t])[:, ::n_th_timestep],
            ]
        ).T,
        delimiter=",",
        fmt="%s",
        comments="",
        header=",".join(header),
    )
