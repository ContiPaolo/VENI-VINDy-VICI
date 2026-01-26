import numpy as np
from reaction_diffusion.utils import switch_data_format


def run_tests():
    n_sims = 4
    n_timesteps = 10
    Nx = 5
    Ny = 5
    channels = 2

    # create full 5D data
    data5 = np.arange(n_sims * n_timesteps * Nx * Ny * channels).reshape(
        n_sims, n_timesteps, Nx, Ny, channels
    )

    # 5D -> 3D
    data3 = switch_data_format(data5, n_sims, n_timesteps, target_format="3d")
    assert data3.shape == (n_sims, n_timesteps, Nx * Ny * channels)
    # check values
    assert data3.reshape(n_sims, n_timesteps, Nx, Ny, channels).shape == data5.shape

    # 5D -> 2D
    data2 = switch_data_format(data5, n_sims, n_timesteps, target_format="2d")
    assert data2.shape == (n_sims * n_timesteps, Nx * Ny * channels)

    # 2D -> 3D (auto)
    data3_from2 = switch_data_format(data2, n_sims, n_timesteps)
    assert data3_from2.shape == (n_sims, n_timesteps, Nx * Ny * channels)

    # 2D -> 5D (need spatial_shape)
    data5_from2 = switch_data_format(
        data2, n_sims, n_timesteps, spatial_shape=(Nx, Ny, channels), target_format="5d"
    )
    assert data5_from2.shape == data5.shape
    assert np.array_equal(data5_from2, data5)

    # 3D -> 5D (with spatial_shape inference)
    data5_from3 = switch_data_format(
        data3, n_sims, n_timesteps, spatial_shape=(Nx, Ny, channels), target_format="5d"
    )
    assert data5_from3.shape == data5.shape
    assert np.array_equal(data5_from3, data5)

    # 3D -> 2D
    data2_from3 = switch_data_format(data3, n_sims, n_timesteps, target_format="2d")
    assert data2_from3.shape == data2.shape

    print("All switch_data_format tests passed")


if __name__ == "__main__":
    run_tests()
