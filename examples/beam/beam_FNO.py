#%%
import torch
from torch.utils.data import DataLoader, TensorDataset
from neuralop.models import FNO1d
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

load_model = True
train = False
save = False
#%%

# ------------------------------
# 1. Dataset
# ------------------------------

# Your data: x = (n_traj, n_x, n_t), b = (n_traj, 1, n_t)
# Convert forcing to match x's shape: repeat along n_x if needed
def prepare_dataset(x, b, n_in=4, n_out=4):
    """
    Prepare autoregressive FNO dataset with multiple input/output timesteps
    n_in: number of input timesteps
    n_out: number of output timesteps
    """
    n_traj, n_x, n_t = x.shape
    X_list, Y_list = [], []

    for traj in range(n_traj):
        for t0 in range(n_t - n_in - n_out + 1):
            x_in = x[traj, :, t0:t0+n_in]             # (n_x, n_in)
            b_in = b[traj, :, t0:t0+n_in]             # (1, n_in)

            # Expand b_in to match x_in along spatial dimension
            b_expanded = b_in.repeat(n_x, 1)          # (n_x, n_in)

            # Concatenate forcing as extra channel
            inp = torch.cat([x_in, b_expanded], dim=1)  # (n_x, n_in + 1)
            x_out = x[traj, :, t0+n_in:t0+n_in+n_out]   # (n_x, n_out)

            X_list.append(inp)
            Y_list.append(x_out)

    X = torch.stack(X_list)  # (n_samples, n_x, n_in+1)
    Y = torch.stack(Y_list)  # (n_samples, n_x, n_out)
    return X, Y

#%% Load data
from utils import load_beam_data
import sys
# Add the examples folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import config
import config

model_name = "beam"
identification_layer = "vindy"  # 'vindy' or 'sindy'
# Script parameter
reduced_order = 1
pca_order = 3
nth_time_step = 4

end_time_step = 14000

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
) = load_beam_data(
    config.beam["processed_data"],
    end_time_step=end_time_step,
    nth_time_step=nth_time_step,
    pca_order=pca_order,
)

#%%
#b = F*cos(wt)
b = params[:,2] * np.cos(params[:,1] * params[:,0])
b_test = params_test[:,2] * np.cos(params_test[:,1] * params_test[:,0])

#Reshape x = (n_traj*n_t, nx) to (n_traj, nx, n_t)
n_t = n_timesteps
n_t_test = 22500//nth_time_step
n_x = x.shape[1]
n_sims = int(x.shape[0]/n_t)
n_sims_test = int(x_test.shape[0]/n_t_test)

x_ = x.reshape(n_sims, n_t, n_x).transpose(0,2,1)  # (n_traj, n_x, n_t)
b_ = b.reshape(n_sims, n_t)[:, np.newaxis, :]

x_test_ = x_test.reshape(n_sims_test, n_t_test, n_x).transpose(0,2,1)  # (n_traj, n_x, n_t)
b_test_ = b_test.reshape(n_sims_test, n_t_test)[:, np.newaxis, :]

#x_test_ = x_test.reshape(-1, n_t, n_x).transpose(0,2,1)


n_traj = n_sims #!!
x_test_ = x_test_
b_test_ = b_test_

x_ = x_[:n_traj]
b_ = b_[:n_traj]

#%%
# Example: 5 input, 3 output timesteps
n_in, n_out = 5, 3
X, Y = prepare_dataset(torch.tensor(x_, dtype=torch.float32),
                       torch.tensor(b_, dtype=torch.float32),
                       n_in=n_in, n_out=n_out)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

#%%
# ------------------------------
# 2. FNO Model
# ------------------------------

# FNO1d: input channels = n_in + 1 (state + forcing), output channels = n_out
model = FNO1d(
    in_channels=n_in + n_in,
    out_channels=n_out,
    n_modes_height=64,      # or another value suitable for your data
    hidden_channels=16       # or another value suitable for your model size
)
#%%
# ------------------------------
# 3. Training
# ------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 100

# Load checkpoint
if load_model:
    checkpoint = torch.load("./results/fno_beam_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])


    # If you want to resume training:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch} with loss {checkpoint['loss']:.6f}")

if train:
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X.permute(0,2,1))  # FNO1d expects (batch, in_channels, n_x)
            pred = pred.permute(0,2,1)            # back to (batch, n_out, n_x)
            loss = criterion(pred, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")

if save:
    # Save model + optimizer state + training metadata
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": total_loss / len(loader),
    }, "./results/fno_beam_model.pt")

#%%
# ------------------------------
# 4. Autoregressive prediction on new trajectories
# ------------------------------

def autoregressive_predict(model, x0, b_seq, n_in=5, n_out=2, n_steps=50, device='cpu'):
    """
    Autoregressive rollout using FNO.

    x0: initial state window, shape (n_x, n_in)
    b_seq: forcing for the whole rollout, shape (n_steps,) or (n_steps, 1)
    n_in: number of input timesteps used by the model
    n_out: number of timesteps predicted per forward pass
    n_steps: total number of timesteps to predict
    """
    n_x = x0.shape[0]
    model.eval()
    preds = []

    x_curr = x0.clone()  # (n_x, n_in)

    for t in range(0, n_steps, n_out):
        # get corresponding forcing for the input window
        b_window = b_seq[t:t+n_in].unsqueeze(0).repeat(n_x, 1)  # (n_x, n_in)
        inp = torch.cat([x_curr, b_window], dim=1).unsqueeze(0) # (1, n_x, n_in+1)
        
        inp = inp.permute(0,2,1).to(device)                    # (1, in_channels, n_x)
        out = model(inp).permute(0,2,1).squeeze(0)             # (n_x, n_out)
        
        preds.append(out.cpu())
        # slide input window forward
        x_curr = torch.cat([x_curr[:, n_out:], out], dim=1)

    return torch.cat(preds, dim=1)  # (n_x, n_steps)



# %%
# Pick a test trajectory
traj_id = 0

x_test_traj = torch.tensor(x_test_[traj_id], dtype=torch.float32)  # (n_x, n_t_test)
b_test_traj = torch.tensor(b_test_[traj_id,0], dtype=torch.float32)  # (n_t_test,)

n_t_test = x_test_traj.shape[1]

# Initial window for autoregressive rollout
x0 = x_test_traj[:, :n_in]  # (n_x, n_in)

# Run autoregressive prediction for the full test trajectory
x_pred = autoregressive_predict(
    model, x0, b_test_traj, 
    n_in=n_in, n_out=n_out, 
    n_steps=n_t_test - n_in,  # predict the rest of the trajectory
    device=device
)

# Ground truth for comparison (skip first n_in since we used them as input)
x_true = x_test_traj[:, n_in:]

# Compute error
mse = torch.mean((x_pred[:,:-2] - x_true) ** 2)
print(f"Test trajectory MSE: {mse.item():.6f}")

# Plot comparison at a single spatial point (for visualization)
node_idx = 0  # choose a node to inspect
plt.figure(figsize=(10,4))
plt.plot(x_true[node_idx].numpy(), label='Ground Truth', lw=2)
plt.plot(x_pred[node_idx].detach().cpu().numpy(), label='FNO Prediction', lw=2, linestyle='--')
plt.title(f"POD mode {node_idx+1} - Autoregressive Prediction")
plt.xlabel("Time step")
#Add a verticla line to indicate where the training time windoes end
plt.axvline(x=n_t, color='red', linestyle=':', label='End of Input Window')
plt.ylabel("Displacement")
plt.legend()
plt.grid()
plt.show()

# %%
