import torch
from monai.networks.nets import UNet
import os


# Select device
device = torch.device("cpu")


# Create the 3D U-Net architecture
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)


# Load trained checkpoint
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_unet.pth"
)

checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("3D U-Net model loaded successfully!")