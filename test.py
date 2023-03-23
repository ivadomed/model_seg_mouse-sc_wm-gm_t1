"""
Spinal cord white and gray matter segmentation (2 classes) using 2D kernel based on MONAI, with WandB monitoring.

To launch:

    python test.py

"""

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.transforms import Activations, Compose, AddChanneld, AsDiscrete, CastToTyped, ScaleIntensityRangePercentilesd
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.networks.layers import Norm


# Used for debugging
def visualize_slices(volume, axis=0):
    assert axis in (0, 1, 2), "Invalid axis, should be 0, 1, or 2."

    num_slices = volume.shape[axis]

    for i in range(num_slices):
        plt.figure()
        if axis == 0:
            plt.imshow(volume[i, :, :], cmap='gray')
        elif axis == 1:
            plt.imshow(volume[:, i, :], cmap='gray')
        else:
            plt.imshow(volume[:, :, i], cmap='gray')
        plt.title(f"Slice {i}")
        plt.axis('off')
        plt.show()


# Instantiate the 2D U-Net model with appropriate parameters
# You need to replace num_classes, channels, strides, and kernel_size with the values used in your trained model
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.3,
)
# Load the trained 2D U-Net model
model_state = torch.load("/Users/julien/Desktop/best_metric_model.pth", map_location=torch.device('cpu'))
model.load_state_dict(model_state)
model.eval()

# Load the 3D NIFTI volume
filename = "/Users/julien/data.neuro/zurich-mouse/sub-mouse1/anat/sub-mouse1_chunk-1_T1w.nii.gz"
nifti_volume = nib.load(filename)
volume = nifti_volume.get_fdata()

# Prepare the keys for the dictionary used by MONAI transforms
keys = ["image"]

# Create a list of dictionaries with the 2D slices
data_list = [{"image": volume[..., i]} for i in range(volume.shape[-1])]

# Define the transforms to apply to the slices
transforms = Compose(
    [
        AddChanneld(keys),
        ScaleIntensityRangePercentilesd(keys, lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True, relative=False),
        CastToTyped(keys, dtype=torch.float32),  # convert to float32 tensors
    ]
)

# Create the dataset and dataloader with the slices and transforms
dataset = Dataset(data_list, transform=transforms)
dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

# Apply the model to each slice
segmented_slices = []
activations = Activations(sigmoid=True)
as_discrete = AsDiscrete(threshold=0.5)

with torch.no_grad():
    for data in tqdm(dataloader, desc="Processing images", unit="image"):
        image = data["image"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        output = model(image)
        output_sigmoid = activations(output)
        output_discrete = as_discrete(output_sigmoid)
        output_slice = output_discrete.squeeze().cpu().numpy()
        segmented_slices.append(output_slice)

# Stack the segmented slices to create the segmented 3D volume
segmented_volume = np.stack(segmented_slices, axis=-1)
segmented_volume = segmented_volume.astype(np.uint8)

# Save the segmented volume as a NIFTI file
segmented_nifti = nib.Nifti1Image(segmented_volume, nifti_volume.affine)
nib.save(segmented_nifti, "prediction.nii.gz")


# For debugging
# plt.figure("debug", (12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(image[0, 0, :, :], cmap="gray")
# plt.title(f"image")
# plt.subplot(1, 2, 2)
# plt.imshow(output[0, 0, :, :], cmap="hot")
# plt.title(f"prediction")
