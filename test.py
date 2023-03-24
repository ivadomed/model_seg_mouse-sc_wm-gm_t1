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
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference


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
    out_channels=3,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.3,
)

# TODO: get these as input arg
# Load the trained 2D U-Net model
model_state = torch.load("/Users/julien/Desktop/best_metric_model_v34.pth", map_location=torch.device('cpu'))
model.load_state_dict(model_state)
model.eval()

# TODO: get these as input arg
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
# TODO: update num_workers
dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

# Apply the model to each slice
segmented_slices = []
activations = Activations(softmax=True)
as_discrete = AsDiscrete(argmax=True)
post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

with torch.no_grad():
    for data in tqdm(dataloader, desc="Processing images", unit="image"):
        image = data["image"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # TODO: parametrize values below
        roi_size = (192, 192)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(image, roi_size, sw_batch_size, model)
        # TODO: consider optimizing prediction function below, because:
        #  Processing images:  85%|████████████████████████████████▉      | 423/500 [00:12<00:02, 33.63image/s]
        #  vs. ~45image/s with using output = model(image)
        #  See old code at a4637c05588511f0ca9da2f298b1effeed8fe880
        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
        if isinstance(val_outputs, list):
            val_outputs = torch.stack(val_outputs)
        segmented_slices.append(val_outputs.numpy().squeeze())

# Stack the segmented slices to create the segmented 3D volume
segmented_volume = np.stack(segmented_slices, axis=-1)
segmented_volume = segmented_volume.astype(np.uint8)

# Save the segmented volume as a NIFTI file
segmented_nifti = nib.Nifti1Image(segmented_volume, nifti_volume.affine)
nib.save(segmented_nifti, "prediction.nii.gz")
