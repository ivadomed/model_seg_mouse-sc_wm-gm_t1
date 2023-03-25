"""
Spinal cord white and gray matter segmentation (2 classes) using 2D kernel based on MONAI, with WandB monitoring.
It assumes the file "best_metric_model.pth" is locally present.
To launch:

    python segment.py -i IMAGE

"""

import argparse
import numpy as np
import nibabel as nib
import os
import torch
from tqdm import tqdm

from monai.transforms import Activations, Compose, EnsureChannelFirstd, AsDiscrete, CastToTyped, \
    ScaleIntensityRangePercentilesd
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference


def add_suffix_to_filename(filename, suffix):
    name, ext = os.path.splitext(filename)

    # Check if there is a double extension
    if ext.lower() in {'.gz', '.bz2', '.xz'}:
        name, ext2 = os.path.splitext(name)
        ext = ext2 + ext

    output_filename = f"{name}{suffix}{ext}"
    return output_filename


def main():
    # Get CLI argument
    parser = argparse.ArgumentParser(description="Segment spinal cord white and gray matter. The function outputs a "
                                                 "single NIfTI file with the 2 classes (WM: 1, GM: 2). The input file "
                                                 "needs to be oriented with the axial slice as the last (3rd) "
                                                 "dimension. The function assumes that the model state "
                                                 "'best_metric_model.pth' is present in the local directory")
    parser.add_argument("-i", "--input", type=str, required=True, help="NIfTI file to process.")
    args = parser.parse_args()
    fname_in = args.input

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

    # Load the trained 2D U-Net model
    model_state = torch.load("best_metric_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.eval()

    # Load the 3D NIFTI volume
    nifti_volume = nib.load(fname_in)
    volume = nifti_volume.get_fdata()

    # Create a list of dictionaries with the 2D slices
    data_list = [{"image": np.expand_dims(volume[..., i], axis=0)} for i in range(volume.shape[-1])]

    # Prepare the keys for the dictionary used by MONAI transforms
    keys = ["image"]

    # Define the transforms to apply to the slices
    transforms = Compose(
        [
            EnsureChannelFirstd(keys, channel_dim=0),
            ScaleIntensityRangePercentilesd(keys, lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True, relative=False),
            CastToTyped(keys, dtype=torch.float32),
        ]
    )

    # Create the dataset and dataloader with the slices and transforms
    dataset = Dataset(data_list, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # Apply the model to each slice
    segmented_slices = []
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=3)])

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Process image", unit="image"):
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
            segmented_slice = np.uint8(torch.argmax(val_outputs, dim=1).cpu().numpy().squeeze())
            # print(segmented_slice.max())
            segmented_slices.append(segmented_slice)

    # Stack the segmented slices to create the segmented 3D volume
    segmented_volume = np.stack(segmented_slices, axis=-1)
    segmented_volume = segmented_volume.astype(np.uint8)

    # Save the segmented volume as a NIFTI file
    fname_out = add_suffix_to_filename(fname_in, '_seg')
    segmented_nifti = nib.Nifti1Image(segmented_volume, nifti_volume.affine)
    nib.save(segmented_nifti, fname_out)
    print(f"Done! Output file: {fname_out}")


if __name__ == "__main__":
    main()


# DEBUGGING CODE
# import matplotlib.pyplot as plt
# plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image[0, 0, :, :].squeeze(), cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("prediction")
# plt.imshow(val_outputs[0, 1, :, :].squeeze(), cmap="hot")
