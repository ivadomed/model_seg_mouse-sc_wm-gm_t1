"""
Spinal cord white and gray matter segmentation (2 classes) using 2D kernel based on MONAI, with WandB monitoring.
It assumes the file(s) "best_metric_model*.pth" is locally present.
To launch:

    python segment.py -i IMAGE

"""

import argparse
import numpy as np
import nibabel as nib
import os
from scipy.ndimage import gaussian_filter1d
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


def apply_gaussian_smoothing_filter(arr, dim, sigma=1):
    """
    Apply a Gaussian smoothing filter to a 3D numpy array along the specified dimension.

    :param arr: A 3D numpy array.
    :param dim: The dimension along which to apply the smoothing filter (0, 1, or 2).
    :param sigma: The standard deviation of the Gaussian kernel (default is 1).
    :return: The smoothed 3D numpy array.
    """

    if dim not in (0, 1, 2):
        raise ValueError("Invalid dimension. Dimension should be 0, 1, or 2.")

    # Move the axis specified by 'dim' to the last position
    arr = np.moveaxis(arr, dim, -1)

    # Apply the Gaussian smoothing filter
    smoothed_arr = np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma=sigma), -1, arr)

    # Move the axis back to its original position
    smoothed_arr = np.moveaxis(smoothed_arr, -1, dim)

    return smoothed_arr


def apply_smoothing_filter(arr, dim, window_size=3):
    """
    Apply a moving average smoothing filter to a 3D numpy array along the specified dimension.

    :param arr: A 3D numpy array.
    :param dim: The dimension along which to apply the smoothing filter (0, 1, or 2).
    :param window_size: The size of the moving average window (default is 3).
    :return: The smoothed 3D numpy array.
    """

    if dim not in (0, 1, 2):
        raise ValueError("Invalid dimension. Dimension should be 0, 1, or 2.")

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("Invalid window size. Window size should be a positive odd integer.")

    # Create the moving average kernel
    kernel = np.ones(window_size) / window_size

    # Move the axis specified by 'dim' to the last position
    arr = np.moveaxis(arr, dim, -1)

    # Pad the array to handle boundaries
    pad_size = window_size // 2
    padded_arr = np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(pad_size, pad_size)], mode='reflect')

    # Apply the smoothing filter
    smoothed_arr = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), -1, padded_arr)

    # Move the axis back to its original position
    smoothed_arr = np.moveaxis(smoothed_arr, -1, dim)

    return smoothed_arr


def majority_voting(segmented_slice_ensemble_all):
    """
    Apply majority voting to the ensemble of segmented slices
    :param segmented_slice_ensemble_all:
    :return:
    """
    unique_elements, counts = np.unique(segmented_slice_ensemble_all, return_counts=True, axis=0)
    max_index = np.argmax(counts)
    return unique_elements[max_index]


def main():
    # Get CLI argument
    parser = argparse.ArgumentParser(description="Segment spinal cord white and gray matter. The function outputs a "
                                                 "single NIfTI file with the 2 classes (WM: 1, GM: 2). The input file "
                                                 "needs to be oriented with the axial slice as the last (3rd) "
                                                 "dimension. The function assumes that the model state "
                                                 "'best_metric_model.pth' is present in the local directory")
    parser.add_argument("-i", "--input", type=str, required=True, help="NIfTI file to process.")
    # add possibility to specify model state file(s). Multiple files can be listed.
    parser.add_argument("-m", "--model", type=str, required=False, nargs='+',
                        help="Model state file(s) to use. Multiple files can be listed (separate with space). If not "
                             "specified, the function will use the file(s) 'best_metric_model*.pth' in the local "
                             "directory.")
    # add parameter to specify sigma for Gaussian smoothing filter
    parser.add_argument("-s", "--sigma", type=float, required=False, default=2,
                        help="Standard deviation of the Gaussian kernel for smoothing the input volume (default is 2)."
                             "Try higher values (e.g. 2.5, 3) if the segmentation is too coarse, or lower values (1.5, "
                             "1) if the segmentation is inaccurate, which could occur if adjacent slices look very "
                             "different due to high curvature.")
    # add optional argument in case user wants to save smoothed volume (write with default filename)
    parser.add_argument("-o", "--output-smooth", required=False, default=None, action='store_true',
                        help="Output the smoothed volume (for debugging purpose).")

    args = parser.parse_args()
    fname_in = args.input

    # Load the 3D NIFTI volume
    nifti_volume = nib.load(fname_in)
    volume = nifti_volume.get_fdata()

    # Check that the input volume is oriented with the axial slice as the last (3rd) dimension
    if volume.shape[2] < volume.shape[0] or volume.shape[2] < volume.shape[1]:
        raise ValueError("The input volume is not oriented with the axial slice as the last (3rd) dimension.")

    # Check that the input volume has the correct number of dimensions
    if volume.ndim != 3:
        raise ValueError("The input volume does not have the correct number of dimensions (3).")

    # Apply a smoothing filter to the volume
    print(f"Smoothing the input volume along Z with sigma: {args.sigma}...")
    volume = apply_gaussian_smoothing_filter(volume, dim=2, sigma=args.sigma)
    # Save volume as NIfTI for visualization
    nifti_volume_smoothed = nib.Nifti1Image(volume, nifti_volume.affine, nifti_volume.header)
    if args.output_smooth:
        fname_smooth = add_suffix_to_filename(fname_in, "_smoothed")
        print(f"Saving smoothed volume to: {fname_smooth}")
        nib.save(nifti_volume_smoothed, fname_smooth)

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
    # TODO: try with different num_workers
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model:
        path_models = args.model
    else:
        # Fetch existing models in the current directory. The models are assumed to be named "best_metric_model*.pth"
        path_models = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith('best_metric_model')]
    # Load the trained 2D U-Net models
    models = []
    for path_model in path_models:
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
        model_state = torch.load(path_model, map_location=device)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        models.append(model)
    print(f"Using models: {path_models}")
    # TODO: make sure that all model states have different parameters (to avoid the situation where the same model has
    #  been loaded twice)

    # Define the post-processing transforms to apply to the model output
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=3)])

    with torch.no_grad():
        segmented_slices = []
        for data in tqdm(dataloader, desc=f"Segment image", unit="image"):
            image = data["image"].to(device)
            # TODO: parametrize values below
            roi_size = (192, 192)
            sw_batch_size = 4
            overlap = 0.25
            segmented_slice_ensemble_all = []
            for model in models:
                # Apply the model to each slice
                val_outputs = sliding_window_inference(image, roi_size, sw_batch_size, model, overlap)
                # TODO: consider optimizing prediction function below, because:
                #  Processing images:  85%|████████████████████████████████▉      | 423/500 [00:12<00:02, 33.63image/s]
                #  vs. ~45image/s with using output = model(image)
                #  See old code at a4637c05588511f0ca9da2f298b1effeed8fe880
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                if isinstance(val_outputs, list):
                    val_outputs = torch.stack(val_outputs)
                segmented_slice = np.uint8(torch.argmax(val_outputs, dim=1).cpu().numpy().squeeze())
                segmented_slice_ensemble_all.append(segmented_slice)

            # Do not aggregate the predictions from the different models if only one model is used
            if len(models) == 1:
                segmented_slice_ensemble_all_aggregated = segmented_slice_ensemble_all[0]
            else:
                # Aggregate the predictions from the different models using majority voting
                segmented_slice_ensemble_all_aggregated = majority_voting(np.array(segmented_slice_ensemble_all))
                # Alternative approach using mean:
                #  segmented_slice_ensemble = np.mean(segmented_slice_ensemble_all, axis=0)
            # Add the segmented slice to the list of segmented slices
            segmented_slices.append(segmented_slice_ensemble_all_aggregated)

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


# # DEBUGGING CODE
# import matplotlib.pyplot as plt
# plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image[0, 0, :, :].squeeze(), cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("prediction")
# plt.imshow(val_outputs[0, 1, :, :].squeeze(), cmap="hot")
