"""
Spinal cord white and gray matter segmentation (2 classes) using 2D kernel based on MONAI, with WandB monitoring.

To launch:

    python test.py

"""

import os
import glob
import numpy as np
import shutil
import tempfile

import wandb
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from monai.utils import first, set_determinism
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Rand2DElasticd,
    RandAffined,
    RandBiasFieldd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandHistogramShiftd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityd,
    ScaleIntensityRangePercentilesd,
    ToTensor,
)

from monai.networks.nets import UNet, UNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, decollate_batch, PatchDataset
from monai.config import print_config
from nibabel import load
from nibabel.nifti1 import Nifti1Image, save

# Let's print configuration of some packages by using a utility function provided by MONAI as `print_config()` which
# basically lists down all the versions of the useful libraries.
print_config()


def patch_func(dataset):
    """Dummy function to output a sequence of dataset of length 1"""
    return [dataset]


# Set default GPU number
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Training parameters
config = {
    # data
    "cache_rate": 1.0,
    "num_workers": 2,  # TODO: Set back to larger number. Set to 0 to debug in Pycharm (avoid multiproc).

    # data augmentation (probability of occurrence)
    "RandFlip": 0.5,
    "RandAffine": 0.5,
    "Rand2DElastic": 0.3,

    # train settings
    "train_batch_size": 32,  # TODO: Change back to 2
    "val_batch_size": 32,
    "learning_rate": 1e-3,
    "max_epochs": 500,
    "val_interval": 10,  # check validation score after n epochs
    "lr_scheduler": "cosine_decay",  # just to keep track

    # Unet model (you can even use nested dictionary and this will be handled by W&B automatically)
    "model_type": "unet",  # just to keep track
    "model_params": dict(spatial_dims=2,
                         in_channels=1,
                         out_channels=1,
                         # channels=(8, 16, 32, 64),  #UNet
                         # strides=(2, 2, 2),  # UNet
                         # num_res_units=2,  # UNet
                         # norm=Norm.BATCH,  # UNet
                         # dropout=0.3,  # UNet
                         img_size=(192, 192),  # UNETR
                         feature_size=16,  # UNETR
                         norm_name='batch',  # UNETR
                         dropout_rate=0.3,  # UNETR
                         )
}

# Setup data directory
# TODO: parametrize input data dir
default_data_dir = "/Users/julien/data.neuro/zurich-mouse"
env_data_dir = os.environ.get("PATH_DATA_ZURICH_MOUSE")
data_dir = default_data_dir if env_data_dir is None else env_data_dir
print(f"Path to data: {data_dir}")
# TODO: check dataset integrity
# data: data.neuro.polymtl.ca:/datasets/basel-mp2rage
# commit: ffe427d4d1f62832e5f3567c8ce814eeff9b9764

# Setup output directory
# TODO: parametrize
root_dir = "./"
# Set MSD dataset path
# TODO: replace with https://github.com/ivadomed/templates/blob/main/dataset_conversion/create_msd_json_from_bids.py
test_files = sorted(glob.glob(os.path.join(data_dir, "**", "*_T1w.nii.gz"), recursive=True))
# test_dict = [{"image": image_name} for image_name in zip(test_files)]
#
# n_samples = 5
# sampler = Compose([
#     LoadImaged(keys=["image"], image_only=True),
#     EnsureChannelFirstd(keys=["image"]),
#     # RandCropByPosNegLabeld(
#     #         keys=["image"],
#     #         image_key="image",
#     #         label_key="image",
#     #         spatial_size=(200, 200, 1),
#     #         pos=1,
#     #         neg=0,
#     #         num_samples=n_samples,
#     #     ),
#     CenterSpatialCropd(keys=["image"], roi_size=(192, 192, 1)),
# ])
#
# # Get this from train.py
# test_transforms = Compose(
#     [
#         AddChanneld(keys=["image"]),
#         ToTensor(dtype=np.dtype('float32')),
#     ]
# )
#
# ds = PatchDataset(data=test_dict, patch_func=sampler, samples_per_image=n_samples, transform=test_transforms)
# check_loader = DataLoader(ds, batch_size=1)
# check_data = first(check_loader)



# test_ds = PatchDataset(data=test_images, patch_func=patch_func, samples_per_image=1, transform=train_transforms)
# test_loader = DataLoader(test_ds, batch_size=4)

# Download model from wandb
run = wandb.init()
artifact = run.use_artifact('jcohenadad/mouse-zurich/unet:v25', type='model')
artifact_model = artifact.download()

# TODO: Get this from train.py
model = UNETR(**config['model_params']).to('cpu')
# TODO: deal with "_IncompatibleKeys" (had to set "strict=False" to avoid error). Something fishy there...
model.load_state_dict(
    torch.load(os.path.join(artifact_model, 'best_metric_model.pth'), map_location=torch.device('cpu')), strict=False)
model.eval()
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# Iterate across image/label 3D volume, fetch non-empty slice and output a single list of image/label pair
patch_data = []
for test_file in test_files:
    nii_image = load(test_file)
    data_pred = np.zeros(nii_image.shape, dtype='uint8')
    for i_z in range(nii_image.shape[2]):
        image_z = nii_image.get_fdata()[:, :, i_z]
        # TODO: run inference
        roi_size = (192, 192)  # TODO: parametrize
        sw_batch_size = 4
        # data_test = DataLoader(image_z)
        image_z_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(np.float32(image_z), axis=0), axis=0))
        test_outputs = sliding_window_inference(image_z_tensor, roi_size, sw_batch_size, model)
        # apply post-processing
        test_outputs_final = [post_pred(i) for i in decollate_batch(test_outputs)]
        # TODO: figure out if this decollate_batch is necessary
        test_outputs_final = test_outputs_final[0]
        data_pred[:, :, i_z] = np.squeeze(test_outputs_final.numpy())
    # TODO: update file name
    nii_prediction = Nifti1Image(data_pred, nii_image.affine)
    save(nii_prediction, 'pred.nii.gz')
    # TODO: figure out why prediction is all ones
