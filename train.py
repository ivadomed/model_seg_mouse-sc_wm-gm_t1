"""
Spinal cord white and gray matter segmentation (2 classes) using 2D kernel based on MONAI, with WandB monitoring.

To launch:

    export CUDA_VISIBLE_DEVICES="0"; export WANDB_RUN_GROUP="GROUP_NAME"; python train.py

"""

import os
import glob
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import wandb

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from monai.utils import first, set_determinism
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    Rand2DElasticd,
    RandAffined,
    RandFlipd,
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

# Let's print configuration of some packages by using a utility function provided by MONAI as `print_config()` which
# basically lists down all the versions of the useful libraries.
print_config()


def split_indices(data: List, ratio: float) -> Tuple[List[int], List[int]]:
    """
    Split the list indices into two sets based on the given ratio.

    :param data: The input list.
    :param ratio: The ratio of the number of indices in the first set to the number of indices in the second set.
    :return: A tuple of two lists of indices, where the first list contains indices for the first set,
             and the second list contains indices for the second set.
    """
    num_items = len(data)
    indices = list(range(num_items))

    first_set_size = int(num_items * ratio / (ratio + 1))
    second_set_size = num_items - first_set_size

    first_set_indices = indices[:first_set_size]
    second_set_indices = indices[first_set_size:first_set_size + second_set_size]

    return first_set_indices, second_set_indices


def interleave_indices(data: List, ratio: float) -> Tuple[List[int], List[int]]:
    """
    Interleave the list indices into two sets based on the given ratio.

    :param data: The input list.
    :param ratio: The ratio of the number of indices in the first set to the number of indices in the second set.
    :return: A tuple of two lists of indices, where the first list contains indices for the first set,
             and the second list contains indices for the second set.
    """
    num_items = len(data)
    indices = list(range(num_items))

    first_set_indices = []
    second_set_indices = []

    count_ratio = 0
    current_set = 1

    for index in indices:
        if current_set == 1:
            first_set_indices.append(index)
            count_ratio += 1
            if count_ratio >= ratio:
                current_set = 2
                count_ratio = 0
        else:
            second_set_indices.append(index)
            count_ratio += 1
            if count_ratio >= 1:
                current_set = 1
                count_ratio = 0

    return first_set_indices, second_set_indices


def match_images_and_labels(images, labels_WM, labels_GM):
    """
    Assumes BIDS format.
    :param images:
    :param labels:
    :return:
    """
    images_match = []
    labels_match = []
    print("Matched image and labels:")
    # Loop across images
    for image in images:
        # Fetch file name without extension
        filename = image.split(os.path.sep)[-1].split('.')[0]
        # Find equivalent in labels
        label_WM = [j for i, j in enumerate(labels_WM) if filename in j]
        label_GM = [j for i, j in enumerate(labels_GM) if filename in j]
        if label_WM and label_GM:
            images_match.append(image)
            labels_match.append([label_WM[0], label_GM[0]])
            print(f"- {image}")
    return images_match, labels_match


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
    "num_workers": 2,  # TODO: Set to 0 to debug in Pycharm (avoid multiproc).
    "split_train_val_ratio": 3,

    # data augmentation (probability of occurrence)
    "RandFlip": 0.5,
    "RandAffine": 0.5,
    "Rand2DElastic": 0.3,

    # train settings
    "train_batch_size": 32,
    "val_batch_size": 32,
    "learning_rate": 1e-3,
    "max_epochs": 200,
    "val_interval": 10,  # check validation score after n epochs
    "lr_scheduler": "cosine_decay",  # just to keep track

    # Unet model (you can even use nested dictionary and this will be handled by W&B automatically)
    "model_type": "unet",  # just to keep track
    "model_params": dict(spatial_dims=2,
                         in_channels=1,
                         out_channels=3,
                         channels=(8, 16, 32, 64),  #UNet
                         strides=(2, 2, 2),  # UNet
                         num_res_units=2,  # UNet
                         norm=Norm.BATCH,  # UNet
                         dropout=0.3,  # UNet
                         # img_size=(192, 192),  # UNETR
                         # feature_size=16,  # UNETR
                         # norm_name='batch',  # UNETR
                         # dropout_rate=0.3,  # UNETR
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
train_images = sorted(glob.glob(os.path.join(data_dir, "**", "*_T1w.nii.gz"), recursive=True))
train_labels_WM = sorted(glob.glob(os.path.join(data_dir, "derivatives", "**", "*_label-WM_mask.nii.gz"), recursive=True))
train_labels_GM = sorted(glob.glob(os.path.join(data_dir, "derivatives", "**", "*_label-GM_mask.nii.gz"), recursive=True))
train_labels = train_labels_WM + train_labels_GM
train_images_match, train_labels_match = match_images_and_labels(train_images, train_labels_WM, train_labels_GM)
data_dicts = [{"image": image_name, "label_WM": label_name[0], "label_GM": label_name[1]}
              for image_name, label_name in zip(train_images_match, train_labels_match)]

# Check if variable data_dicts is empty
if not data_dicts:
    raise ValueError("No data found. Please check your data directory.")

# Iterate across image/label 3D volume, fetch non-empty slice and output a single list of image/label pair
patch_data = []
for data_dict in tqdm(data_dicts, desc="Load images", unit="image"):
    nii_image = load(data_dict['image'])
    nii_label_WM = load(data_dict['label_WM'])
    nii_label_GM = load(data_dict['label_GM'])
    for i_z in range(nii_label_WM.shape[2]):
        label_z_WM = nii_label_WM.get_fdata()[:, :, i_z]
        # If WM label is not empty, consider this as a pair of image/label for training
        if label_z_WM.sum() > 0:
            image_z = nii_image.get_fdata()[:, :, i_z]
            label_z_GM = nii_label_GM.get_fdata()[:, :, i_z] * 2
            label_z = label_z_WM + label_z_GM
            patch_data.append({
                'image': np.expand_dims(image_z, axis=0),  # Expand because torch expects channel at dim=0
                'label': np.expand_dims(label_z, axis=0),
                'file': os.path.basename(data_dict['image']),
                'slice': i_z
            })

# TODO: optimize hyperparam:
#  RandAffined
train_transforms = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim=0),
        # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys="image", offsets=0.2, prob=0.5),
        # RandHistogramShiftd(keys=["image"], num_control_points=10, prob=1.0),
        # RandBiasFieldd(keys=["image"], degree=3, coeff_range=(0.0, 0.1)),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True,
                                        relative=False),
        # ScaleIntensityd(keys=["image"]),
        RandFlipd(keys=["image", "label"], prob=config['RandFlip'], spatial_axis=1),
        RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), spatial_size=(192, 192),
                    translate_range=(20, 20), rotate_range=np.pi/2, scale_range=(0.2, 0.2), prob=config['RandAffine']),
        Rand2DElasticd(keys=["image", "label"], spacing=(30, 30), magnitude_range=(3, 3), prob=config['Rand2DElastic']),
        ToTensor(dtype=np.dtype('float32')),
    ]
)

#val_transforms = Compose(
#    [
#        AddChanneld(keys=["image", "label"]),
        # ScaleIntensityd(keys=["image"]),
#        ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True,
#                                        relative=False),
#        ToTensor(dtype=np.dtype('float32')),
#    ]
#)
val_transforms = train_transforms

# Split train/validation datasets
# TODO: consider using random split with split_indices() for final Ensemble model.
train_id, val_id = interleave_indices(patch_data, config['split_train_val_ratio'])
print("Train indices:", train_id)
print("Validation indices:", val_id)
train_ds = PatchDataset(
    data=[patch_data[i] for i in train_id], patch_func=patch_func, samples_per_image=1, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1)
val_ds = PatchDataset(
    data=[patch_data[i] for i in val_id], patch_func=patch_func, samples_per_image=1, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

# Create Model, Loss, Optimizer and Scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO: use logging
if device.type == 'cuda':
    print(f"device: {device}:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")
else:
    print(f"device: {device}")
model = UNet(**config['model_params']).to(device)
# TODO: optimize params: https://docs.monai.io/en/stable/losses.html#diceloss
# loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
dice_metric = DiceMetric(include_background=False, reduction="mean")
scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=1e-9)

# To avoid https://github.com/jcohenadad/model-seg-ms-mp2rage-monai/issues/1
torch.multiprocessing.set_sharing_strategy('file_system')

# 🐝 initialize a wandb run
wandb.init(project="mouse-zurich", config=config)

# 🐝 log gradients of the model to wandb
wandb.watch(model, log_freq=100)

max_epochs = config['max_epochs']
val_interval = config['val_interval']
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
# post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=3)])
post_label = Compose([AsDiscrete(to_onehot=3)])
# post_label = Compose()
wandb_mask_logs = []
wandb_img_logs = []

# 🐝 add this training script as an artifact
artifact_script = wandb.Artifact(name='script', type='file')
artifact_script.add_file(local_path=os.path.abspath(__file__), name=os.path.basename(__file__))
wandb.log_artifact(artifact_script)

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
        
        # 🐝 log train_loss for each step to wandb
        wandb.log({"Training/loss": loss.item()})
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    # step scheduler after each epoch (cosine decay)
    scheduler.step()
    
    # 🐝 log train_loss averaged over epoch to wandb
    wandb.log({"Training/loss_epoch": epoch_loss})
    
    # 🐝 log learning rate after each epoch to wandb
    wandb.log({"Training/learning_rate": scheduler.get_last_lr()[0]})

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                # TODO: parametrize this
                roi_size = (192, 192)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)

                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

                # 🐝 show image with ground truth and prediction on eval dataset
                # Convert val_outputs and val_labels to a tensor if it's a list
                if isinstance(val_outputs, list):
                    val_outputs = torch.stack(val_outputs)
                if isinstance(val_labels, list):
                    val_labels = torch.stack(val_labels)
                # TODO: use hot palette for labels
                wandb.log({"Validation_Image/Image":
                               wandb.Image(val_inputs.cpu().numpy().squeeze(),
                                           caption=f"{val_data['file'][0]} (z={int(val_data['slice'])})")})
                val_labels_scaled = np.uint8(torch.argmax(val_labels, dim=1).cpu().numpy().squeeze() * 255 // 2)
                wandb.log({"Validation_Image/Ground truth": wandb.Image(val_labels_scaled)})
                val_outputs_scaled = np.uint8(torch.argmax(val_outputs, dim=1).cpu().numpy().squeeze() * 255 // 2)
                wandb.log({"Validation_Image/Prediction": wandb.Image(val_outputs_scaled)})

            # 🐝 aggregate the final mean dice result
            metric = dice_metric.aggregate().item()

            # 🐝 log validation dice score for each validation round
            wandb.log({"Validation/dice_metric": metric})

            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
print(
    f"\ntrain completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")

# 🐝 log best score and epoch number to wandb
wandb.log({"best_dice_metric": best_metric, "best_metric_epoch": best_metric_epoch})

# 🐝 Version your model
best_model_path = os.path.join(root_dir, "best_metric_model.pth")
model_artifact = wandb.Artifact(
            "unet", type="model",
            description="Unet for 3D Segmentation of MS lesions",
            metadata=dict(config['model_params']))
model_artifact.add_file(best_model_path)
wandb.log_artifact(model_artifact)

"""# Check best model output with the input image and label"""

model.load_state_dict(torch.load(
    os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
# with torch.no_grad():
#     for i, val_data in enumerate(val_loader):
#         roi_size = (200, 200)
#         sw_batch_size = 4
#         val_outputs = sliding_window_inference(
#             val_data["image"].to(device), roi_size, sw_batch_size, model
#         )
#         # plot the slice [:, :, 80]
#         plt.figure("check", (18, 6))
#         plt.subplot(1, 3, 1)
#         plt.title(f"image {i}")
#         plt.imshow(val_data["image"][0, 0, :, :], cmap="gray")
#         plt.subplot(1, 3, 2)
#         plt.title(f"label {i}")
#         plt.imshow(val_data["label"][0, 0, :, :])
#         plt.subplot(1, 3, 3)
#         plt.title(f"output {i}")
#         plt.imshow(torch.argmax(
#             val_outputs, dim=1).detach().cpu()[0, 0, :])
#         plt.show()
#         if i == 2:
#             break

"""# Log predictions to W&B in form of table"""

# # 🐝 create a wandb table to log input image, ground_truth masks and predictions
# columns = ["filename", "image", "ground_truth", "prediction"]
# table = wandb.Table(columns=columns)
#
# model.load_state_dict(torch.load(
#     os.path.join(root_dir, "best_metric_model.pth")))
# model.eval()
# with torch.no_grad():
#     for i, val_data in enumerate(val_loader):
#         # get the filename of the current image
#         fn = val_data['image_meta_dict']['filename_or_obj'][0].split("/")[-1].split(".")[0]
#
#         roi_size = (200, 200)
#         sw_batch_size = 4
#         val_outputs = sliding_window_inference(
#             val_data["image"].to(device), roi_size, sw_batch_size, model
#         )
#
#         # log each 2D image
#         img = val_data["image"][0, 0, :, :]
#         label = val_data["label"][0, 0, :, :]
#         prediction = torch.argmax(
#             val_outputs, dim=1).detach().cpu()[0, :, :]
#
#         # 🐝 Add data to wandb table dynamically
#         table.add_data(fn, wandb.Image(img), wandb.Image(label), wandb.Image(prediction))
#
# # log predictions table to wandb with `val_predictions` as key
# wandb.log({"val_predictions": table})

# 🐝 Close your wandb run
wandb.finish()


# DEBUGGING CODE
# ==============
# Plot slice
# image, label, prediction = (val_inputs[0][0], val_labels[0][0], val_outputs[0][0])
# plt.figure("check", (18, 6))
# plt.subplot(1, 3, 1)
# plt.title("image")
# plt.imshow(image[:, :], cmap="gray")
# plt.subplot(1, 3, 2)
# plt.title("label")
# plt.imshow(label[:, :])
# plt.subplot(1, 3, 3)
# plt.title("prediction")
# plt.imshow(prediction[:, :])
# plt.show()
