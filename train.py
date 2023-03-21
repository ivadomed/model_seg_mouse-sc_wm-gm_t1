"""
Spinal cord white and gray matter segmentation (2 classes) using 2D kernel based on MONAI, with WandB monitoring.

To launch:

    export CUDA_VISIBLE_DEVICES="0"; export WANDB_RUN_GROUP="GROUP_NAME"; python train.py

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
    Compose,
    Rand2DElasticd,
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandHistogramShiftd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityd,
    ScaleIntensityRangePercentilesd,
    ToTensor,
)

from monai.networks.nets import UNet
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


def match_images_and_labels(images, labels):
    """
    Assumes BIDS format.
    :param images:
    :param labels:
    :return:
    """
    images_match = []
    labels_match = []
    # Loop across images
    for image in images:
        # Fetch file name without extension
        filename = image.split(os.path.sep)[-1].split('.')[0]
        # Find equivalent in labels
        # TODO: check if label has 2 entries
        label = [j for i, j in enumerate(labels) if filename in j]
        if label:
            images_match.append(image)
            labels_match.append(label[0])
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
    "num_workers": 2,  # TODO: Set back to larger number. Set to 0 to debug in Pycharm (avoid multiproc).

    # train settings
    "train_batch_size": 32,  # TODO: Change back to 2
    "val_batch_size": 32,
    "learning_rate": 1e-3,
    "max_epochs": 200,
    "val_interval": 10,  # check validation score after n epochs
    "lr_scheduler": "cosine_decay",  # just to keep track

    # Unet model (you can even use nested dictionary and this will be handled by W&B automatically)
    "model_type": "unet",  # just to keep track
    "model_params": dict(spatial_dims=2,
                         in_channels=1,
                         out_channels=1,
                         channels=(8, 16, 32, 64),
                         strides=(2, 2, 2),
                         num_res_units=2,
                         norm=Norm.BATCH,
                         dropout=0.3,
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
# TODO: also include GM mask
train_labels = \
    sorted(glob.glob(os.path.join(data_dir, "derivatives", "**", "*_label-WM_mask.nii.gz"), recursive=True))
train_images_match, train_labels_match = match_images_and_labels(train_images, train_labels)
data_dicts = [{"image": image_name, "label": label_name}
              for image_name, label_name in zip(train_images_match, train_labels_match)]
# TODO: add check if data empty

# Iterate across image/label 3D volume, fetch non-empty slice and output a single list of image/label pair
patch_data = []
for data_dict in data_dicts:
    # i=1
    nii_image = load(data_dict['image'])
    nii_label = load(data_dict['label'])
    for i_z in range(nii_label.shape[2]):
        image_z = nii_image.get_fdata()[:, :, i_z]
        label_z = nii_label.get_fdata()[:, :, i_z]
        if label_z.sum() > 0:
            patch_data.append({'image': image_z, 'label': label_z})

# TODO: optimize hyperparam:
#  RandAffined
train_transforms = Compose(
    [
        AddChanneld(keys=["image", "label"]),
        # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys="image", offsets=0.2, prob=0.5),
        # RandHistogramShiftd(keys=["image"], num_control_points=10, prob=1.0),
        # RandBiasFieldd(keys=["image"], degree=3, coeff_range=(0.0, 0.1)),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True,
                                        relative=False),
        # ScaleIntensityd(keys=["image"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.5, spatial_size=(200, 200),
                    translate_range=(20, 20), rotate_range=np.pi/30, scale_range=(0.1, 0.1)),
        Rand2DElasticd(keys=["image", "label"], spacing=(30, 30), magnitude_range=(3, 3), prob=0.3),
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

# TODO: Randomize train/val
train_id = [0, 2, 3, 4, 6, 8, 10, 12, 14, 15, 17, 19, 20, 21, 23]
val_id = [1, 5, 7, 9, 11, 13, 16, 18, 22]
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
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)
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
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
# post_label = Compose([AsDiscrete(to_onehot=2)])
post_label = Compose()
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
    wandb.log({"Training/learning_rate": scheduler.get_lr()[0]})

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                # TODO: parametrize this
                roi_size = (200, 200)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)

                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

                # 🐝 show image with ground truth and prediction on eval dataset
                # TODO: display subject name and slice number
                slice_num = 45
                wandb.log({"Validation_Image/Image":
                               wandb.Image(val_inputs.cpu(), caption=f"Slice: {slice_num}")})
                wandb.log({"Validation_Image/Ground truth":
                               wandb.Image(val_labels[0].cpu(), caption=f"Slice: {slice_num}")})
                wandb.log({"Validation_Image/Prediction":
                               wandb.Image(val_outputs[0].cpu(), caption=f"Slice: {slice_num}")})

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
