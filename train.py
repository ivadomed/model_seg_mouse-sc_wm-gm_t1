"""
Spinal cord white and gray matter segmentation (2 classes) using 2D kernel based on MONAI, with WandB monitoring.

Based on this tutorial: https://wandb.ai/gladiator/MONAI_Spleen_3D_Segmentation/reports/3D-Segmentation-with-MONAI-and-PyTorch-Supercharged-by-Weights-Biases---VmlldzoyNDgxNDMz
"""

import os
import glob
import shutil
import tempfile

import wandb
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract

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
        # Fetch subject name
        subject = image.split(os.path.sep)[-1].split("_")[0]
        # Find equivalent in labels
        # TODO: check if label has 2 entries
        label = [j for i, j in enumerate(labels) if subject in j]
        if label:
            images_match.append(image)
            labels_match.append(label[0])
    return images_match, labels_match


"""# Setup data directory

You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.
This allows you to save results and reuse downloads.
If not specified a temporary directory will be used.
"""

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

# Split train/val
# TODO: add randomization in the split
train_files, val_files = data_dicts[:-9], data_dicts[-9:]

# Set deterministic training for reproducibility
set_determinism(seed=0)

# Setup transforms for training and validation
# 1. `LoadImaged` loads the spleen CT images and labels from NIfTI format files.
# 1. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
# 1. `Orientationd` unifies the data orientation based on the affine matrix.
# 1. `Spacingd` adjusts the spacing by `pixdim=(1.5, 1.5, 2.)` based on the affine matrix.
# 1. `ScaleIntensityRanged` extracts intensity range [-57, 164] and scales to [0, 1].
# 1. `CropForegroundd` removes all zero borders to focus on the valid body area of the images and labels.
# 1. `RandCropByPosNegLabeld` randomly crop patch samples from big image based on pos / neg ratio.
# The image centers of negative samples must be in valid body area.
# 1. `RandAffined` efficiently performs `rotate`, `scale`, `shear`, `translate`, etc. together based on PyTorch affine transform.
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=4095,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=4095,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ]
)

# Check DataLoader
check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}")

# Create a function which will log all the slices of the 3D image to W&B to visualize them interactively. Furthermore,
# we will also log the slices with segmentation masks to see the overlayed view of segmentations masks on the slices
# interactively in the W&B dashboard.

# Define Configuration
# Here, we define the configuration for dataloaders, models, train settings in a dictionary. Note that this config
# object would be passed to `wandb.init()` method to log all the necessary parameters that went into the experiment.

config = {
    # data
    "cache_rate": 1.0,
    "num_workers": 0,  # Set to 0 to debug under Pycharm (avoid multiproc). Otherwise, set to 2.

    # train settings
    "train_batch_size": 2,
    "val_batch_size": 1,
    "learning_rate": 1e-3,
    "max_epochs": 100,
    "val_interval": 10, # check validation score after n epochs
    "lr_scheduler": "cosine_decay", # just to keep track

    # Unet model (you can even use nested dictionary and this will be handled by W&B automatically)
    "model_type": "unet", # just to keep track
    "model_params": dict(spatial_dims=3,
                         in_channels=1,
                         out_channels=2,
                         channels=(16, 32, 64, 128, 256),
                         strides=(2, 2, 2, 2),
                         num_res_units=2,
                         norm=Norm.BATCH,
    )
}

# Define CacheDataset and DataLoader for training and validation
# Here we use `CacheDataset` to accelerate training and validation process, it's 10x faster than the regular Dataset.
# To achieve best performance, set `cache_rate=1.0` to cache all the data, if memory is not enough, set lower value.
# Users can also set `cache_num` instead of `cache_rate`, will use the minimum value of the 2 settings.
# And set `num_workers` to enable multi-threads during caching.
# If want to try the regular Dataset, just change to use the commented code below.
train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=config['cache_rate'], num_workers=config['num_workers'])
# train_ds = Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=config['train_batch_size'], shuffle=True, num_workers=config['num_workers'])

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=config['cache_rate'], num_workers=config['num_workers'])
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=config['val_batch_size'], num_workers=config['num_workers'])

# Create Model, Loss, Optimizer and Scheduler

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(**config['model_params']).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
dice_metric = DiceMetric(include_background=False, reduction="mean")
scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=1e-9)

# To avoid https://github.com/jcohenadad/model-seg-ms-mp2rage-monai/issues/1
torch.multiprocessing.set_sharing_strategy('file_system')

# Execute a typical PyTorch training process

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
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

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
        wandb.log({"train/loss": loss.item()})
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    # step scheduler after each epoch (cosine decay)
    scheduler.step()
    
    # 🐝 log train_loss averaged over epoch to wandb
    wandb.log({"train/loss_epoch": epoch_loss})
    
    # 🐝 log learning rate after each epoch to wandb
    wandb.log({"learning_rate": scheduler.get_lr()[0]})

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # 🐝 aggregate the final mean dice result
            metric = dice_metric.aggregate().item()

            # 🐝 log validation dice score for each validation round
            wandb.log({"val/dice_metric": metric})

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
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_data["image"].to(device), roi_size, sw_batch_size, model
        )
        # plot the slice [:, :, 80]
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(val_data["label"][0, 0, :, :, 80])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        plt.imshow(torch.argmax(
            val_outputs, dim=1).detach().cpu()[0, :, :, 80])
        plt.show()
        if i == 2:
            break

"""# Log predictions to W&B in form of table"""

# 🐝 create a wandb table to log input image, ground_truth masks and predictions
columns = ["filename", "image", "ground_truth", "prediction"]
table = wandb.Table(columns=columns)

model.load_state_dict(torch.load(
    os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        # get the filename of the current image
        fn = val_data['image_meta_dict']['filename_or_obj'][0].split("/")[-1].split(".")[0]

        roi_size = (160, 160, 160)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_data["image"].to(device), roi_size, sw_batch_size, model
        )

        # log last 20 slices of each 3D image
        for slice_no in range(80, 100):
            img = val_data["image"][0, 0, :, :, slice_no]
            label = val_data["label"][0, 0, :, :, slice_no]
            prediction = torch.argmax(
                val_outputs, dim=1).detach().cpu()[0, :, :, slice_no]

            # 🐝 Add data to wandb table dynamically    
            table.add_data(fn, wandb.Image(img), wandb.Image(label), wandb.Image(prediction))

# log predictions table to wandb with `val_predictions` as key
wandb.log({"val_predictions": table})

# 🐝 Close your wandb run
wandb.finish()

