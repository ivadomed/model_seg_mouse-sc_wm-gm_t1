# Training of a nnUNet model for SC WM and GM segmentation

Here, we detail all the steps necessary to train and use an nnUNet model for the segmentation of mouse SC WM an GM. 
The steps detail how to :
- set-up the environment
- preprocess the data
- train the model
- performing inference

## Installation

This section explains how to install and use the model on new images. 

Clone the repository:
~~~
git clone https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1.git
cd model_seg_mouse-sc_wm-gm_t1
~~~

We recommend to use a virtual environment with python 3.9 to use nnUNet: 
~~~
conda create -n venv_nnunet python=3.9
~~~

We activate the environment:
~~~
conda activate venv_nnunet
~~~

Then install the required libraries:
~~~
pip install -r utils/requirements.txt
~~~

## Data

Download dataset (internal git-annex at NeuroPoly): `zurich-mouse`

## Data pre-processing

Create the following folders:

~~~
mkdir nnUNet_raw
mkdir nnUNet_preprocessed
mkdir nnUNet_results
~~~

### Extract slices

Extract slices from the original file (dimension (500,500,200)) and their label, and create a new file (dimesion (500,500,1)) in another dataset folder:

~~~
python ./utils/extract_slices.py --path-data /path/to/data --path-out /path/to/project/folder
~~~

### Crop images

Crop images and their respective mask in order to remove certain slices: 

~~~
python ./utils/crop_image_and_mask.py --folder-path /path/to/folder/  --file-name file_name.nii.gz --first-slice XXX --last-slice XXX
~~~

### Convert from BIDS to nnU-Net file structure

Before using the nnU-Net model, we convert the dataset from the BIDS format to the nnU-Net fornat:

~~~
python ./utils/convert_bids_to_nnunet.py --path-data /path/to/data_extracted --path-out /path/to/nnUNet_raw --taskname TASK-NAME --tasknumber DATASET-ID
~~~

This will output a dataset called `DatasetDATASET-ID_TASK-NAME` in the `/nnUNet_raw` folder. (DATASET-ID has to be between 100 and 999).

> [!NOTE] 
> In the `convert_bids_to_nnunet` script, all the labeled data is used for training and the unlabeled-data is used for inference.

### Convert from nnU-Net file structure to BIDS

After using an nnU-Net, if you want to convert back to the BIDS format, run:

~~~
python ./utils/convert_nnunet_to_bids.py --path-conversion-dict /PATH/TO/DICT --path-segmentation-folder /PATH/SEG --path-dataset /PATH/DATASET --mask-name MASK_NAME
~~~

This will output a dataset add a segmentation `mask_name` in the dataset derivatives.

### nnUNet data preprocessing

Before training the model, nnU-Net performs data preprocessing and checks the integrity of the dataset:

~~~
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

nnUNetv2_plan_and_preprocess -d DATASET-ID --verify_dataset_integrity
~~~

You will get the configuration plan for all four configurations (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres).
> [!NOTE] 
> In the case of the zurich_mouse dataset, nifti files are not fully annotated, therefore we use a 2d configuration.


## Train model

To train the model, use the following command:
~~~
CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASET-ID CONFIG FOLD --npz
~~~
> [!NOTE] 
> Example for Dataset 101, on 2d config on fold 0: CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 101 2d 0 --npz

You can track the progress of the model with: 
~~~
nnUNet_results/DatasetDATASET-ID_TASK-NAME/nnUNetTrainer__nnUNetPlans__CONFIG/fold_FOLD/progress.png
~~~

## Running inference

To run inference using our trained model, we recommend using the instructions in [README.md](../README.md). However, if you want to perform inference on your own model, there are multiple ways to do so. 

### Method 1 - Using your previous training

Format the image data to the nnU-Net file structure. 
Use a terminal command line:
~~~
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

CUDA_VISIBLE_DEVICES=XXX nnUNetv2_predict -i /path/to/image/folder -o /path/to/predictions -d DATASET_ID -c CONFIG --save_probabilities -chk checkpoint_best.pth -f FOLD
~~~

You can now access the predictions in the folder `/path/to/predictions`. 

### Method 2 - Using our trained model on terminal 

Format the image data to the nnU-Net file structure. 
Download the `model.zip` from the [release](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/releases/tag/v0.3) and unzip it in the `/nnUNet_results` folder (it also requires to export the 3 variables as done previously). 
Then run the terminal command linde:
~~~
CUDA_VISIBLE_DEVICES=XXX nnUNetv2_predict -i /path/to/image/folder -o /path/to/predictions -d 500 -c 3d_fullres --save_probabilities -chk checkpoint_best.pth -f 4
~~~

You can now access the predictions in the folder `/path/to/predictions`. 

## Apply post-processing

nnU-Net v2 comes with the possiblity of performing post-processing on the segmentation images. This was not included in the run inference script as it doesn't bring notable change to the result. To run post-processing run the following script.

~~~
CUDA_VISIBLE_DEVICES=XX nnUNetv2_apply_postprocessing -i /seg/folder -o /output/folder -pp_pkl_file /path/to/postprocessing.pkl -np 8 -plans_json /path/to/post-processing/plans.json
~~~
> [!NOTE]  
> The file `postprocessing.pkl` is stored in `Dataset500_zurich_mouse/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl`.<br>
> The file `plans.json` is stored in `Dataset500_zurich_mouse/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json`. 
