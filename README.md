# White and grey matter segmentation on T1-weighted exvivo mouse spinal cord

https://user-images.githubusercontent.com/2482071/227744144-ff9b21c3-d757-4e4c-a990-f6d7bf3084b0.mov

## Citation

Publication linked to the dataset: Coming soon!

Publication linked to this model: see [CITATION.cff](./CITATION.cff)


## Installation

Clone the repository:
~~~
git clone https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1.git
cd model_seg_mouse-sc_wm-gm_t1
~~~

We recommend installing a virtual environnment with python 3.9 installed (required for nnU-Net v2). Follow the instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) 

Then install the required libraries:
~~~
pip install -r requirements.txt
~~~

## Project description

In this project, we trained a 3D nnU-Net for spinal cord white and grey matter segmentation. The data contains 22 mice with different number of chunks, for a total of 72 MRI 3D images. Each MRI image is T2-weighted, has a size of 200x200x500, with the following resolution: 0.05x0.05x0.05 mm. 

In order to train a 3D nnU-Net, the following steps were completed: 
- First, a total of 161 slices were labelled on various subjects. See [Notes](#notes) for details on the manual labeling.
- The slices were then extracted using the [extract_slices.py](./utils/extract_slices.py) function: it extracted both the slice from the MRI image as well as the mask's slice. These were gathered into a temporary dataset, on which a 2D nnU-Net model was trained to segment spinal cord white and grey matter. The inference was then performed using this model on the full 3D volume from the original dataset. 
- Then, a 3D nnU-Net was trained on the images, using the results from the previous inference as ground truth as well as using extracted slices (of shape (200x200x1)) and their manual segmentation. The inference, was again performed on the full zurich-mouse dataset. Going from a 2D nnU-Net to a 3D nnU-Net helped improved the continuity of the segmentation on the z-axis. 
- After that, we selected the best segmentation masks on the dataset totalling 31 images. For each of these images we noted that the top and bottom slices were often poorly annotated. Using the [crop_image_and_mask.py](./utils/crop_image_and_mask.py) script we removed these slices. The objective was to keep only qualitative annotations. 
- Finally, a 3D nnU-Net was trained on these qualitative image segmentations (31 images) with various dimension as well as annotated slices (161 images). The nnU-Net was trained on 1000 epochs, with "3d_fullres" configuration and on 5 folds. The best Dice score were the following (fold 0 : 0.9135, fold 1: 0.9083, fold 2: 0.9109 , fold 3: 0.9132, fold 4: 0.9173). 

For the packaging we decided to keep only fold 4 as it has the best dice score and all performed similarly in terms of final results as well as training evolution (meaning that the dataset is rather homogeneous). The reason for this is to avoid having to upload the full results model which weight around 5 GB and limit ourself to 250 MB. Also, inference is much longer when performed on 5 folds instead of 1 and results are comparable. 

## Data

Download dataset (internal git-annex at NeuroPoly): `zurich-mouse`


## Data pre-processing

Create the following folders:

~~~
mkdir nnUNet_raw
mkdir nnUNet_preprocessed
mkdir nnUNet_results
~~~

**Extract slices**

Extract slices from the original file (dimension (500,500,200)) and their label, and create a new file (dimesion (500,500,1)) in another dataset folder:

~~~
python ./utils/extract_slices.py --path-data /path/to/data --path-out /path/to/project/folder
~~~

**Crop images**

Crop images and their respective mask in order to remove certain slices: 

~~~
python ./utils/crop_image_and_mask.py --folder-path /path/to/folder/  --file-name file_name.nii.gz --first-slice XXX --last-slice XXX
~~~

**Convert from BIDS to nnU-Net file structure**

Before using the nnU-Net model, we convert the dataset from the BIDS format to the nnU-Net fornat:

~~~
python ./utils/convert_bids_to_nnunet.py --path-data /path/to/data_extracted --path-out /path/to/nnUNet_raw --taskname TASK-NAME --tasknumber DATASET-ID
~~~

This will output a dataset called `DatasetDATASET-ID_TASK-NAME` in the `/nnUNet_raw` folder. (DATASET-ID has to be between 100 and 999).

> **Note**
> In the `convert_bids_to_nnunet` script, all the labeled data is used for training and the unlabeled-data is used for inference. 

## Data preprocessing

Before training the model, nnU-Net performs data preprocessing and checks the integrity of the dataset:

~~~
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

nnUNetv2_plan_and_preprocess -d DATASET-ID --verify_dataset_integrity
~~~

You will get the configuration plan for all four configurations (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres).

> **Note**
> In the case of the zurich_mouse dataset, nifti files are not fully annotated, therefore we use a 2d configuration.


## Train model

To train the model, use the following command:
~~~
CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASET-ID CONFIG FOLD --npz
~~~

> **Note**
> Example for Dataset 101, on 2d config on fold 0: CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 101 2d 0 --npz

You can track the progress of the model with: 
~~~
nnUNet_results/DatasetDATASET-ID_TASK-NAME/nnUNetTrainer__nnUNetPlans__CONFIG/fold_FOLD/progress.png
~~~

## Run inference

To run an inference and obtain a segmentation, there are two ways to do so. 

### Method 1 - Using your previous training

Format the image data to the nnU-Net file structure. 
Use a terminal command line:
~~~
CUDA_VISIBLE_DEVICES=XXX nnUNetv2_predict -i /path/to/image/folder -o /path/to/predictions -d DATASET_ID -c CONFIG --save_probabilities -chk checkpoint_best.pth -f FOLD
~~~

You can now access the predictions in the folder `/path/to/predictions`. 

### Method 2 - Using our trained model on terminal 

Format the image data to the nnU-Net file structure. 
Download the `Dataset500_zurich_mouse.zip` from the release and unzip it in the `/nnUNet_results` folder. 
Then run the terminal command linde:
~~~
CUDA_VISIBLE_DEVICES=XXX nnUNetv2_predict -i /path/to/image/folder -o /path/to/predictions -d 500 -c 3d_fullres --save_probabilities -chk checkpoint_best.pth -f 4
~~~

You can now access the predictions in the folder `/path/to/predictions`. 

### Method 3 - Using our trained model with `test.py`

Download the model `Dataset500_zurich_mouse.zip` from the release and unzip it. 
Use the `test.py` function:

To run on the whole dataset:
~~~
python test.py --path-dataset /path/to/test-dataset --path-out /path/to/output --path-model /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres
~~~

To run on individual(s) NIfTI image(s):
~~~
python test.py --path-images /path/to/image1 /path/to/image2 --path-out /path/to/output --path-model /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres
~~~

## Apply post-processing

nnU-Net v2 comes with the possiblity of performing post_processing on the segmentation images. This was not included in the run inference script as it doesn't bring notable change to the result. To run post-processing run the following script.

~~~
CUDA_VISIBLE_DEVICES=XX nnUNetv2_apply_postprocessing -i /seg/folder -o /output/folder -pp_pkl_file /path/to/postprocessing.pkl -np 8 -plans_json /path/to/post-processing/plans.json
~~~

## Notes

Procedure for ground truth mask creation: https://youtu.be/KVL-JzcSRTo

Before applying the model, make sure the image orientation is correct. More details [here](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/issues/25). 
