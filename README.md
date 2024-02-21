# White and grey matter segmentation on T1-weighted exvivo mouse spinal cord

[![DOI](https://zenodo.org/badge/587907110.svg)](https://doi.org/10.5281/zenodo.7772350)

https://user-images.githubusercontent.com/2482071/227744144-ff9b21c3-d757-4e4c-a990-f6d7bf3084b0.mov

## Citation

Publication linked to the dataset: Coming soon!

Publication linked to this model: see [CITATION.cff](./CITATION.cff)


## Project description

In this project, we trained a 3D nnU-Net for spinal cord white and grey matter segmentation. The data contains 22 mice with different number of chunks, for a total of 72 MRI 3D images. Each MRI image is T1-weighted, has a size of 200x200x500, with the following resolution: 0.05x0.05x0.05 mm. 

<details>
  <summary>Expand this for more information on how we trained the model</summary>
  
In order to train a 3D nnU-Net, the following steps were completed: 
- First, a total of 161 slices were labelled on various subjects. See [Notes](#notes) for details on the manual labeling.
- The slices were then extracted using the [extract_slices.py](./utils/extract_slices.py) function: it extracted both the slice from the MRI image as well as the mask's slice. These were gathered into a temporary dataset, on which a 2D nnU-Net model was trained to segment spinal cord white and grey matter. The inference was then performed using this model on the full 3D volume from the original dataset. 
- Then, a 3D nnU-Net was trained on the images, using the results from the previous inference as ground truth as well as using extracted slices (of shape (200x200x1)) and their manual segmentation. The inference, was again performed on the full zurich-mouse dataset. Going from a 2D nnU-Net to a 3D nnU-Net helped improved the continuity of the segmentation on the z-axis. 
- After that, we selected the best segmentation masks on the dataset totalling 31 images. For each of these images we noted that the top and bottom slices were often poorly annotated. Using the [crop_image_and_mask.py](./utils/crop_image_and_mask.py) script we removed these slices. The objective was to keep only qualitative annotations. 
- Finally, a 3D nnU-Net was trained on these qualitative image segmentations (31 images) with various dimension as well as annotated slices (161 images). The nnU-Net was trained on 1000 epochs, with "3d_fullres" configuration and on 5 folds. The best Dice score were the following (fold 0 : 0.9135, fold 1: 0.9083, fold 2: 0.9109 , fold 3: 0.9132, fold 4: 0.9173). 

For the packaging we decided to keep only fold 4 as it has the best dice score and all performed similarly in terms of final results as well as training evolution (meaning that the dataset is rather homogeneous). The reason for this is to avoid having to upload the full results model which weight around 5 GB and limit ourself to 250 MB. Also, inference is much longer when performed on 5 folds instead of 1 and results are comparable. 

</details>

For information on how to retrain the same model, refer to this file [README_training_model.md](./utils/README.md). 

## How to use the model

This is the recommended method to use the model (for other methode refer to [README_training_model.md](./utils/README.md)).

### Install dependencies

- [Spinal Cord Toolbox (SCT) v6.2](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/6.2) or higher -- follow the installation instructions [here](https://github.com/spinalcordtoolbox/spinalcordtoolbox?tab=readme-ov-file#installation)
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
- Python

Once the dependencies are installed, download the latest model:

```bash
sct_deepseg -install-task seg_mouse_gm_wm_t1w
```

### Getting the rootlet segmentation

To segment a single image, run the following command: 

```bash
sct_deepseg -i <INPUT> -o <OUTPUT> -task seg_mouse_gm_wm_t1w
```

For example:

```bash
sct_deepseg -i sub-001_T2w.nii.gz -o sub-001_T2w_wm-gm-seg.nii.gz -task seg_mouse_gm_wm_t1w
```

## Notes

Procedure for ground truth mask creation: https://youtu.be/KVL-JzcSRTo
