# White and grey matter segmentation on T1-weighted exvivo mouse spinal cord

https://user-images.githubusercontent.com/2482071/227744144-ff9b21c3-d757-4e4c-a990-f6d7bf3084b0.mov

Publication linked to the dataset: https://pubmed.ncbi.nlm.nih.gov/35585865/

Dataset: TODO

Procedure for ground truth mask creation: https://youtu.be/KVL-JzcSRTo

## Installation

Clone the repository:
~~~
git clone https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1.git
cd model_seg_mouse-sc_wm-gm_t1
~~~

We recommend installing under a [virtual environment](https://docs.python.org/3/library/venv.html). Then run:
~~~
pip install -r requirements.txt
~~~

## Run segmentation

~~~
python test.py -i NIFTI_IMAGE
~~~
