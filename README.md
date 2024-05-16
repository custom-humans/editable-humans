# Learning Locally Editable Virtual Humans

## [Project Page](https://custom-humans.github.io/) | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ho_Learning_Locally_Editable_Virtual_Humans_CVPR_2023_paper.pdf) | [Youtube(3min)](https://youtu.be/aT8ql5hB3ZM), [Shorts(18sec)](https://youtube.com/shorts/6LTXma_wn4c) | [Dataset](https://custom-humans.ait.ethz.ch/)

<img src="assets/teaser.gif" width="800"/> 

Official code release for CVPR 2023 paper [*Learning Locally Editable Virtual Humans*](https://custom-humans.github.io/).

If you find our code, dataset, and paper useful, please cite as
```
@inproceedings{ho2023custom,
    title={Learning Locally Editable Virtual Humans},
    author={Ho, Hsuan-I and Xue, Lixin and Song, Jie and Hilliges, Otmar},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
  }
```

## Installation
Our code has been tested with PyTorch 1.11.0, CUDA 11.3, and an RTX 3090 GPU.

```bash
pip install -r requirements.txt
```

## Quick Start

⚠️ The model checkpoint contains several real human bodies and faces. To download the checkpoint file, you need to agree the CustomHumans Dataset Terms of Use. Click [here](https://custom-humans.ait.ethz.ch/) to apply for the dataset. You will find the checkpoint file in the dataset download link.

1. Download and put the checkpoint file into the `checkpoints` folder.

2. Download the test meshes and images from [here](https://files.ait.ethz.ch/projects/custom-humans/test.zip) and put them into the `data` folder.

3. Run a quick demo on fitting to the unseen 3D scan and 2D images.
```bash!
python demo.py --pretrained-root checkpoints/demo --model-name model-1000.pth
```
You should be able to wear me a Doge T-shirt.

<img src="assets/doge.gif" width="512"/> 

4. Try out different functions such as reposing and cloth transfer in `demo.py`. 

## Data Preparation

### CustomHumans
Apply our dataset by sending a [request](https://custom-humans.ait.ethz.ch/). After downloading, you should get 646 textured meshes and SMPL-X meshes. We use only 100 meshes for training. We provide the indices of training meshes [here](https://github.com/custom-humans/editable-humans/blob/main/data/Custom_train.json).

1. Prepare the training data following the folder structure:
```
	training_dataset
	├── 0003
	│   ├── mesh-f00101.obj
	│   ├── mesh-f00101.mtl
	│   ├── mesh-f00101.png
	│   ├── mesh-f00101.json
	│   └── mesh-f00101_smpl.obj
	├── 0007
	│   ...

```
You can use the following script to generate the training dataset folder:
```bash!
python tools/prepare_dataset.py
```

2. Download [SMPL-X](https://smpl-x.is.tue.mpg.de/) models and move them to the `smplx` folder.
You should have the following data structure:
```
	smplx
	├── SMPLX_NEUTRAL.pkl
	├── SMPLX_NEUTRAL.npz
	├── SMPLX_MALE.pkl
	├── SMPLX_MALE.npz
	├── SMPLX_FEMALE.pkl
	└── SMPLX_FEMALE.npz
```
3. Since online sampling points on meshes during training can be slow, we sample 18M points per mesh and cache them in an h5 file for training. Run the following script to generate the h5 file.

```bash!
python generate_dataset.py -i /path/to/dataset/folder
```

⚠️ The script will generate a large h5 file (>80GB). If you don't want to generate that many points, you can adjust the `NUM_SAMPLES` parameter [here](https://github.com/custom-humans/editable-humans/blob/main/generate_dataset.py#L18).

### THuman2.0

We also train our model using 150 scans in Thuman2.0 and you can find their indices [here](https://github.com/custom-humans/editable-humans/blob/main/data/THUMAN_train.json). Please apply for the dataset and SMPL-X registrations through their [official repo](https://github.com/ytrock/THuman2.0-Dataset).

⚠️ Note that the scans in THuman2.0 are in various scales. We rescale them to -1~1 based on the SMPL-X models. You can find the rescaling script [here](https://github.com/custom-humans/editable-humans/blob/main/tools/align_thuman.py)

⚠️ THuman2.0 uses different settings for creating SMPL-X body meshes. When generating the h5 file, please change to `flat_hand_mean=False` in the [`generate_dataset.py`](https://github.com/custom-humans/editable-humans/blob/main/generate_dataset.py#L42) script.

## Training

Once your h5 dataset is ready, simply run the command to train the model. 
```
python train.py 
```
Here are some configuration flags you can use, they will override the setting in `config.yaml`
* `--config`: path to the config file. Default is `config.yaml`
* `--wandb`: we use wandb for monitoring the training. Activate this flag if you want to use it.
* `--save-root`: path to the folder to save the checkpoints. Default is `checkpoints`
* `--data_root`: path to the training h5 dataset. Default is `CustomHumans.h5`
* `--use_2d_from_epoch`: use 2D adversarial loss after this epoch. -1 means never use 2D loss. Default is 10.

## Evaluation

We use SIZER to evaluate the geometry fitting performance. Please follow the instructions to download their [dataset](https://github.com/garvita-tiwari/sizer).

We provide subjets' [indices](https://github.com/custom-humans/editable-humans/blob/main/data/SIZER_test.json) and [scripts](https://github.com/custom-humans/editable-humans/blob/main/tools/evaluate.py) for evaluation. 

# Acknowledgement
We have used codes from other great research work, including [gdna](https://github.com/xuchen-ethz/gdna), [kaolin-wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp), [SMPL-X](https://github.com/vchoutas/smplx), [ML-GSN](https://github.com/apple/ml-gsn/), [StyleGAN-Ada](https://github.com/NVlabs/stylegan2-ada-pytorch), [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks). 

We create all the videos using powerful [aitviewer](https://eth-ait.github.io/aitviewer/).

We sincerely thank the authors for their awesome work!
