# MaskTrack

The [MaskTrack method](https://arxiv.org/abs/1612.02646) is the baseline for state-of-the-art methods in video object segmentation like [Video Object Segmentation with Re-identification](https://arxiv.org/abs/1708.00197) and[ Lucid Data Dreaming for Multiple Object Tracking] (https://arxiv.org/abs/1703.09554). The top three methods in [DAVIS 2017 challenge](https://davischallenge.org/challenge2017/index.html) were based on the MaskTrack method. However, no open source code is available for the MaskTrack method. Here I provide the MaskTrack method with following specifications:
1. The code gives a score of 0.466 on the DAVIS 2017 test-dev dataset. J-mean is 0.440 and F-mean is 0.492.
2. The code handles multiple objects present in DAVIS 2017.
3. Data generation code in matlab for offline training on DAVIS 17 train+val and online training on DAVIS 17 test is also included. Thus, all of the code is packaged together.

## Getting Started

Machine configuration used for testing: 
1. Two 'GeForce GTX 1080 Ti' cards with 11GB memory each.
2. CPU RAM memory of 32 GB (though only about 11GB is required)

Offline training is done on DAVIS 2017 train data. The online training and testing is done on DAVIS 2017 test dataset. I recommend using conda for downloading and managing the environments.

Download the Deeplab Resnet 101 pretrained COCO model from [here](https://drive.google.com/open?id=1jqf7zdYATK2GcgHWzsB6-dzHHPA4Ow2f) and place it in 'training/pretrained' folder.

If you want to skip offline training and directly perform online training and testing, download the offline trained model from [here](https://drive.google.com/open?id=10fHnpSwPrW1jOvLQAM8YKqbAvlTYmxkp) and place it in 'training/offline_save_dir/lr_0.001_wd_0.001' folder.


### Prerequisites

What things you need to install the software and how to install them

Software used:
1. Pytorch 0.3.1
2. Matlab 2017b
3. Python 2.7

Dependencies:
Create a conda environment using the training/deeplab_resnet_env.yml file.
Use: conda env create -f training/deeplab_resnet_env.yml

If you are not using conda as a package manager, refer to the yml file and install the libraries manually.

## Running the code

Please refer to the following instructions:

A. Offline training data generation
* Download the DAVIS 17 train+val dataset from: https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
* Go to generating_masks/offline_training/script_multiple_objs.m and change the base_path variable to point to the DAVIS 2017 dataset (train+val)
* Run script_multiple_objs.m (This will generate the deformations needed for offline training)

B. Setting paths for python files
* Go to training/path.py and change the paths returned by the following methods:
1. db_root_dir(): Point this to the DAVIS 17 test dataset (download from: https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip)
2. db_offline_train_root_dir(): Point this to the DAVIS 17 train+val dataset

C. Offline training
* Run training/train_offline.py with appropriate parameters. Recommended: --NoLabels 2 --lr 0.001 --wtDecay 0.001 --epochResume 0 --epochs 15 --batchSize 6

D. Online Training data generation
* Go to generating_masks/online_training/script_DAVIS17_test_dev.m and change required paths.
* Run the script with 12 iterations (can be varied) as argument: run script_DAVIS17_test_dev(12)

E. Online Training and testing
* Run training/train_online.py with appropriate parameters. Recommended: --NoLabels 2 --lr 0.0025 --wtDecay 0.0001 --seqName aerobatics --parentEpochResume 8 --epochsOnline 10 --noIterations 5 --batchSize 2
* Results are stored in training/models17_DAVIS17_test/lr_xxx_wd_xxx/seq_name

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

This code was produced during my internship at Nanyang Technological University under Prof. Guosheng Lin. I would like to thank him for providing access to the GPUs.

1. The code for generation of masks was based on: www.mpi-inf.mpg.de/masktrack
2. The code for Deeplab Resnet was taken from: https://github.com/isht7/pytorch-deeplab-resnet
3. Some of the dataloader code was based on: https://github.com/fperazzi/davis-2017
4. Template of Readme.md file: https://gist.github.com/PurpleBooth/109311bb0361f32d87a2

I would like to thank K.K. Maninis for providing this code: https://github.com/kmaninis/OSVOS-PyTorch for reference.

