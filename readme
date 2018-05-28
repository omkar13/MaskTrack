The MaskTrack method (originally published in link) is the baseline for state-of-the-art methods in video object segmentation like Video Object Segmentation with re-identification (link) and Lucid Data Dreaming (link). The top three methods in DAVIS 2017 challenge (link) were based on the MaskTrack method. However, no open source code is available for the MaskTrack method. Here I provide the MaskTrack method with following specifications:
1. The code gives a score of 0.466 on the DAVIS 2017 test-dev dataset. J-mean is 0.440 and F-mean is 0.492.
2. The code handles multiple objects present in DAVIS 2017.
3. Data generation code in matlab for offline training on DAVIS 17 train+val and online training on DAVIS 17 test is also included. Thus, all of the code is packaged together.

Machine configuration used for testing: 
1. Two 'GeForce GTX 1080 Ti' cards with 11GB memory each.
2. CPU RAM memory of 32 GB (though only about 11GB is required)

Offline training is done on DAVIS 2017 train data. The online training and testing is done on DAVIS 2017 test dataset. I recommend using conda for downloading and managing the environments.

Software used:
1. Pytorch 0.3.1
2. Matlab 2017b
3. Python 2.7

Dependencies:
Create a conda environment using the training/deeplab_resnet_env.yml file.
Use: conda env create -f training/deeplab_resnet_env.yml

If you are not using conda as a package manager, refer to the yml file and install the libraries manually.

Please refer to the following instructions:

A. Offline training data generation
0. Download the DAVIS 17 train+val dataset from: https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
1. Go to generating_masks/offline_training/script_multiple_objs.m and change the base_path variable to point to the DAVIS 2017 dataset (train+val)
2. Run script_multiple_objs.m (This will generate the deformations needed for offline training)

B. Setting paths for python files
Go to training/path.py and change the paths returned by the following methods:
	a. db_root_dir(): Point this to the DAVIS 17 test dataset (download from: https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip)
	b. db_offline_train_root_dir(): Point this to the DAVIS 17 train+val dataset

C. Offline training
Run training/train_offline.py with appropriate parameters. Recommended: --NoLabels 2 --lr 0.001 --wtDecay 0.001 --epochResume 0 --epochs 15 --batchSize 6

D. Online Training data generation
1. Go to generating_masks/online_training/script_DAVIS17_test_dev.m and change required paths.
2. Run the script with 12 iterations (can be varied) as argument: run script_DAVIS17_test_dev(12)

E. Online Training and testing
1. Run training/train_online.py with appropriate parameters. Recommended: --NoLabels 2 --lr 0.0025 --wtDecay 0.0001 --seqName aerobatics --parentEpochResume 8 --epochsOnline 10 --noIterations 5 --batchSize 2
2. Results are stored in training/models17_DAVIS17_test/lr_xxx_wd_xxx/seq_name

Acknowledgements - 

This code was produced during my internship at Nanyang Technological University under Prof. Guosheng Lin. I would like to thank him for providing access to the GPUs.

1. The code for generation of masks was based on: www.mpi-inf.mpg.de/masktrack
2. The code for Deeplab Resnet was taken from: https://github.com/isht7/pytorch-deeplab-resnet
3. Some of the dataloader code was based on: https://github.com/fperazzi/davis-2017
I would like to thank K.K. Maninis for providing this code: https://github.com/kmaninis/OSVOS-PyTorch for reference.


