"""
Author: Omkar Damle
Date: May 2018

Training code for Masktrack
 - using Deeplab Resnet-101
 - Offline training

"""


from docopt import docopt
import timeit

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from utility_functions import *
from path import Path
from dataloaders import davis17_offline_dataset as db17_offline
import deeplab_resnet

docstr = """
Usage: 
    train.py [options]

Options:
    -h, --help                  Print this message
    --NoLabels=<int>            The number of different labels in training data, Masktrack has 2 labels - foreground and background, including background [default: 2]
    --lr=<float>                Learning Rate [default: 0.001]
    --wtDecay=<float>          Weight decay during training [default: 0.001]
    --epochResume=<int>        Epoch from which to resume offline training [default: 0]
    --epochs=<int>             Training epochs [default: 1]
    --batchSize=<int>           Batch Size [default: 1]
"""

args = docopt(docstr, version='v0.1')

######################################################################################################################

# Setting of parameters
debug_mode = False
cudnn.enabled = True
weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])
resume_epoch = int(args['--epochResume'])  # Default is 0, change if want to resume
nEpochs = int(args['--epochs'])  # Number of epochs for training (500.000/2079)
batch_size = int(args['--batchSize'])
db_root_dir = Path.db_offline_train_root_dir()
nAveGrad = 4  # keep it even

save_dir = os.path.join(Path.save_offline_root_dir(), 'lr_' + str(base_lr) + '_wd_' + str(weight_decay))

if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))

######################################################################################################################
"""Initialise the network"""
net = deeplab_resnet.Res_Deeplab_no_msc(int(args['--NoLabels']))
net.float()
modelName = 'parent'

if resume_epoch == 0:
    saved_state_dict = torch.load('pretrained/MS_DeepLab_resnet_pretrained_COCO_init.pth')

    """
    Last layer structure is modified according to the number of segmentation classes.
    For Mastrack, #classes=2
    """

    if int(args['--NoLabels']) != 21:
        for i in saved_state_dict:
            i_parts = i.split('.')
            if i_parts[1] == 'layer5':
                saved_state_dict[i] = net.state_dict()[i]

            if i_parts[1] == 'conv1':
                saved_state_dict[i] = torch.cat((saved_state_dict[i], torch.FloatTensor(64, 1, 7, 7).normal_(0,0.0001)), 1)

    net.load_state_dict(saved_state_dict)

    # Let us make it run on multiple GPUs!
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    if torch.cuda.is_available():
        print('CUDA available')
        net.cuda()
    else:
        print('CUDA not available')

else:

    # Let us make it run on multiple GPUs!
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    if torch.cuda.is_available():
        print('CUDA available')
        net.cuda()
    else:
        print('CUDA not available')

    print("Updating weights from: {}".format(
        os.path.join(save_dir, modelName + '_epoch-' + str(resume_epoch) + '.pth')))

    net.load_state_dict(torch.load(os.path.join(save_dir, modelName + '_epoch-' + str(resume_epoch) + '.pth')))


# net.train()

optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(net), 'lr': base_lr},
                       {'params': get_10x_lr_params(net), 'lr': 10 * base_lr}],
                      lr=base_lr, momentum=0.9, weight_decay=weight_decay)

if os.path.exists(os.path.join(save_dir, 'logs')) == False:
    os.mkdir(os.path.join(save_dir, 'logs'))


######################################################################################################################
"""Open file pointers for logging"""

file_offline_loss = open(os.path.join(save_dir, 'logs/logs_offline_training_start_epoch_' + str(resume_epoch) + '.txt'), 'w+')
file_offline_train_precision = open(os.path.join(save_dir, 'logs/logs_offline_training_train_precision_start_epoch_' + str(resume_epoch) + '.txt'),'w+')
file_offline_train_recall = open(os.path.join(save_dir, 'logs/logs_offline_training_train_recall_start_epoch_' + str(resume_epoch) + '.txt'), 'w+')


file_offline_val_loss = open(os.path.join(save_dir, 'logs/logs_offline_training_val_start_epoch_' + str(resume_epoch) + '.txt'), 'w+')
file_offline_val_precision = open(os.path.join(save_dir, 'logs/logs_offline_training_val_precision_start_epoch_' + str(resume_epoch) + '.txt'), 'w+')
file_offline_val_recall = open(os.path.join(save_dir, 'logs/logs_offline_training_val_recall_start_epoch_' + str(resume_epoch) + '.txt'), 'w+')

loss_array = []
loss_minibatch_array = []
precision_train_array  = []
recall_train_array = []

loss_val_array = []
precision_val_array = []
recall_val_array = []

aveGrad = 0



######################################################################################################################
"""Initialise the dataloaders"""

dataset17_train = db17_offline.DAVIS17Offline(train=True, mini=False, mega=True, db_root_dir=Path.db_offline_train_root_dir(), transform=apply_custom_transform, inputRes=(480,854))
dataloader17_train = DataLoader(dataset17_train, batch_size=batch_size, shuffle=True, num_workers=2)

dataset17_val = db17_offline.DAVIS17Offline(train=False, mini=False, mega=True, db_root_dir=Path.db_offline_train_root_dir(), transform=apply_val_custom_transform,inputRes=(480,854))
dataloader17_val = DataLoader(dataset17_val, batch_size=batch_size, shuffle=False, num_workers=2)

lr_factor_array = [1,1,1,0.1,1,1,1,0.1,1,1,1,1,1,0.1,1,1,1,1]

######################################################################################################################
print("Training Network")

for epoch in range(resume_epoch+1, nEpochs+1):

    trainingDataSetSize = 0
    epochLoss = 0
    epochTrainPrecision = 0
    epochTrainRecall = 0

    valDataSetSize = 0
    epochValLoss = 0
    epochValPrecision = 0
    epochValRecall = 0

    start_time = timeit.default_timer()

    ######################################################################################################################
    print('Training phase')
    print('len of loader: ' + str(len(dataloader17_train)))

    net.train()
    optimizer.zero_grad()
    aveGrad = 0

    for data_id, sample in enumerate(dataloader17_train):

        dic = net.state_dict()

        if data_id%100 ==0:
            print(data_id)

        image = sample['image']
        anno = sample['gt']
        deformation = sample['deformation']

        # Making sure the mask input is similar to RGB values
        deformation[deformation==0] = -100
        deformation[deformation==1] = 100

        if debug_mode:
            cv2.imwrite('anno.png', anno[0].numpy().squeeze()*255)
            cv2.imwrite('defo.png', deformation[0].numpy().squeeze()*255)


        prev_frame_mask = Variable(deformation).float()
        inputs, gts = Variable(image), Variable(anno)

        if torch.cuda.is_available():
            inputs, gts, prev_frame_mask = inputs.cuda(), gts.cuda(), prev_frame_mask.cuda()

        input_rgb_mask = torch.cat([inputs, prev_frame_mask], 1)
        noImages, noChannels, height, width = input_rgb_mask.shape

        output_mask = net(input_rgb_mask)

        upsampler = torch.nn.Upsample(size=(height, width), mode='bilinear')
        output_mask = upsampler(output_mask)

        if debug_mode:
            temp_out = np.zeros(output_mask[0][0].shape)
            temp_out[output_mask.data.cpu().numpy()[0][1] > output_mask.data.cpu().numpy()[0][0]] = 1
            cv2.imwrite('output.png',temp_out*255)

        loss1 = cross_entropy_loss(output_mask, gts)
        epochTrainPrecision += calculate_precision(output_mask, gts)
        epochTrainRecall += calculate_recall(output_mask, gts)

        print(loss1.data[0])
        loss_minibatch_array.append(loss1.data[0])

        epochLoss += loss1.data[0]
        trainingDataSetSize += 1

        # Backward the averaged gradient
        loss1 /= nAveGrad
        loss1.backward()
        aveGrad += 1

        # Update the weights once in nAveGrad forward passes
        if aveGrad % nAveGrad == 0:
            optimizer.step()
            optimizer.zero_grad()
            aveGrad = 0

    ######################################################################################################################
    print('Validation phase')
    aveGrad = 0
    net.eval()

    for data_id, sample in enumerate(dataloader17_val):

        image = sample['image']
        anno = sample['gt']
        deformation = sample['deformation']

        deformation[deformation==0] = -100
        deformation[deformation==1] = 100

        prev_frame_mask = Variable(deformation, volatile=True).float()
        inputs, gts = Variable(image, volatile=True), Variable(anno, volatile=True)

        if torch.cuda.is_available():
            inputs, gts, prev_frame_mask = inputs.cuda(), gts.cuda(), prev_frame_mask.cuda()

        input_rgb_mask = torch.cat([inputs, prev_frame_mask], 1)

        noImages, noChannels, height, width = input_rgb_mask.shape

        output_mask = net(input_rgb_mask)

        upsampler = torch.nn.Upsample(size=(height, width), mode='bilinear')
        output_mask = upsampler(output_mask)

        loss1 = cross_entropy_loss(output_mask, gts)

        epochValPrecision += calculate_precision(output_mask, gts)
        epochValRecall += calculate_recall(output_mask, gts)

        epochValLoss += loss1.data[0]
        valDataSetSize += 1

    epochLoss = epochLoss / trainingDataSetSize
    epochTrainPrecision = epochTrainPrecision / trainingDataSetSize
    epochTrainRecall = epochTrainRecall / trainingDataSetSize

    epochValLoss = epochValLoss / valDataSetSize
    epochValPrecision = epochValPrecision / valDataSetSize
    epochValRecall = epochValRecall / valDataSetSize

    print('Epoch: ' + str(epoch) + ', Training Loss: ' + str(epochLoss) + '\n')
    print('Epoch: ' + str(epoch) + ', Train Precision: ' + str(epochTrainPrecision) + '\n')
    print('Epoch: ' + str(epoch) + ', Train Recall: ' + str(epochTrainRecall) + '\n')

    print('Epoch: ' + str(epoch) + ', Val Loss: ' + str(epochValLoss) + '\n')
    print('Epoch: ' + str(epoch) + ', Val Precision: ' + str(epochValPrecision) + '\n')
    print('Epoch: ' + str(epoch) + ', Val Recall: ' + str(epochValRecall) + '\n')

    file_offline_loss.write('Epoch: ' + str(epoch) + ', Loss: ' + str(epochLoss) + '\n')
    file_offline_train_precision.write('Epoch: ' + str(epoch) + ', Precision: ' + str(epochTrainPrecision) + '\n')
    file_offline_train_recall.write('Epoch: ' + str(epoch) + ', Recall: ' + str(epochTrainRecall) + '\n')

    file_offline_val_loss.write('Epoch: ' + str(epoch) + ', Loss: ' + str(epochValLoss) + '\n')
    file_offline_val_precision.write('Epoch: ' + str(epoch) + ', Precision: ' + str(epochValPrecision) + '\n')
    file_offline_val_recall.write('Epoch: ' + str(epoch) + ', Recall: ' + str(epochValRecall) + '\n')

    loss_array.append(epochLoss)
    precision_train_array.append(epochTrainPrecision)
    recall_train_array.append(epochTrainRecall)

    loss_val_array.append(epochValLoss)
    precision_val_array.append(epochValPrecision)
    recall_val_array.append(epochValRecall)

    file_offline_loss.flush()
    file_offline_val_loss.flush()
    file_offline_train_precision.flush()
    file_offline_val_precision.flush()
    file_offline_train_recall.flush()
    file_offline_val_recall.flush()

    epochLoss = 0


    stop_time = timeit.default_timer()

    epoch_secs = stop_time - start_time
    epoch_mins = epoch_secs / 60
    epoch_hr = epoch_mins / 60

    print('This epoch took: ' + str(epoch_hr) + ' hours')

    torch.save(net.state_dict(), os.path.join(save_dir, modelName + '_epoch-' + str(epoch) + '.pth'))

    #change_lr(optimizer, epoch)
    #read_lr(optimizer, save_dir)

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*lr_factor_array[epoch-1]


    plot_loss1(loss_array, resume_epoch, epoch , save_dir)
    plot_loss1(loss_val_array, resume_epoch, epoch , save_dir, val=True)
    plot_loss_minibatch(loss_minibatch_array, save_dir)
    plot_precision_recall(precision_train_array, recall_train_array, precision_val_array, recall_val_array, resume_epoch, epoch, save_dir)


file_offline_loss.close()
file_offline_train_precision.close()
file_offline_train_recall.close()

file_offline_val_loss.close()
file_offline_val_precision.close()
file_offline_val_recall.close()

print('Offline training completed. Have a look at the plots to ensure hyperparameter selection is appropriate.')
print('Need to fine-tune on each test video using online training.')