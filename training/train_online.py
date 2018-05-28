"""
Author: Omkar Damle
Date: April, 2018.

Aim: Train the model online
"""

from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

import deeplab_resnet
from path import Path
from dataloaders import davis17_online_data as db
from dataloaders import custom_transforms as tr
from dataloaders import davis17 as db17
from dataloaders.config import cfg
from torch.utils.data import DataLoader
from utility_functions import *

import gc
from PIL import Image
from docopt import docopt
import timeit
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

docstr = """

Usage: 
    train.py [options]

Options:
    -h, --help                  Print this message
    --NoLabels=<int>            The number of different labels in training data [default: 2]
    --lr=<float>                Learning Rate [default: 0.0025]
    --wtDecay=<float>          Weight decay during training [default: 0.0001]
    --seqName=<str>            Name of sequence on which online training is to be performed [default: bear]
    --parentEpochResume=<int> Parent epoch from which to resume online training [default: 10]
    --epochsOnline=<int>       Online training epochs [default: 1]    
    --noIterations=<int>       Number of iterations of training data [default: 10]
    --batchSize=<int>           Batch size during online training [default: 2]
"""

######################################################################################################################
"""Setting up parameters"""

args = docopt(docstr, version='v0.1')
cudnn.enabled = True
weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])
seq_name = args['--seqName']
batchSize = int(args['--batchSize'])

p = {
    'trainBatch': batchSize,
}

resume_epoch_parent = int(args['--parentEpochResume'])  # Default is 0, change if want to resume
nEpochs = int(args['--epochsOnline'])
noIterations = int(args['--noIterations'])
testBatch = 1  # Testing Batch
db_root_dir = Path.db_root_dir()
db_parent_root_dir = Path.save_offline_root_dir()

parent_lr = 0.001
parent_wd = 0.001

nAveGrad = 8  # 4 initially....keep it even
aveGrad = 0

save_dir = os.path.join(Path.save_root_dir(),'lr_'+str(base_lr)+'_wd_'+str(weight_decay))

if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))


davis17loader = db17.DAVISLoader(year=cfg.YEAR, phase=cfg.PHASE)
seq_data = davis17loader[seq_name]

images = seq_data.images
anno = seq_data.annotations

composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.5, 1.3)),
                                          tr.ToTensor()])

alreadyTrained = False
file_name = os.path.join(save_dir, 'online_training_' + seq_name + '_object_id_' + str(1) + 'epoch_' + str(nEpochs) + '.pth')

if os.path.exists(file_name):
    print('Training already completed! Not doing it again.')
    alreadyTrained = True

if not alreadyTrained:
    if os.path.exists(os.path.join(save_dir,'logs')) == False:
        os.mkdir(os.path.join(save_dir,'logs'))

    file_online_loss = open(os.path.join(save_dir, 'logs', seq_name + '_loss.txt'), 'w+')
    file_precision = open(os.path.join(save_dir, 'logs', seq_name + '_precision.txt'), 'w+')
    file_recall = open(os.path.join(save_dir, 'logs', seq_name + '_recall.txt'), 'w+')

    loss_array = [[] for i in range(anno.n_objects+1)]
    loss_minibatch_array = [[] for i in range(anno.n_objects+1)]
    precision_array = [[] for i in range(anno.n_objects+1)]
    recall_array = [[] for i in range(anno.n_objects+1)]

    """Initialise the networks"""
    optimizers = [None for i in range(anno.n_objects+1)]
    db_train = [None for i in range(anno.n_objects+1)]
    trainloader = [None for i in range(anno.n_objects+1)]

######################################################################################################################
"""Load the network parameters"""

nets = [None for i in range(anno.n_objects+1)]

for i in range(1,anno.n_objects+1):

    if not alreadyTrained:

        # Training dataset and its iterator
        db_train[i] = db.DAVIS17OnlineDataset(train=True, inputRes=(480,854),db_root_dir=db_root_dir,
                                           transform=composed_transforms, seq_name=seq_name, noIterations=noIterations,
                                              object_id=i)

        trainloader[i] = DataLoader(db_train[i], batch_size=p['trainBatch'], shuffle=False, num_workers=2)

        nets[i] = deeplab_resnet.Res_Deeplab_no_msc(int(args['--NoLabels']))
        nets[i].float()
        nets[i].train()

        # Let us make it run on multiple GPUs!
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            nets[i] = nn.DataParallel(nets[i])

        if torch.cuda.is_available():
            nets[i].cuda()

        print("Updating weights from: {}".format(
            os.path.join(db_parent_root_dir, 'lr_' + str(parent_lr) + '_wd_' + str(parent_wd) , 'parent_epoch-' + str(resume_epoch_parent) + '.pth')))

        nets[i].load_state_dict(
            torch.load(os.path.join(db_parent_root_dir, 'lr_' + str(parent_lr) + '_wd_' + str(parent_wd) , 'parent_epoch-' + str(resume_epoch_parent) + '.pth')))

        optimizers[i] = optim.SGD([{'params': get_1x_lr_params_NOscale(nets[i]), 'lr': base_lr},
                           {'params': get_10x_lr_params(nets[i]), 'lr': 10 * base_lr}],
                          lr=base_lr, momentum=0.9, weight_decay=weight_decay)

    else:
        nets[i] = deeplab_resnet.Res_Deeplab_no_msc(int(args['--NoLabels']))
        nets[i].float()
        #nets[i].eval()
        nets[i].train()

        modelName = 'online_training'

        print("Updating weights from: {}".format(
            os.path.join(save_dir, modelName + '_' + seq_name + '_object_id_' + str(i) + 'epoch_' + str(nEpochs) + '.pth')))

        # Let us make it run on multiple GPUs!
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            nets[i] = nn.DataParallel(nets[i])

        if torch.cuda.is_available():
            nets[i].cuda()

        nets[i].load_state_dict(
           torch.load(os.path.join(save_dir, modelName + '_' + seq_name + '_object_id_' + str(i) + 'epoch_' + str(nEpochs) + '.pth')))

num_img_tr = len(images)
print('Number of images: ' + str(num_img_tr))

# no of training instances
deformations = [1, 2]

lr_factor_array = [1,1,1,0.1,1,1,1,0.1,1,1,1,1,1,0.1,1,1,1,1]

######################################################################################################################
"""Online training loop"""

print("Training Network")
for epoch in range(1,nEpochs+1):

    if alreadyTrained:
        continue

    gc.collect()

    for object_id in range(1, anno.n_objects+1):

        trainingDataSetSize = 0
        epochLoss = 0
        epochTrainPrecision = 0
        epochTrainRecall = 0

        aveGrad = 0
        optimizers[object_id].zero_grad()
        start_time = timeit.default_timer()

        print('Starting for object ID: ' + str(object_id) + ', Epoch no: ' + str(epoch))

        for ii, sample_batched in enumerate(trainloader[object_id]):

            inputs, gts, df1, df2 = sample_batched['image'], sample_batched['gt'], sample_batched['df1'], sample_batched['df2']

            for df_id in deformations:
                if df_id == 1:
                    prev_frame_mask = Variable(df1).float()
                else:
                    inputs, gts = sample_batched['image'], sample_batched['gt']
                    prev_frame_mask = Variable(df2).float()

                prev_frame_mask[prev_frame_mask==0] = -100
                prev_frame_mask[prev_frame_mask==1] = 100

                # Forward-Backward of the mini-batch
                inputs, gts = Variable(inputs), Variable(gts)

                if torch.cuda.is_available():
                    inputs, gts, prev_frame_mask = inputs.cuda(), gts.cuda(), prev_frame_mask.cuda()

                input_rgb_mask = torch.cat([inputs, prev_frame_mask], 1)
                noImages, noChannels, height, width = input_rgb_mask.shape
                output_mask = nets[object_id](input_rgb_mask)
                upsampler = torch.nn.Upsample(size=(height, width), mode='bilinear')
                output_mask = upsampler(output_mask)

                loss1 = cross_entropy_loss(output_mask, gts)

                epochLoss += loss1.data[0]
                epochTrainPrecision += calculate_precision(output_mask, gts)
                epochTrainRecall += calculate_recall(output_mask, gts)
                trainingDataSetSize += 1
                print(loss1.data[0])

                loss_minibatch_array[object_id].append(loss1.data[0])

                # Backward the averaged gradient
                loss1 /= nAveGrad
                loss1.backward()
                aveGrad += 1

                # Update the weights once in nAveGrad forward passes
                if aveGrad % nAveGrad == 0:
                    optimizers[object_id].step()
                    optimizers[object_id].zero_grad()
                    aveGrad = 0

        epochLoss = epochLoss / trainingDataSetSize
        epochTrainPrecision = epochTrainPrecision / trainingDataSetSize
        epochTrainRecall = epochTrainRecall / trainingDataSetSize

        print('Object ID: ' + str(object_id))
        print('Epoch: ' + str(epoch) + ', Loss: ' + str(epochLoss) + '\n')
        print('Epoch: ' + str(epoch) + ', Train Precision: ' + str(epochTrainPrecision) + '\n')
        print('Epoch: ' + str(epoch) + ', Train Recall: ' + str(epochTrainRecall) + '\n')

        file_online_loss.write('Epoch: ' + str(epoch) + ', Object ID: ' + str(object_id)  + ', Loss: ' + str(epochLoss) + '\n')
        file_precision.write('Epoch: ' + str(epoch) + ', Object ID: ' + str(object_id) + ', Precision: ' + str(epochTrainPrecision) + '\n')
        file_recall.write('Epoch: ' + str(epoch) + ', Object ID: ' + str(object_id) + ', Recall: ' + str(epochTrainRecall) + '\n')

        loss_array[object_id].append(epochLoss)
        precision_array[object_id].append(epochTrainPrecision)
        recall_array[object_id].append(epochTrainRecall)

        file_online_loss.flush()
        file_precision.flush()
        file_recall.flush()

        stop_time = timeit.default_timer()

        epoch_secs = stop_time - start_time
        epoch_mins = epoch_secs / 60

        print('This epoch took: ' + str(epoch_mins) + ' minutes')

        torch.save(nets[object_id].state_dict(), os.path.join(save_dir, 'online_training_' + seq_name + '_object_id_' + str(object_id) + 'epoch_' + str(epoch) + '.pth'))

        for param_group in optimizers[object_id].param_groups:
            param_group['lr'] = param_group['lr']*lr_factor_array[epoch-1]

        if os.path.exists(os.path.join(save_dir, 'plots', seq_name, str(object_id))) == False:
            os.makedirs(os.path.join(save_dir, 'plots', seq_name, str(object_id)))

        plot_loss1(loss_array[object_id], 0, epoch, save_dir, online=True, seq_name=seq_name, object_id=object_id)
        plot_loss_minibatch(loss_minibatch_array[object_id], save_dir, online=True, seq_name=seq_name, object_id=object_id)
        plot_precision_recall(precision_array[object_id], recall_array[object_id],nepochs=epoch, save_dir=save_dir, online=True, seq_name=seq_name, object_id=object_id)

if not alreadyTrained:
    file_online_loss.close()
    file_precision.close()
    file_recall.close()

######################################################################################################################
"""Test the fine-tuned network for this particular video"""

print('Testing Network')

epoch_no = nEpochs
num_img_ts = len(anno)
loss_tr = []
aveGrad = 0
output_mask = [None for i in range(0, anno.n_objects + 1)]
large_neg = -1e6

if not os.path.exists(os.path.join(save_dir, seq_name)):
    os.makedirs(os.path.join(save_dir, seq_name))

for ii in range(len(anno)):
    img = apply_val_transform_image(images[ii])
    print('Processing image: ' + str(ii))

    for object_id in range(1, anno.n_objects + 1):

        print('Processing Object ID: ' + str(object_id))

        # convert into single object case
        temp_anno = np.copy(anno[ii])

        temp_anno[temp_anno != object_id] = 0
        temp_anno[temp_anno == object_id] = 1

        gt = apply_val_transform_anno(temp_anno)
        #cv2.imwrite('gt' + str(object_id) + '.png', gt.numpy().squeeze())

        inputs, gts = Variable(img, volatile=True), Variable(gt, volatile=True)

        if torch.cuda.is_available():
            inputs, gts = inputs.cuda(), gts.cuda()

        if ii == 0:
            output_mask[object_id] = gts.clone()
            output_mask[object_id] = output_mask[object_id].unsqueeze(0).float().clone()
            continue

        output_mask[object_id] = output_mask[object_id].clone()

        output_mask[object_id][output_mask[object_id] == 0] = -100
        output_mask[object_id][output_mask[object_id] == 1] = 100
        inputs = inputs.unsqueeze(0)
        input4channels = torch.cat([inputs, output_mask[object_id].detach()], 1)
        noImages, noChannels, height, width = input4channels.shape

        outputs = nets[object_id](input4channels)
        upsampler = torch.nn.Upsample(size=(height, width), mode='bilinear')
        outputs = upsampler(outputs)
        temp_bool = torch.le(outputs[0,0], outputs[0,1])
        outputs.data[0][1][temp_bool.data == 0] = large_neg
        output_mask[object_id] = outputs[0,1]

    if ii==0:
        continue

    #Now we have multiple object masks with scores. Select the max scores and keep them in each mask.

    flag = False
    for object_id in range(1, anno.n_objects+1):
        if flag == False:
            max_mask = output_mask[object_id].clone()
            flag = True
        else:
            max_mask = torch.max(max_mask, output_mask[object_id])

    for object_id in range(1, anno.n_objects+1):

        output_mask[object_id].data[output_mask[object_id].data == large_neg] = 0
        output_mask[object_id].data[max_mask.data == output_mask[object_id].data] = 1
        output_mask[object_id] = output_mask[object_id].unsqueeze(0).unsqueeze(0).float()

    flag1=False

    for object_id in range(1, anno.n_objects+1):

        if flag1 == False:
            final_output = output_mask[object_id].clone()
            flag1=True

        final_output[output_mask[object_id]==1] = object_id

    #Get the palette and attach it

    final_pil_image = Image.fromarray(np.uint8(final_output.data.cpu().numpy()[0][0]))
    final_pil_image.putpalette(anno.color_palette.ravel())
    final_pil_image.save(os.path.join(save_dir, seq_name, str(ii) + '.png'))