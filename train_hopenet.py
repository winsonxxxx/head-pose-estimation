# convert PyTorch hopenet scripts to MXNet

from mxboard import SummaryWriter
import sys, os, argparse, time
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.autograd import record
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
import mxnet.gluon.data.vision
import datasets
import mxnet.gluon.model_zoo as model_zoo
import pickle as pk
from mxnet import ndarray as nd
#from mxnet.gluon import HybridBlock
import datetime
import hopenet

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=32, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='E:\\face-dataset\\300W_LP', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='E:\\face-dataset\\300W_LP\\filename_list.txt', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=0.001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()
    return args

    
def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def reg_criterion(A, B):
    mse = ((A - B)**2).mean()
    return mse

if __name__ == '__main__':
    args = parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    lr = args.lr
    #ctx = mx.cpu()
    ctx = mx.gpu(gpu)

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    model = hopenet.Hopenet(model_zoo.vision.BottleneckV1, [3, 4, 6, 3], 66)
    
    # ResNet50 structure
    model.hybridize()
    
    print('Loading data.')
    transformations = transforms.Compose([transforms.Resize(240),
            transforms.RandomResizedCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
                                      
    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Synhead':
        pose_dataset = datasets.Synhead(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()

    train_loader = gluon.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    #reg_criterion = gluon.loss.L2Loss()
    alpha = args.alpha
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = np.float32(idx_tensor)

    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})
    #optimizer = gluon.Trainer(gluon.ParameterDict(lst), 'Adam', {'learning_rate': lr})
    
    unique_id = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.','-')
    with SummaryWriter(logdir='./data/' + unique_id + '/logs', flush_secs=5) as sw:
        try:
            global_step = 0
            for epoch in range(num_epochs):
                thefile = open('./data/' + unique_id + '//test'+'-'+str(epoch)+'.txt', 'w')
                for batch_idx, (images, labels, cont_labels) in enumerate(train_loader):
                    global_step += 1
                    images = images.as_in_context(ctx)

                    # Binned labels
                    labels.as_in_context(ctx)
                    label_yaw = labels[:,0].as_in_context(ctx)
                    label_pitch = labels[:,1].as_in_context(ctx)
                    label_roll = labels[:,2].as_in_context(ctx)

                    # Continuous labels
                    cont_labels = cont_labels.as_in_context(ctx)
                    label_yaw_cont = cont_labels[:,0].as_in_context(ctx)
                    label_pitch_cont = cont_labels[:,1].as_in_context(ctx)
                    label_roll_cont = cont_labels[:,2].as_in_context(ctx)

                    # Forward pass
                    yaw, pitch, roll = model(images)

                    yaw.attach_grad()
                    pitch.attach_grad()
                    roll.attach_grad()
                    with mx.autograd.record():
                        # Cross entropy loss
                        loss_yaw = criterion(yaw, label_yaw)
                        loss_pitch = criterion(pitch, label_pitch)
                        loss_roll = criterion(roll, label_roll)
                        # MSE loss
                        yaw_predicted = softmax(yaw.asnumpy(),axis=1)
                        pitch_predicted = softmax(pitch.asnumpy(),axis=1)
                        roll_predicted = softmax(roll.asnumpy(),axis=1)
                        #yaw_predicted = nd.softmax(yaw, axis=1)  
                        yaw_predicted = np.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
                        pitch_predicted = np.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
                        roll_predicted = np.sum(roll_predicted * idx_tensor, 1) * 3 - 99
                        loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont.asnumpy())
                        loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont.asnumpy())
                        loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont.asnumpy())
                        loss_yaw = loss_yaw + alpha * loss_reg_yaw
                        loss_pitch = loss_pitch + alpha * loss_reg_pitch
                        loss_roll = loss_roll + alpha * loss_reg_roll
                        loss_seq = [loss_yaw.as_in_context(ctx), loss_pitch.as_in_context(ctx), loss_roll.as_in_context(ctx)]
                        if epoch == 0 and batch_idx == 0:
                            sw.add_graph(model)
                    # if global_step % 10 == 0:
                    #     sw.add_scalar(tag='Log10_of_loss_yaw',value=loss_yaw.log10().asscalar(), global_step=global_step)
                    #     sw.add_scalar(tag='epoch', value=epoch, global_step=global_step)
                    loss_yaw_sum = loss_yaw.mean().asscalar()
                    loss_pitch_sum = loss_pitch.mean().asscalar()
                    loss_roll_sum = loss_roll.mean().asscalar()
                    mx.autograd.backward(loss_seq)
                    optimizer.step(batch_size,ignore_stale_grad=True)    
                    thefile.write("Epoch: %d; Batch %d; Loss Yaw %f; Loss Pitch %f; Loss Roll %f \n" % (epoch, batch_idx, loss_yaw_sum, loss_pitch_sum, loss_roll_sum))
                    print("Epoch: %d; Batch %d; Loss Yaw %f; Loss Pitch %f; Loss Roll %f" % (epoch, batch_idx, loss_yaw_sum, loss_pitch_sum, loss_roll_sum))
                thefile.close()
                model.export('./data/' + unique_id + '/hopenet-'+str(epoch))
                sw.close()
        except KeyboardInterrupt:
            print("KeyboardInterrupted")
            sw.close()
            exit(0)
            pass