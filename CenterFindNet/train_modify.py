import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.dataset import ycb_dataset
from lib.centroid_prediction_network_modify import CentroidPredictionNetwork_modify
import matplotlib.pyplot as plt
from torchsummary import summary
from contextlib import redirect_stdout

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root_dir', type=str, default = '', help='dataset root dir YCB_Video_Dataset')
parser.add_argument('--batch_size', type=int, default = 64, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.1, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.001, help='margin to decay lr & w')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--nepoch', type=int, default=51, help='max number of epochs to train')
parser.add_argument('--resume_CPN', type=str, default = '',  help='resume CPN model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    opt.num_points = 1000 #number of points on the input pointcloud
    opt.outf = 'CenterFindNet/trained_model/test' #folder to save trained models
    opt.weight_decay = 0.01
    
    estimator = CentroidPredictionNetwork_modify(num_points = opt.num_points)
    estimator.cuda()
    with open('CenterFindNet/trained_model/test/model_structure.txt', 'wt+') as file:
        with redirect_stdout(file):
            input1 = torch.randn(1, opt.num_points, 3).cuda()
            input2 = torch.randn(1, opt.num_points, 3).cuda()
            summary(estimator, input1, input2)
    file.close()

    if opt.resume_CPN != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_CPN)))

    opt.decay_start = False
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    # optimizer = optim.SGD(estimator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    dataset = ycb_dataset('train', opt.num_points, True, opt.dataset_root_dir, opt.noise_trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    test_dataset = ycb_dataset('test', opt.num_points, False, opt.dataset_root_dir, 0.0)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    criterion = nn.MSELoss()
    best_test = np.Inf
    st_time = time.time()

    train_losses = []
    test_losses = []

    with open('CenterFindNet/trained_model/test/log_{0}.txt'.format(time.strftime("%Hh %Mm %Ss", time.localtime(time.time()))), 'wt+') as log_file:
        for epoch in range(opt.start_epoch, opt.nepoch):
            print('learning rate ' ,optimizer.param_groups[0]['lr'], file=log_file)
            print('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
            print('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'), file=log_file)
            train_count = 0
            train_dis = 0.0
            estimator.train()
            optimizer.zero_grad()

            for i, data in enumerate(dataloader, 0):
                points, model_points, gt_centroid = data
                points, model_points, gt_centroid = Variable(points).cuda(), Variable(model_points).cuda(), Variable(gt_centroid).cuda()
                centroid = estimator(points, model_points)

                loss = criterion(centroid, gt_centroid)
                loss.backward()

                train_dis += loss.item()
                train_count += 1

                print('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, train_count, train_count * opt.batch_size, loss.item()))
                if train_count % 1000 == 0:
                    print('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_loss:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, train_count, train_count * opt.batch_size, loss.item()), file=log_file)
                optimizer.step()
                optimizer.zero_grad()
                

                if train_count != 0 and train_count % 1000 == 0:
                    torch.save(estimator.state_dict(), '{0}/CPN_model_current.pth'.format(opt.outf))
            train_dis_avg = train_dis / train_count
            train_losses.append(train_dis_avg)
            print('Train time {0} Epoch {1} TRAIN FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, train_dis_avg))
            print('Train time {0} Epoch {1} TRAIN FINISH Avg loss: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, train_dis_avg), file=log_file)
            # print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))
            # print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch), file=log_file)

            print('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
            print('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'), file=log_file)
            

            test_dis = 0.0
            test_count = 0
            estimator.eval()

            for j, data in enumerate(testdataloader, 0):
                points, model_points, gt_centroid = data
                points, model_points, gt_centroid = Variable(points).cuda(), Variable(model_points).cuda(), Variable(gt_centroid).cuda()
                centroid = estimator(points, model_points)
                loss = criterion(centroid, gt_centroid)

                test_dis += loss.item()
                print('Test time {0} Test Batch No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss))

                test_count += 1

            test_dis = test_dis / test_count
            print('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
            print('Test time {0} Epoch {1} TEST FINISH Avg loss: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis), file=log_file)
            test_losses.append(test_dis)
            if test_dis <= best_test:
                best_test = test_dis
                torch.save(estimator.state_dict(), '{0}/CPN_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
                print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
                print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<', file=log_file)
            if best_test < opt.decay_margin and not opt.decay_start:
                opt.decay_start = True
                opt.lr *= opt.lr_rate
                optimizer = optim.Adam(estimator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    log_file.close()

    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
