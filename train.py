import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
import archs
import losses
from dataset import Dataset
from utils import AverageMeter, str2bool
import cv2

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--train_dataset', default=None,
                        help='dataset name')
    parser.add_argument('--val_dataset', default=None,
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)
    
    ###ADD argument to resume training
    parser.add_argument('--saved_model', default=None, type=str)
    ###
    
    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                 'acc': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()


        output = model(input)
        loss = criterion(output, target)
        _, preds = output.max(1)

        accuracy = (target == preds).float().mean()

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['acc'].update(accuracy.item(), input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('acc', avg_meters['acc'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                       ('acc', avg_meters['acc'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                 'acc': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            _, preds = output.max(1)
            accuracy = (target == preds).float().mean()
            
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['acc'].update(accuracy.item(), input.size(0))
            
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('acc', avg_meters['acc'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)
            
        pbar.close()
        insave0 = input[-1,:,:,:].cpu().numpy()
        insave0 = insave0.transpose(1,2,0)
        cv2.imwrite("pred_img.png", insave0)
        print(target.shape)
        targsave0 = target[-1,0,:,:].cpu.numpy()
        targsave1 = target[-1,1,:,:].cpu.numpy()
        targsave2 = target[-1,2,:,:].cpu.numpy()
        np.savetxt('bldg-targ.npy', targsave0)
        np.savetxt('back-targ.npy', targsave1)
        np.savetxt('out-targ.npy', targsave2)
 
        outsave0 = torch.sigmoid(output[-1,0,:,:]).cpu().numpy()
        outsave1 = torch.sigmoid(output[-1,1,:,:]).cpu().numpy()
        outsave2 = torch.sigmoid(output[-1,2,:,:]).cpu().numpy()
        np.savetxt('bldg.npy', outsave0)
        np.savetxt('back.npy', outsave1)
        np.savetxt('out.npy', outsave2)
        
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('acc', avg_meters['acc'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['train_dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['train_dataset'], config['arch'])
    
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)
        
    # weight given to loss for pixels of background, building interior and building border classes
    loss_weights = torch.tensor([0.1, 0.8, 0.1])
    criterion = nn.CrossEntropyLoss(weight=loss_weights).cuda()
    
    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    #Resume training by loading model
    if config['saved_model'] != None:
        model.load_state_dict(torch.load(config['saved_model']))
    
    # Data loading code
    train_img_ids = glob(os.path.join(config['train_dataset'], 'images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
    
    val_img_ids = glob(os.path.join(config['val_dataset'], 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    train_transform = Compose([
        transforms.RandomCrop(config['input_h'], config['input_w']),
        transforms.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.5, rotate_limit=180, p=0.5),
        transforms.Flip(),
    ])

    val_transform = Compose([
        transforms.RandomCrop(config['input_h'], config['input_w']),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['train_dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['train_dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['val_dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['val_dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('acc', []),
        ('val_loss', []),
        ('val_acc',[])
        ])

    best_acc = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f'
              % (train_log['loss'],
                 train_log['acc'],
                 val_log['loss'],
                 val_log['acc']
                ))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['acc'].append(train_log['acc'])
        log['val_loss'].append(val_log['loss'])
        log['val_acc'].append(val_log['acc'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['acc'] > best_acc:
            torch.save(model.state_dict(), '/lfs/jonas/unetplus/model.pth')
            best_acc = val_log['acc']
            print("=> saved best model")
            f = open('/lfs/jonas/unetplus/model_info.txt', 'a')
            f.write('Epoch: %i, Loss: %f, Acc: %f' % (epoch, val_log['loss'], val_log['acc']))
            f.close()
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
