import os
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

#from vgg import Vgg16BnCifar
from vgg_nobn import Vgg16BnCifar
import augment as aug


def load_model(model, path, strict=False):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module'
        else:
            name = key
        new_state_dict[name] = val
    model.load_state_dict(new_state_dict, strict=strict)
    all_keys = set(new_state_dict.keys())
    actual_keys = set(model.state_dict().keys())
    missing_keys = actual_keys - all_keys
    for k in missing_keys:
        print(k)


def train(train_dataloader, dev_dataloader, model, optimizer, lr_scheduler, args):
    model.train()
    if args.loss_function == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError
    batch_size = args.batch_size
    print('batch_size: {}'.format(batch_size))
    log_f_name = '%s/log.txt' % (args.output_dir)
    log_f = open(log_f_name, "w")

    list_loss = []
    list_accuracy = []
    for epoch in range(args.epoch):
        lr_scheduler.step()
        count = 0
        total_loss = 0.0
        total_accuracy = 0.0
        for idx, data in enumerate(train_dataloader):
            feature, label = data
            feature, label = feature.cuda(), label.cuda()
            cur_length = label.shape[0]
            output = model(feature)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_choice = output.data.max(dim=1)[1]
            correct = output_choice.eq(label).sum().cpu().numpy()
            count += 1
            total_loss += loss.item()
            total_accuracy += correct * 1.0 / cur_length
            print('[epoch%d: batch%d], train loss: %f, accuracy: %f' % (epoch, idx, loss.item(), correct * 1.0 / cur_length))
        list_loss.append(total_loss/count)
        list_accuracy.append(total_accuracy/count)
        print('[epoch%d], avg train loss: %f, avg accuracy: %f' % (epoch, total_loss/count, total_accuracy/count), file=log_f)
        print('[epoch%d], avg train loss: %f, avg accuracy: %f' % (epoch, total_loss/count, total_accuracy/count))

        if epoch % 3 == 0:
            torch.save(model.state_dict(), "%s/epoch_%d.pth" % (args.output_dir, epoch))
    list_loss_np = np.array(list_loss)
    list_accuracy_np = np.array(list_accuracy)
    np.save('%s/loss' % (args.output_dir), list_loss_np)
    np.save('%s/accuracy' % (args.output_dir), list_accuracy_np)
    log_f.close()


def validate(test_dataloader, model):
    total_correct = 0
    total = 0
    for idx, data in enumerate(test_dataloader):
        feature, label = data
        feature, label = feature.cuda(), label.cuda()
        cur_length = label.shape[0]
        output = model(feature)

        output_choice = output.data.max(dim=1)[1]
        correct = output_choice.eq(label).sum().cpu().numpy()
        total += cur_length
        total_correct += correct
    print('test accuracy: %f' % (total_correct * 1.0 / total))

def main(args):
    if args.model == 'vgg16_bn':
        model = Vgg16BnCifar(in_channels=3, num_classes=10)
    else:
        raise ValueError
    model.cuda()

    if args.optim == 'SGD_momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    else:
        raise ValueError

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 95], gamma=0.1)
    if args.load_path:
        if args.recover:
            load_model(model, args.load_path, strict=True)
            print('load model state dict in {}'.format(args.load_path))

    transform_dict = {'Affine': aug.AffineTransform(done_p=0.5),
                      'Salt': aug.SaltPepperNoise(p=0.05, done_p=0.5),
                      'Blur': aug.Blur((7, 7), 2.0, done_p=0.5)}
    augments = []
    augments.append(transforms.ToTensor())
    for key in args.augment:
        if key in transform_dict:
            augments.append(transform_dict[key])
    augments.append(transforms.Normalize(mean=(120.707/255, 120.707/255, 120.707/255), std=(64.15/255, 64.15/255, 64.15/255)))
    transform = transforms.Compose(augments)
    train_set = torchvision.datasets.CIFAR10(args.root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(args.root, train=False, download=True, transform=transform)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.evaluate:
        validate(test_dataloader, model)
        return

    train(train_dataloader, test_dataloader, model, optimizer, lr_scheduler, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='optimization algorithm experiment')
    parser.add_argument('--root', default='./')
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--recover', default=False, type=bool)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--optim', default='SGD_momentum', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--model', default='vgg16_bn', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--loss_function', default='CrossEntropy', type=str)
    #parser.add_argument('--augment', default=['Affine', 'Salt', 'Blur'])
    parser.add_argument('--augment', default=['Salt', 'Blur'])
    parser.add_argument('--output_dir', default='./experiment/SGD_momentum_noaffine', type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print(args)
    main(args)
