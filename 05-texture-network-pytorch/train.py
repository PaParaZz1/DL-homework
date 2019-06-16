#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : train.py
# Purpose : run a session for training
# Creation Date : 2019-04-10
# Last Modified :
# Created By : niuyazhe
# =======================================


import os
import argparse
import torch
import numpy as np
import cv2
from torch.optim import Adam
from vgg import vgg16_bn
from loss import GramLoss


def load_img(path, img_shape=(224, 224)):
    img = cv2.imread(path)
    img = cv2.resize(img, img_shape)
    img = (img/255).astype(np.float32)
    return img


def sigmoid(x):
    return (1./(1+np.exp(-x)))


def output_img(img, name):
    assert(isinstance(img, np.ndarray))
    img = (img*255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(name, img)


def extract_feature(model, inputs, layer_list):
    feature_map_list = []
    handle_list = []

    def hook(module, input_feature, output_feature):
        feature_map_list.append(output_feature)

    for name in layer_list:
        layer = model
        for item in name:
            layer = getattr(layer, item)
        handle = layer.register_forward_hook(hook)
        handle_list.append(handle)
    _ = model(inputs)
    for item in handle_list:
        item.remove()
    return feature_map_list


def train(args):
    log_dir = os.path.join(args.output_dir, "L{}_W{}_LOSS{}".format(
        '_'.join([item[1] for item in args.layer_list]),
        '_'.join([str(item) for item in args.weight_layer]),
        args.loss_type)
    )
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = open(os.path.join(log_dir, "log.txt"), "w")

    model = vgg16_bn(pretrained=True)
    model.cuda()
    model.eval()

    img = load_img(args.img_path, args.img_shape)
    pre_noise = np.random.uniform(
        low=-3, high=3, size=img.shape).astype(np.float32)
    pre_noise = sigmoid(pre_noise)
    img_tensor = torch.from_numpy(img).permute(
        2, 0, 1).contiguous().unsqueeze(0).cuda()
    noise_tensor = torch.from_numpy(pre_noise).permute(
        2, 0, 1).contiguous().unsqueeze(0).cuda()
    noise_tensor.requires_grad_(True)

    criterion = GramLoss(args.weight_layer, dist_type=args.loss_type)

    def lr_func(epoch):
        lr_factor = args.lr_factor_dict
        lr_key = list(lr_factor.keys())
        index = 0
        for i in range(len(lr_key)):
            if epoch < lr_key[i]:
                break
            else:
                index = i
        return lr_factor[lr_key[index]]

    optimizer = Adam([noise_tensor], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    for epoch in range(args.epoch):
        scheduler.step()
        img_output = extract_feature(model, img_tensor, args.layer_list)
        noise_output = extract_feature(model, noise_tensor, args.layer_list)

        data = list(zip(noise_output, img_output))
        loss = criterion(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.show_interval == 0:
            print("e:{}---loss:{:.6f}".format(epoch, loss.item()))
            print("e:{}---loss:{:.6f}".format(epoch, loss.item()), file=log_file)
        if epoch % args.save_interval == 0:
            noise_np = noise_tensor.data.cpu().squeeze(
                0).permute(1, 2, 0).contiguous().numpy()
            output_img(noise_np, os.path.join(
                log_dir, "epoch_{}.png".format(epoch)))
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='experiment/')
    parser.add_argument('--img_path', default='images/red-peppers256.jpg')
    parser.add_argument('--img_shape', default=(224, 224))
    parser.add_argument('--loss_type', default='earth_move')
    parser.add_argument('--lr', default=1e-2)
    parser.add_argument('--lr_factor_dict', default={0: 1, 200: 0.5, 1500: 0.1})
    parser.add_argument('--weight_layer', default=[1.0 for x in range(13)])
    parser.add_argument('--epoch', default=5000)
    parser.add_argument('--save_interval', default=10)
    parser.add_argument('--show_interval', default=10)
    parser.add_argument('--layer_list', default=[('features', '2'),
                                                 ('features', '5'),
                                                 ('features', '9'),
                                                 ('features', '12'),
                                                 ('features', '16'),
                                                 ('features', '19'),
                                                 ('features', '22'),
                                                 ('features', '26'),
                                                 ('features', '29'),
                                                 ('features', '32'),
                                                 ('features', '36'),
                                                 ('features', '39'),
                                                 ('features', '42')])
    args = parser.parse_args()
    train(args)
