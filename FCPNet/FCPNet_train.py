import csv
import random
import os.path

import torch
import logging
# import deepspeed
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim

from Model.FCPNet import FCPNet
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from Utils.Options import FCPNet_args
from Utils.Functions import (HDF5Dataset, copy_paste_patch_3d, copy_paste_patch_3d_position,
                             MutualInformation, setup_seed, save_checkpoint, load_checkpoint)


def log_to_csv(filename, file1, file2, file3, file4, file5, file6, file7, file8):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([file1, file2, file3, file4, file5, file6, file7, file8])

def test(model, device, test_dataloader):
    model.eval()
    cor_thin, cor_thick, total = 0, 0, 0
    all_labels, all_preds_thin, all_preds_thick= [], [], []
    with (torch.no_grad()):
        for img_thin, img_thick, egfr_label in test_dataloader:
            egfr_label, img_thin, image_thick = (egfr_label.long().to(device),img_thin.float().to(device),
                                                    img_thick.float().to(device))

            _, _, _, _, sig_thin = model(img_thin.to(device), img_thin.to(device),
                                         img_thin.to(device), img_thin.to(device))
            _, pred_thin = torch.max(sig_thin.data, 1)

            _, _, _, _, sig_thick = model(image_thick.to(device), image_thick.to(device),
                                         image_thick.to(device), image_thick.to(device))
            _, pred_thick = torch.max(sig_thick.data, 1)

            total += egfr_label.size(0)
            cor_thin += (pred_thin == egfr_label).sum().item()
            cor_thick += (pred_thick == egfr_label).sum().item()

            all_labels.extend(egfr_label.cpu().numpy())
            all_preds_thin.extend(sig_thin.cpu().numpy()[:, 1])
            all_preds_thick.extend(sig_thick.cpu().numpy()[:, 1])

        acc_thin, acc_thick = cor_thin / total * 100, cor_thick / total * 100
        auc_thin, auc_thick = roc_auc_score(all_labels, all_preds_thin), roc_auc_score(all_labels, all_preds_thick)
        return acc_thin, acc_thick, auc_thin, auc_thick


def train(model, device, optimizer, train_loader):
    epoch_loss = 0.
    correct, samples = 0, 0
    all_labels, all_preds = [], []
    for iter, (img_thin, img_thick, egfr_label, _) in enumerate(train_loader):
        egfr_label, img_thin, img_thick = (egfr_label.long().to(device),
                                        img_thin.float().to(device), img_thick.float().to(device))

        cp_thin, cp_thick, patch_position = copy_paste_patch_3d(img_thin, img_thick, cp_patch)

        f_thin, f_thick, f_cp_thin, f_cp_thick, sig = model(img_thin.to(device), img_thick.to(device),
                                                            cp_thin.to(device), cp_thick.to(device))

        cp_f_thin, cp_f_thick = copy_paste_patch_3d_position(f_cp_thin, f_cp_thick, patch_position)

        floss_1 = fea_loss(f_thin, cp_f_thin, device)
        floss_2 = fea_loss(f_thick, cp_f_thick, device)
        ploss = loss_criterion(sig, egfr_label)

        loss = (floss_1 + floss_2) + ploss

        epoch_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Iteration [{}/{}], Floss1: {:.4f}, Floss2: {:.4f}, Ploss: {:.4f}, Loss: {:.4f}'
              .format(epoch, total_epoch, iter + 1, len(train_loader), floss_1.item(), floss_2.item(), ploss.item(), loss.item()))

        _, predicted = torch.max(sig.data, 1)
        samples += egfr_label.size(0)
        correct += (predicted == egfr_label).sum().item()

        all_labels.extend(egfr_label.cpu().numpy())
        all_preds.extend(sig.cpu().detach().numpy()[:, 1])

    '''acc ## auc ## epoch_loss'''
    train_acc = correct / samples * 100
    train_auc = roc_auc_score(all_labels, all_preds)
    average_loss = epoch_loss / len(train_loader)

    acc_thin, acc_thick, auc_thin, auc_thick = test(model, device, val_loader)
    return train_acc, train_auc, acc_thin, acc_thick, auc_thin, auc_thick, average_loss


if __name__ == "__main__":
    args = FCPNet_args()
    setup_seed(args.seed)

    multi_gpu = args.multi_gpu
    gpu_ids = list(map(int, args.gpus.split(',')))
    gpu = args.gpu
    opt = args.opt
    data_dir = args.data_path
    batch_size = args.batch_size
    base_lr = args.base_lr
    pre_epoch = args.pre_epoch
    total_epoch = args.total_epoch
    check_path = args.check_path
    cp_patch = args.cp_patch

    save_dir = args.save
    logging_dir = args.save
    check_save = args.check_save
    model_save = args.model_save

    # deepspeed_config = {
    #     "train_batch_size": batch_size,
    #     "fp16": {
    #         "enabled": enabled
    #     },
    #     "zero_optimization": {
    #         "stage": 1,
    #     }
    # }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(check_save):
        os.makedirs(check_save)
    if not os.path.exists(model_save):
        os.makedirs(model_save)


    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        filename=logging_dir + '/{}.txt'.format(args.model),
                        filemode='a')

    model = FCPNet().float()

    if multi_gpu:
        device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        if len(gpu_ids) > 1:
            print(f"Using GPUs: {gpu_ids}")
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    if opt == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0005)

    # model, optimizer, _, _ = deepspeed.initialize(
    #     model=model,
    #     optimizer=optimizer,
    #     config=deepspeed_config
    # )

    train_dataset = HDF5Dataset(data_dir, 'train', normalization='minmax', traAug=True)
    val_dataset = HDF5Dataset(data_dir, 'val', normalization='minmax')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    logging.info(f'''
            ##################{args.model + ' ----- training started'}##################
            ''')
    loss_criterion = torch.nn.CrossEntropyLoss()

    if args.loss == 'mi':
        fea_loss = MutualInformation()
    if args.loss == 'ncc':
        fea_loss = NCC()
    if args.loss == 'mse':
        fea_loss = torch.nn.MSELoss()

    # checkpoint loading
    if check_path != None:
        pre_epoch = load_checkpoint(model, optimizer, check_path)

    best_auc, best_log = 0.0, ''

    for epoch in range(pre_epoch, total_epoch):
        train_acc, train_auc, acc_thin, acc_thick, auc_thin, auc_thick, average_loss \
            = train(model=model, device=device, optimizer=optimizer, train_loader=train_loader)

        logging.info(
            'epoch:{} # train_loss:{:.4f} # train_acc:{:.4f} # train_auc:{:.4f} # val_acc_thin:{:.4f} '
            '# val_acc_thick:{:.4f} # val_auc_thin:{:.4f} # val_auc_thick:{:.4f}'
            .format(epoch, average_loss, train_acc, train_auc, acc_thin, acc_thick, auc_thin, auc_thick))

        log_to_csv(os.path.join(args.save, 'csv_log.csv'),
                   epoch, average_loss.item(), train_acc, train_auc, acc_thin, acc_thick, auc_thin, auc_thick)

        if auc_thin + auc_thick / 2 > best_auc:
            best_auc = auc_thin + auc_thick / 2
            best_log = str(epoch) + '_' + f'{auc_thin:.4f}' + '_' + f'{auc_thick:.4f}'

        if epoch > args.model_flag:
            save_model = str(epoch) + '_' + f'{auc_thin:.4f}' + '_' + f'{auc_thick:.4f}' + '_model.pth'
            model_dir = os.path.join(args.model_save, save_model)
            torch.save(model.state_dict(), model_dir)

        if epoch > args.check_flag:
            save_checkpoint(epoch, model, optimizer, args.check_save,
                            filename=f'checkpoint_epoch_{epoch}.pth')

    logging.info('best epoch | auc : {}'.format(best_log))

    logging.info(f'''
            Net:             {args.model}
            Dataset:         {os.path.basename(data_dir)}
            Epochs:          {args.total_epoch}
            Batch size:      {args.batch_size}
            Learning rate:   {args.base_lr}
            Optimizer:       {args.opt}
            Multi GPUs       {args.multi_gpu}
            Devices          {args.gpus}
            Device:          {args.gpu}
        ''')





