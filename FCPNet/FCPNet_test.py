import csv
import os.path

import torch
import logging
# import deepspeed
import numpy as np
import pandas as pd
import seaborn as sns
import torch.optim as optim

from Model.FCPNet import FCPNet
from torch.utils.data import DataLoader
from Utils.Options import FCPNet_args
from Utils.Functions import (HDF5Dataset, copy_paste_patch_3d, copy_paste_patch_3d_position,
                             MutualInformation, setup_seed, save_checkpoint, load_checkpoint)

import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt


def log_to_csv(filename, file1, file2, file3, file4, file5, file6, file7, file8):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([file1, file2, file3, file4, file5, file6, file7, file8])

def calculate_metrics(y_true, y_pred_probs):
    y_pred = np.round(y_pred_probs)
    tp = tn = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1

    sen = tp / (tp + fn) if (tp + fn) != 0 else 0
    spe = tn / (tn + fp) if (tn + fp) != 0 else 0
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    return sen, spe, fpr, tpr, roc_auc

def rocPlot(fpr_thin, tpr_thin, auc_thin, fpr_thick, tpr_thick, auc_thick):
    plt.figure(figsize=(6, 6))
    sns.set(style="whitegrid")
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(fpr_thin, tpr_thin, color='darkorange', lw=2, label=f'Thin-slice (Auc = {auc_thin:.3f})')
    plt.plot(fpr_thick, tpr_thick, color='blue', lw=2, label=f'Thick-slice (Auc = {auc_thick:.3f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(loc="lower right", fontsize=17)

    save_path = os.path.join(args.test_save, 'roc.png')
    plt.savefig(save_path)

    plt.show()

def test(args, model, t_data_loader):
    model.eval()
    cor_thin, cor_thick, total = 0, 0, 0
    (all_labels, all_thinPred_probs, all_thickPred_probs,
     all_thinPred_labels, all_thickPred_labels) = [], [], [], [], []

    with torch.no_grad():
        for img_ori, _, egfr_label, img_thick in t_data_loader:
            egfr_label, img_thin, img_thick = (
            egfr_label.long().to(device), img_ori.float().to(device), img_thick.float().to(device))

            _, _, _, _, sig_thin = model(img_thin.to(device), img_thin.to(device),
                                         img_thin.to(device), img_thin.to(device))
            _, _, _, _, sig_thick = model(img_thick.to(device), img_thick.to(device),
                                          img_thick.to(device), img_thick.to(device))
            _, pred_thin = torch.max(sig_thin.data, 1)
            _, pred_thick = torch.max(sig_thick.data, 1)

            total += egfr_label.size(0)
            cor_thin += (pred_thin == egfr_label).sum().item()
            cor_thick += (pred_thick == egfr_label).sum().item()

            all_labels.extend(egfr_label.cpu().numpy())
            all_thinPred_probs.extend(sig_thin.cpu().numpy()[:, 1])
            all_thinPred_labels.extend(pred_thin.cpu().numpy())

            all_thickPred_probs.extend(sig_thick.cpu().numpy()[:, 1])
            all_thickPred_labels.extend(pred_thick.cpu().numpy())

        acc_thin, acc_thick = cor_thin / total * 100, cor_thick / total * 100
        sen_thin, spe_thin, fpr_thin, tpr_thin, auc_thin = calculate_metrics(all_labels, all_thinPred_probs)
        sen_thick, spe_thick, fpr_thick, tpr_thick, auc_thick = calculate_metrics(all_labels, all_thickPred_probs)

    results_df = pd.DataFrame({
        'True Labels': all_labels,
        'Pred Thin Probs': all_thinPred_probs,
        'Pred Thin labels': all_thinPred_labels,
        'Pred Thick Probs': all_thickPred_probs,
        'Pred Thick labels': all_thickPred_labels,
    })
    results_df.to_csv(os.path.join(args.test_save, 'details.csv'), index=False)

    return (acc_thin, auc_thin, sen_thin, spe_thin, fpr_thin, tpr_thin,
            acc_thick, auc_thick, sen_thick, spe_thick, fpr_thick, tpr_thick)


if __name__ == "__main__":
    # args selected
    args = Dense_args()
    setup_seed(args.seed)
    gpu_ids = list(map(int, args.gpus.split(',')))
    if not os.path.exists(args.test_save):
        os.makedirs(args.test_save)

    model = FCPNet().float()

    if args.multi_gpu:
        device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        if len(gpu_ids) > 1:
            print(f"Using GPUs: {gpu_ids}")
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.load_state_dict(torch.load(args.test_model))

    t_dataset = HDF5Dataset(args.data_path, 'test', normalization='minmax')
    t_data_loader = DataLoader(t_dataset, batch_size=args.batch_size, shuffle=False)

    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        filename=args.test_save + '/{}.txt'.format(os.path.basename(args.test_save)),
                        filemode='a')

    logging.info(f'''
                        ##################{os.path.basename(args.test_model) + ' ----- test started'}##################
                        ''')

    (acc_thin, auc_thin, sen_thin, spe_thin, fpr_thin, tpr_thin,
     acc_thick, auc_thick, sen_thick, spe_thick, fpr_thick, tpr_thick) = test(args, model, t_data_loader)

    print('thin || acc:{:.4f} ## auc:{:.4f} ## sen:{:.4f} ## spe:{:.4f} '.format(acc_thin, auc_thin, sen_thin, spe_thin))
    print('thick || acc:{:.4f} ## auc:{:.4f} ## sen:{:.4f} ## spe:{:.4f} '.format(acc_thick, auc_thick, sen_thick, spe_thick))

    log_to_csv(os.path.join(args.test_save, 'test_plog.csv'),
               acc_thin, sen_thin, spe_thin, auc_thin, acc_thick, sen_thick, spe_thick, auc_thick)

    rocPlot(fpr_thin, tpr_thin, auc_thin, fpr_thick, tpr_thick, auc_thick)

    logging.info(f'''
                           Dataset:         {os.path.basename(args.data_path)}
                           Batch size:      {args.batch_size}
                           Multi GPUs       {args.multi_gpu}
                           Devices          {args.gpus}
                           Device:          {args.gpu}
                           Test Path:       {args.test_model}
                           Acc_thin:        {acc_thin:.4f}
                           Sen_thin:        {sen_thin:.4f}
                           Spe_thin:        {spe_thin:.4f}
                           Auc_thin:        {auc_thin:.4f}
                           Acc_thick:       {acc_thick:.4f}
                           Sen_thick:       {sen_thick:.4f}
                           Spe_thick:       {spe_thick:.4f}
                           Auc_thick:       {auc_thick:.4f}                           
                       ''')

