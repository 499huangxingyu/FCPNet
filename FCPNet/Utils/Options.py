import argparse

def FCPNet_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FCPNet', help='model_name')
    parser.add_argument('--pre_epoch', type=int, default=0, help='maximum epoch number to train')
    parser.add_argument('--total_epoch', type=int, default=60, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--opt', type=str, default='SGD', help='opt modules')
    # data
    parser.add_argument('--data_path', type=str, default="your data files", help='data path')
    # costs
    parser.add_argument('--multi_gpu', type=bool, default=True, help='Multi GPU to use')
    parser.add_argument('--gpus', type=str, default='2,3', help='Multi GPU to use')
    parser.add_argument('--gpu', type=str, default='cuda:2', help='GPU to use')
    parser.add_argument('--loss', type=str, default='mi', help='Losses to use')
    parser.add_argument('--cp_patch', type=lambda s: tuple(map(int, s.split(','))), default=(180, 180),
                        help='cp size as a tuple (width, height)')

    # save
    parser.add_argument('--model_flag', type=int, default=20, help='save model with epoch')
    parser.add_argument('--check_flag', type=int, default=20, help='save checkpoint with epoch')
    parser.add_argument('--save', type=str, default='./Result/', help='Result saving')
    parser.add_argument('--check_save', type=str, default='./Result/checkpoint', help='checkpoint saving')
    parser.add_argument('--model_save', type=str, default='./Result/model', help='model saving')
    parser.add_argument('--test_save', type=str, default='./Result/test', help='model saving')
    parser.add_argument('--check_path', type=str, default=None, help='checkpoint path')
    # test
    parser.add_argument('--test_model', type=str, default='your models', help='saved test model')
    args = parser.parse_args()
    return args

