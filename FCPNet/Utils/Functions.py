import io
import os
import random
import h5py
import numpy as np
import torch
import paramiko
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F
from scipy.ndimage import median_filter, gaussian_filter, uniform_filter
from torch.autograd import Variable
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped,
    RandFlipd, RandRotate90d, RandAffined
)

def load_4D(file_path):
    # Load 3D image and convert to 4D (if necessary)
    image = nib.load(file_path).get_fdata()
    return image

def imgnorm(img):
    # Normalize the image
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def copy_paste_patch_3d(A, B, patch_size):
    """
    在 A 和 B 中进行 2D patch 交换操作
    :param A: 3D 数据张量 A，大小为 [1, 长, 宽, 切片]
    :param B: 3D 数据张量 B，大小为 [1, 长, 宽, 切片]
    :param patch_size: patch 的大小，[长, 宽]
    :return: 交换后的 A 和 B 以及进行交换的位置
    """
    cp_a = A.clone()
    cp_b = B.clone()

    _, C, L, W, D = A.shape
    p_l, p_w = patch_size

    # 计算随机的起始位置
    start_l = np.random.randint(0, L - p_l)
    start_w = np.random.randint(0, W - p_w)

    # 提取 patch 在所有切片上的区域
    patch_A = cp_a[:, :, start_l:start_l + p_l, start_w:start_w + p_w, :].clone()
    patch_B = cp_b[:, :, start_l:start_l + p_l, start_w:start_w + p_w, :].clone()

    # 交换 patch
    cp_a[:, :, start_l:start_l + p_l, start_w:start_w + p_w, :] = patch_B
    cp_b[:, :, start_l:start_l + p_l, start_w:start_w + p_w, :] = patch_A

    return cp_a, cp_b, (start_l, start_w, p_l, p_w)

def copy_paste_patch_3d_position(A, B, patch_position):
    """
    在 A 和 B 中进行 2D patch 交换操作，使用固定的位置
    :param A: 3D 数据张量 A，大小为 [1, 长, 宽, 切片]
    :param B: 3D 数据张量 B，大小为 [1, 长, 宽, 切片]
    :param patch_size: patch 的大小，[长, 宽]
    :param patch_position: patch 的位置，[start_l, start_w, p_l, p_w]
    :return: 交换后的 A 和 B
    """
    cp_a = A.clone()
    cp_b = B.clone()

    start_l, start_w, p_l, p_w = patch_position

    # 提取 patch 在所有切片上的区域
    patch_A = cp_a[:, :, start_l:start_l + p_l, start_w:start_w + p_w, :].clone()
    patch_B = cp_b[:, :, start_l:start_l + p_l, start_w:start_w + p_w, :].clone()

    # 交换 patch
    cp_a[:, :, start_l:start_l + p_l, start_w:start_w + p_w, :] = patch_B
    cp_b[:, :, start_l:start_l + p_l, start_w:start_w + p_w, :] = patch_A

    return cp_a, cp_b

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, group_name, normalization='minmax', traAug=False):
        """
        Initialize HDF5Dataset.

        Args:
            hdf5_path: Path to the HDF5 file.
            group_name: Name of the group in the HDF5 file (e.g., train, val, test).
            normalization: Normalization method, either 'minmax' or 'zscore'.
        """
        self.hdf5_path = hdf5_path
        self.group_name = group_name
        self.traAug = traAug
        self.normalization = normalization
        self.file_paths = None
        self.egfr_labels = None
        self.traTrans = Compose([
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.2),
            RandRotate90d(keys=["image"], prob=0.2, max_k=3),
            RandAffined(keys=["image"], prob=0.2,
                        rotate_range=(0.1, 0.1, 0.1),
                        scale_range=(0.1, 0.1, 0.1),
                        translate_range=(10, 10, 5)),
            EnsureTyped(keys=["image"]),
        ])

        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            self.file_paths = hdf5_file[f'{self.group_name}/path'][:]
            self.egfr_labels = hdf5_file[f'{self.group_name}/label'][:]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        thin_path, thick_path = self.file_paths[idx]
        egfr_label = self.egfr_labels[idx]

        thin_data = self.load_nifti_image(thin_path.decode('utf-8'))
        thick_data = self.load_nifti_image(thick_path.decode('utf-8'))

        thin_data = thin_data[None, ...]
        thick_data = thick_data[None, ...]

        if self.traAug:
            thin_data = self.traTrans({"image": thin_data})["image"]
            thick_data = self.traTrans({"image": thick_data})["image"]

        if self.normalization == 'minmax':
            thin_data = (thin_data - thin_data.min()) / (thin_data.max() - thin_data.min())
            thick_data = (thick_data - thick_data.min()) / (thick_data.max() - thick_data.min())
        elif self.normalization == 'zscore':
            thin_data = (thin_data - thin_data.mean()) / thin_data.std()
            thick_data = (thick_data - thick_data.mean()) / thick_data.std()
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization}")

        return thin_data, thick_data, egfr_label

    def load_nifti_image(self, file_path):
        """
        Read a NIfTI file and return a normalized NumPy array.

        Args:
            file_path: Path to the NIfTI file.

        Returns:
            A normalized NumPy array of the image data.
        """
        import nibabel as nib
        import numpy as np
        image = nib.load(file_path)
        image_data = image.get_fdata()
        image_data = torch.tensor(image_data, dtype=torch.float32)
        return image_data


class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=16):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False)
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred, device):
        y_true = y_true.to(device)
        y_pred = y_pred.to(device)
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).to(device)

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred, device):
        return -self.mi(y_true, y_pred, device)

class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-8):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False).float()
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)

def save_volfile(image, path, affine=None, header=None, dtype=None):
    if dtype is not None:
        image = image.astype(dtype=dtype)
    if header is None:
        header = nib.Nifti1Header()
    if affine is None:
        affine = np.eye(4)
    nifty = nib.Nifti1Image(image, affine, header)
    nib.save(nifty, path)

def setup_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

def save_checkpoint(epoch, model, optimizer, save_dir, filename='checkpoint.pth'):
    """
    Save a model checkpoint.

    Args:
        epoch: Current epoch number.
        model: The model to save.
        optimizer: The optimizer whose state will be saved.
        save_dir: Directory to save the checkpoint.
        filename: Checkpoint file name.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(checkpoint, os.path.join(save_dir, filename))


def load_checkpoint(model, optimizer, save_dir):
    """
    Load a model checkpoint.

    Args:
        model: The model to load.
        optimizer: The optimizer to restore.
        save_dir: Directory where the checkpoint is stored.
        filename: Checkpoint file name.

    Returns:
        The current epoch number.
    """
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch





