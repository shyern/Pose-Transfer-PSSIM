import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blured tensors
    """

    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, full=False, K=(0.01,0.03), nonnegative_ssim=False):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.

    Returns:
        torch.Tensor: ssim results
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2) # set alpha=beta=gamma=1
    if nonnegative_ssim:
        cs_map = F.relu( cs_map, inplace=True )

    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    # ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)).pow(0) * cs_map

    if nonnegative_ssim:
        ssim_map = F.relu(ssim_map, inplace=True)

    if size_average:
        # if ssim_map.shape[1] == 3:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
        # else:
        #     no_zero_count = torch.sum(ssim_map>0.4)
        #     print(no_zero_count)
        #     ssim_val = ssim_map.sum()/no_zero_count
    else:
        # if ssim_map.shape[1] == 3:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)
        # else:
        #     no_zero_count = torch.sum(ssim_map>0.4, (1,2,3))
        #     print(no_zero_count)
        #     ssim_val = ssim_map.sum((ssim_map.shape[1:]))/no_zero_count
        #     cs = cs_map.sum((ssim_map.shape[1:]))/no_zero_count

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, K=(0.01, 0.03), nonnegative_ssim=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.

    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True, K=K, nonnegative_ssim=nonnegative_ssim)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, weights=None, K=(0.01, 0.03), nonnegative_ssim=False):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    smaller_side = min( X.shape[-2:] )
    if weights is None:
        min_img_smaller_size = (win_size-1) * (2**2)
    else:
        min_img_smaller_size = (win_size - 1) * (2 ** weights.shape[0])
    assert smaller_side >= min_img_smaller_size, \
         "Image size should be larger than %d due to the 4 downsamplings in ms-ssim"% ((win_size-1) * (2**4))

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001]).to(X.device, dtype=X.dtype)
    # [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True, K=K, nonnegative_ssim=nonnegative_ssim)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    # msssim_val = torch.prod((torch.sign(mcs[:-1]) * torch.abs(mcs[:-1]) ** weights[:-1].unsqueeze(1)), dim=0) * (torch.sign(ssim_val) * torch.abs(ssim_val) ** weights[-1])
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1)), dim=0) * (ssim_val ** weights[-1]) # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, K=(0.01, 0.03), nonnegative_ssim=False):
        r""" class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.
        """

        super(SSIM, self).__init__()
        # self.win = _fspecial_gauss_1d(
        #     win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma)
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        B, channel, H, W = X.shape
        if channel == 3:
            # mean = torch.FloatTensor(3)
            # mean[0] = 0.485
            # mean[1] = 0.456
            # mean[2] = 0.406
            # mean = Variable(mean)
            # mean = mean.resize(1, 3, 1, 1).cuda()
            #
            # std = torch.FloatTensor(3)
            # std[0] = 0.229
            # std[1] = 0.224
            # std[2] = 0.225
            # std = Variable(std)
            # std = std.resize(1, 3, 1, 1).cuda()
            #
            # X_p2_norm = (X + 1) / 2  # [-1, 1] => [0, 1]
            # X_p2_norm = (X_p2_norm - mean) / std
            #
            # Y_p2_norm = (Y + 1) / 2  # [-1, 1] => [0, 1]
            # Y_p2_norm = (Y_p2_norm - mean) / std
            X_norm = (X+1)/2.0
            Y_norm = (Y+1)/2.0
        else:
            X_norm = X
            Y_norm = Y
            # X_max_channel = (X.transpose(0,1).reshape((channel,-1))).max(1)[0]
            # X_max_channel_rep = X_max_channel.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat((B, 1, H, W))
            # X_norm = X / X_max_channel_rep
            # Y_max_channel = (Y.transpose(0, 1).reshape((channel, -1))).max(1)[0]
            # Y_max_channel_rep = Y_max_channel.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat((B, 1, H, W))
            # Y_norm = Y / Y_max_channel_rep

            # X_transpose = (X.transpose(0,1).reshape((channel,-1)))
            # X_mean_channel = X_transpose.mean(1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat((B,1,H,W))
            # X_std_channel = X_transpose.std(1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat((B,1,H,W))
            # X_norm = (X-X_mean_channel)/X_std_channel
            #
            # Y_transpose = (Y.transpose(0, 1).reshape((channel, -1)))
            # Y_mean_channel = Y_transpose.mean(1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat((B, 1, H, W))
            # Y_std_channel = Y_transpose.std(1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat((B, 1, H, W))
            # Y_norm = (Y - Y_mean_channel) / Y_std_channel

        self.win = torch.unsqueeze(self.win[0,...],0).repeat(channel, 1, 1, 1)

        return ssim(X_norm, Y_norm, win_size=self.win_size, win_sigma=self.win_sigma, win=self.win, data_range=self.data_range, size_average=self.size_average, K=self.K, nonnegative_ssim=self.nonnegative_ssim)

    
class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, weights=None, K=(0.01, 0.03), nonnegative_ssim=False):
        r""" class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
        self.win_size = win_size
        self.win_sigma = win_sigma

    def forward(self, X, Y):
        return ms_ssim(X, Y, win_size=self.win_size, win_sigma=self.win_sigma, win=self.win, size_average=self.size_average, data_range=self.data_range, weights=self.weights, K=self.K, nonnegative_ssim=self.nonnegative_ssim)

# class Bodypart_SSIM(torch.nn.Module):
#     def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, weights=None, K=(0.01, 0.03), nonnegative_ssim=False):
#         r""" class for ms-ssim
#         Args:
#             win_size: (int, optional): the size of gauss kernel
#             win_sigma: (float, optional): sigma of normal distribution
#             data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
#             size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
#             channel (int, optional): input channels (default: 3)
#             weights (list, optional): weights for different levels
#             K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
#             nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
#         """
#
#         super(Bodypart_SSIM, self).__init__()
#         self.win = _fspecial_gauss_1d(
#             win_size, win_sigma).repeat(channel, 1, 1, 1)
#         self.size_average = size_average
#         self.data_range = data_range
#         self.weights = weights
#         self.K = K
#         self.nonnegative_ssim = nonnegative_ssim
#         self.win_size = win_size
#         self.win_sigma = win_sigma
#
#     def forward(self, X, Y, bodypart_mask_12):
#         # fg_mask = torch.unsqueeze(bodypart_mask[:,0,:,:], 1).repeat(1,self.channel,1,1)
#         # fg_x = torch.mul(X, fg_mask)
#         # fg_y = torch.mul(Y, fg_mask)
#         #
#         # bg_mask = torch.unsqueeze(bodypart_mask[:,1,:,:], 1).repeat(1,self.channel,1,1)
#         # bg_x = torch.mul(X, bg_mask)
#         # bg_y = torch.mul(Y, bg_mask)
#         #
#         # loss_fg = ssim(fg_x, fg_y, win_size=self.win_size, win_sigma=self.win_sigma, win=self.win,
#         #                data_range=self.data_range,
#         #                size_average=self.size_average, K=self.K, nonnegative_ssim=self.nonnegative_ssim)
#         # loss_bg = ssim(bg_x, bg_y, win_size=self.win_size, win_sigma=self.win_sigma, win=self.win,
#         #                data_range=self.data_range,
#         #                size_average=self.size_average, K=self.K, nonnegative_ssim=self.nonnegative_ssim)
#         # loss = loss_fg + loss_bg
#
#         # X = (X+1)/2.0
#         # Y = (Y+1)/2.0
#         # bodypart_mask = bodypart_mask_12[:,1:,:,:]
#         # mask_channel = bodypart_mask.shape[1]
#         # mask_repeat = torch.unsqueeze(bodypart_mask, 2).repeat(1, 1, self.channel, 1, 1)
#         # X_repeat = torch.unsqueeze(X, 1).repeat(1, mask_channel, 1, 1, 1)
#         # Y_repeat = torch.unsqueeze(Y, 1).repeat(1, mask_channel, 1, 1, 1)
#         #
#         # new_shape = (mask_repeat.shape[0], mask_repeat.shape[1]*mask_repeat.shape[2], mask_repeat.shape[3], mask_repeat.shape[4])
#         # X_masked = torch.mul(X_repeat, mask_repeat).reshape(new_shape)
#         # Y_masked = torch.mul(Y_repeat, mask_repeat).reshape(new_shape)
#         # self.new_win = self.win.repeat(mask_channel, 1, 1, 1)
#         # loss = ssim(X_masked, Y_masked, win_size=self.win_size, win_sigma=self.win_sigma, win=self.new_win,
#         #             data_range=self.data_range,
#         #             size_average=self.size_average, K=self.K, nonnegative_ssim=self.nonnegative_ssim)
#
#         mask_t = 0.4
#         bodypart_mask = bodypart_mask_12[:,1:,:,:]
#         loss = bp_ssim(X, Y, bodypart_mask, win_size=self.win_size, win_sigma=self.win_sigma, mask_t=mask_t,
#                        win=self.win,
#                        data_range=self.data_range, size_average=self.size_average, K=self.K,
#                        nonnegative_ssim=self.nonnegative_ssim)
#
#         return loss
