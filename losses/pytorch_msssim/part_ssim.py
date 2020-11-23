import torch
from .ssim import _fspecial_gauss_1d, ssim

# part ssim calculation using 2D Gaussian masks + mean & std calculation using weight average method
def part_ssim(X, Y, part_mask, mask_t=0.4, data_range=255, size_average=True, K=(0.01,0.03)):
    batch_size, img_channel, H, W = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    mask_channel = part_mask.shape[1]
    new_shape = (batch_size, img_channel*mask_channel, H, W)

    X_repeat = torch.unsqueeze(X, 1).repeat(1, mask_channel, 1, 1, 1).reshape(new_shape)
    Y_repeat = torch.unsqueeze(Y, 1).repeat(1, mask_channel, 1, 1, 1).reshape(new_shape)
    part_mask_repeat = torch.unsqueeze(part_mask, 2).repeat(1,1,img_channel,1,1).reshape(new_shape)

    K1, K2 = K
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # calculate mean of X and Y (method 1)
    part_mask_sum = torch.unsqueeze(torch.unsqueeze(part_mask_repeat.sum((-2,-1)),-1),-1).repeat(1,1,H,W)
    part_mask_avg = part_mask_repeat/(part_mask_sum + 1e-6) # sum(part_mask[0,0,:,:]) = 1
    mu1 = torch.mul(X_repeat, part_mask_avg).sum((-2,-1))
    mu2 = torch.mul(Y_repeat, part_mask_avg).sum((-2,-1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    mu11 = torch.mul(X_repeat*X_repeat, part_mask_avg).sum((-2,-1))
    mu22 = torch.mul(Y_repeat*Y_repeat, part_mask_avg).sum((-2,-1))
    mu12 = torch.mul(X_repeat*Y_repeat, part_mask_avg).sum((-2,-1))
    sigma1_sq = compensation * (mu11 - mu1_sq)
    sigma2_sq = compensation * (mu22 - mu2_sq)
    sigma12   = compensation * (mu12 - mu1_mu2)

    cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2) # set alpha=beta=gamma=1
    ssim = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs

    if size_average:
        ssim_avg = ssim.mean()
    else:
        ssim_avg = ssim.mean(-1)

    return ssim_avg


class FPart_BSSIM(torch.nn.Module):  #  foreground part-SSIM + background SSIM
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, K=(0.01, 0.03), nonnegative_ssim=False):
        r""" class for foreground part_ssim + background ssim
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

        super(FPart_BSSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
        self.channel = channel

    def forward(self, X, Y, part_mask):
        # from [-1,1] to [0,1]
        X = (X + 1) / 2.0
        Y = (Y + 1) / 2.0
        f_mask = part_mask[:,1:,:,:]

        f_ssim = part_ssim(X, Y, f_mask, data_range=self.data_range, size_average=self.size_average)

        b_mask = torch.unsqueeze(part_mask[:,0,:,:], 1).repeat(1, self.channel, 1, 1)
        bX = torch.mul(b_mask, X)
        bY = torch.mul(b_mask, Y)
        b_ssim = ssim(bX, bY, win_size=self.win_size, win_sigma=self.win_sigma, win=self.win, data_range=self.data_range,
             size_average=self.size_average, K=self.K, nonnegative_ssim=self.nonnegative_ssim)

        return (f_ssim+b_ssim)/2
