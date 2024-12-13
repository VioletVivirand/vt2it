import torch
from torch import nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import os
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """
    @torch.no_grad()
    def _default_init_weights(self, module_list, scale=1, bias_fill=0, **kwargs):
        """Initialize network weights.

        Args:
            module_list (list[nn.Module] | nn.Module): Modules to be initialized.
            scale (float): Scale initialized weights, especially for residual
                blocks. Default: 1.
            bias_fill (float): The value to fill bias. Default: 0
            kwargs (dict): Other arguments for initialization function.
        """
        if not isinstance(module_list, list):
            module_list = [module_list]
        for module in module_list:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, **kwargs)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.fill_(bias_fill)
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, **kwargs)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.fill_(bias_fill)
                elif isinstance(m, _BatchNorm):
                    init.constant_(m.weight, 1)
                    if m.bias is not None:
                        m.bias.data.fill_(bias_fill)

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        self._default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """
    def _make_layer(self, basic_block, num_basic_block, **kwarg):
        """Make layers by stacking the same blocks.

        Args:
            basic_block (nn.module): nn.module class for basic block.
            num_basic_block (int): number of blocks.

        Returns:
            nn.Sequential: Stacked blocks in nn.Sequential.
        """
        layers = []
        for _ in range(num_basic_block):
            layers.append(basic_block(**kwarg))
        return nn.Sequential(*layers)

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = self._make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _pixel_unshuffle(self, x, scale):
        """ Pixel unshuffle.

        Args:
            x (Tensor): Input feature with shape (b, c, hh, hw).
            scale (int): Downsample ratio.

        Returns:
            Tensor: the pixel unshuffled feature.
        """
        b, c, hh, hw = x.size()
        out_channel = c * (scale**2)
        assert hh % scale == 0 and hw % scale == 0
        h = hh // scale
        w = hw // scale
        x_view = x.view(b, c, h, scale, w, scale)
        return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

    def forward(self, x):
        if self.scale == 2:
            feat = self._pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = self._pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class RealESRGAN:
    def __init__(self, scale=4, model_path=None):
        """ Initialize Real-ESRGAN model.

        Args:
            scale (int): Upsample scale.
            model_path (str): Path to the pretrained model.
                              Automatically download model from huggingface if model_path it not provided.
        """
        self.device = torch.device('cpu')
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale
        )
        self._load_weights(model_path=model_path)

    def _pad_reflect(self, image, pad_size):
        imsize = image.shape
        height, width = imsize[:2]
        new_img = np.zeros([height+pad_size*2, width+pad_size*2, imsize[2]]).astype(np.uint8)
        new_img[pad_size:-pad_size, pad_size:-pad_size, :] = image

        new_img[0:pad_size, pad_size:-pad_size, :] = np.flip(image[0:pad_size, :, :], axis=0) #top
        new_img[-pad_size:, pad_size:-pad_size, :] = np.flip(image[-pad_size:, :, :], axis=0) #bottom
        new_img[:, 0:pad_size, :] = np.flip(new_img[:, pad_size:pad_size*2, :], axis=1) #left
        new_img[:, -pad_size:, :] = np.flip(new_img[:, -pad_size*2:-pad_size, :], axis=1) #right

        return new_img

    def _pad_patch(self, image_patch, padding_size, channel_last=True):
        """ Pads image_patch with with padding_size edge values. """

        if channel_last:
            return np.pad(
                image_patch,
                ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
                'edge',
            )
        else:
            return np.pad(
                image_patch,
                ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
                'edge',
            )

    def _split_image_into_overlapping_patches(self, image_array, patch_size, padding_size=2):
        """ Splits the image into partially overlapping patches.
        The patches overlap by padding_size pixels.
        Pads the image twice:
            - first to have a size multiple of the patch size,
            - then to have equal padding at the borders.
        Args:
            image_array: numpy array of the input image.
            patch_size: size of the patches from the original image (without padding).
            padding_size: size of the overlapping area.
        """

        xmax, ymax, _ = image_array.shape
        x_remainder = xmax % patch_size
        y_remainder = ymax % patch_size

        # modulo here is to avoid extending of patch_size instead of 0
        x_extend = (patch_size - x_remainder) % patch_size
        y_extend = (patch_size - y_remainder) % patch_size

        # make sure the image is divisible into regular patches
        extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')

        # add padding around the image to simplify computations
        padded_image = self._pad_patch(extended_image, padding_size, channel_last=True)

        xmax, ymax, _ = padded_image.shape
        patches = []

        x_lefts = range(padding_size, xmax - padding_size, patch_size)
        y_tops = range(padding_size, ymax - padding_size, patch_size)

        for x in x_lefts:
            for y in y_tops:
                x_left = x - padding_size
                y_top = y - padding_size
                x_right = x + patch_size + padding_size
                y_bottom = y + patch_size + padding_size
                patch = padded_image[x_left:x_right, y_top:y_bottom, :]
                patches.append(patch)

        return np.array(patches), padded_image.shape

    def _unpad_patches(self, image_patches, padding_size):
        return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]

    def _stich_together(self, patches, padded_image_shape, target_shape, padding_size=4):
        """ Reconstruct the image from overlapping patches.
        After scaling, shapes and padding should be scaled too.
        Args:
            patches: patches obtained with split_image_into_overlapping_patches
            padded_image_shape: shape of the padded image contructed in split_image_into_overlapping_patches
            target_shape: shape of the final image
            padding_size: size of the overlapping area.
        """

        xmax, ymax, _ = padded_image_shape
        patches = self._unpad_patches(patches, padding_size)
        patch_size = patches.shape[1]
        n_patches_per_row = ymax // patch_size

        complete_image = np.zeros((xmax, ymax, 3))

        row = -1
        col = 0
        for i in range(len(patches)):
            if i % n_patches_per_row == 0:
                row += 1
                col = 0
            complete_image[
            row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size,:
            ] = patches[i]
            col += 1
        return complete_image[0: target_shape[0], 0: target_shape[1], :]

    def _unpad_image(self, image, pad_size):
        return image[pad_size:-pad_size, pad_size:-pad_size, :]

    def _load_weights(self, model_path):
        if not model_path:
            assert self.scale in [2, 4, 8], 'You can download models only with scales: 2, 4, 8'
            repo_id = 'ai-forever/Real-ESRGAN'
            filename = f'RealESRGAN_x{str(self.scale)}.pth'
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision='main',
                )
            print(f'Weights downloaded to:', os.path.abspath(model_path))

        loadnet = torch.load(model_path)

        if 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        self.model.eval()

    def to(self, device):
        """ Load model with specified device.

        Args:
            device (str): 'cpu' for CPU or 'cuda' for Nvidia GPU... etc.
        """
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.cuda.amp.autocast()
    def __call__(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        lr_image = self._pad_reflect(lr_image, pad_size)

        patches, p_shape = self._split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device).detach()

        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i+batch_size])), 0)

        sr_image = res.permute((0,2,3,1)).clamp_(0, 1).cpu()
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = self._stich_together(
            np_sr_image, padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape, padding_size=padding * scale
        )
        sr_img = (np_sr_image*255).astype(np.uint8)
        sr_img = self._unpad_image(sr_img, pad_size*scale)
        sr_img = Image.fromarray(sr_img)

        return sr_img
