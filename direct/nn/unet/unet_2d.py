# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code borrowed / edited from: https://github.com/facebookresearch/fastMRI/blob/
import math
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp
from direct.data import transforms as T
from direct.nn.types import InitType
from collections import OrderedDict



class ConvBlock(nn.Module):
    """U-Net convolutional block.

    It consists of two convolution layers each followed by instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_probability: float):
        """Inits ConvBlock.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of :class:`ConvBlock`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        """Representation of :class:`ConvBlock`."""
        return (
            f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dropout_probability={self.dropout_probability})"
        )


class TransposeConvBlock(nn.Module):
    """U-Net Transpose Convolutional Block.

    It consists of one convolution transpose layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Inits :class:`TransposeConvBlock`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`TransposeConvBlock`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        """Representation of "class:`TransposeConvBlock`."""
        return f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels})"


class UnetModel2d(nn.Module):
    """PyTorch implementation of a U-Net model based on [1]_.

    References
    ----------

    .. [1] Ronneberger, Olaf, et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, edited by Nassir Navab et al., Springer International Publishing, 2015, pp. 234–41. Springer Link, https://doi.org/10.1007/978-3-319-24574-4_28.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
    ):
        """Inits :class:`UnetModel2d`.

        Parameters
        ----------
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_channels, num_filters, dropout_probability)])
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, dropout_probability)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, dropout_probability)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, dropout_probability)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, dropout_probability),
                nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1),
            )
        ]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`UnetModel2d`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        stack = []
        output = input_data

        # Apply down-sampling layers
        for _, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/bottom if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output


class NormUnetModel2d(nn.Module):
    """Implementation of a Normalized U-Net model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        norm_groups: int = 2,
    ):
        """Inits :class:`NormUnetModel2d`.

        Parameters
        ----------
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        norm_groups: int,
            Number of normalization groups.
        """
        super().__init__()

        self.unet2d = UnetModel2d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters,
            num_pool_layers=num_pool_layers,
            dropout_probability=dropout_probability,
        )

        self.norm_groups = norm_groups

    @staticmethod
    def norm(input_data: torch.Tensor, groups: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
        # group norm
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)

        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)

        output = (input_data - mean) / std
        output = output.reshape(b, c, h, w)

        return output, mean, std

    @staticmethod
    def unnorm(input_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, groups: int) -> torch.Tensor:
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    @staticmethod
    def pad(input_data: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = input_data.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        output = F.pad(input_data, w_pad + h_pad)
        return output, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(
        input_data: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return input_data[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`NormUnetModel2d`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        output, mean, std = self.norm(input_data, self.norm_groups)
        output, pad_sizes = self.pad(output)
        output = self.unet2d(output)

        output = self.unpad(output, *pad_sizes)
        output = self.unnorm(output, mean, std, self.norm_groups)

        return output


class Unet2d(nn.Module):
    """PyTorch implementation of a U-Net model for MRI Reconstruction."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        skip_connection: bool = False,
        normalized: bool = False,
        image_initialization: InitType = InitType.ZERO_FILLED,
        **kwargs,
    ):
        """Inits :class:`Unet2d`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_filters: int
            Number of first layer filters.
        num_pool_layers: int
            Number of pooling layers.
        dropout_probability: float
            Dropout probability.
        skip_connection: bool
            If True, skip connection is used for the output. Default: False.
        normalized: bool
            If True, Normalized Unet is used. Default: False.
        image_initialization: InitType
            Type of image initialization. Default: InitType.ZERO_FILLED.
        kwargs: dict
        """
        super().__init__()
        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "sensitivity_map_model",
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")
        self.unet: nn.Module
        if normalized:
            self.unet = NormUnetModel2d(
                in_channels=2,
                out_channels=2,
                num_filters=num_filters,
                num_pool_layers=num_pool_layers,
                dropout_probability=dropout_probability,
            )
        else:
            self.unet = UnetModel2d(
                in_channels=2,
                out_channels=2,
                num_filters=num_filters,
                num_pool_layers=num_pool_layers,
                dropout_probability=dropout_probability,
            )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.skip_connection = skip_connection
        self.image_initialization = image_initialization
        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        r"""Computes sense initialization :math:`x_{\text{SENSE}}`:

        .. math::
            x_{\text{SENSE}} = \sum_{k=1}^{n_c} {S^{k}}^* \times y^k

        where :math:`y^k` denotes the data from coil :math:`k`.

        Parameters
        ----------
        kspace: torch.Tensor
            k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        input_image: torch.Tensor
            Sense initialization :math:`x_{\text{SENSE}}`.
        """
        input_image = T.complex_multiplication(
            T.conjugate(sensitivity_map),
            self.backward_operator(kspace, dim=self._spatial_dims),
        )
        input_image = input_image.sum(self._coil_dim)
        return input_image

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes forward pass of Unet2d.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.

        Returns
        -------
        output: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        if self.image_initialization == InitType.SENSE:
            if sensitivity_map is None:
                raise ValueError("Expected sensitivity_map not to be None with InitType.SENSE image_initialization.")
            input_image = self.compute_sense_init(
                kspace=masked_kspace,
                sensitivity_map=sensitivity_map,
            )
        elif self.image_initialization == InitType.ZERO_FILLED:
            input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)
        else:
            raise ValueError(
                f"Unknown image_initialization. Expected InitType.ZERO_FILLED or InitType.SENSE. "
                f"Got {self.image_initialization}."
            )

        output = self.unet(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.skip_connection:
            output += input_image
        return output






class UNet2dPytorch(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet2dPytorch, self).__init__()

        features = init_features
        self.encoder1 = UNet2dPytorch._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2dPytorch._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2dPytorch._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2dPytorch._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2dPytorch._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2dPytorch._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2dPytorch._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2dPytorch._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2dPytorch._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class Unet2dImageSpace(nn.Module):
    """PyTorch implementation of a U-Net model for MRI Reconstruction refinement."""

    def __init__(
        self,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        normalized: bool = False,
        forward_operator: Callable = None,
        backward_operator: Callable = None,
        num_inputs: int = 1,
        model_type='default',
        final_activations = 'relu',
        **kwargs,
    ):
        """Inits :class:`Unet2dImageSpace`.

        Parameters
        ----------
        num_filters: int
            Number of first layer filters.
        num_pool_layers: int
            Number of pooling layers.
        dropout_probability: float
            Dropout probability.
        skip_connection: bool
            If True, skip connection is used for the output. Default: False.
        normalized: bool
            If True, Normalized Unet is used. Default: False.
        kwargs: dict
        """

        super().__init__()
        extra_keys = kwargs.keys()
        if model_type == 'default':
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=num_inputs, out_channels=16,
                          kernel_size=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32,
                          kernel_size=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64,
                          kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=1,
                          kernel_size=1)
            )

        elif model_type == 'smp':
            self.model = smp.Unet(
            encoder_name = 'resnet50',
            encoder_weights = None,
            decoder_channels = (512,256,128,64,32),
            in_channels = 3)
    
    
            weights = torch.load('pretrained_weights/resnet50-19c8e357.pth');
            self.model.encoder.load_state_dict(weights);

            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
                    break;
            weight = module.weight.detach();
            module.in_channels = num_inputs;

            new_weight = torch.Tensor(
            module.out_channels, num_inputs // module.groups, *module.kernel_size
            )

            for i in range(num_inputs):
                new_weight[:, i] = weight[:, i % 3]

            new_weight = new_weight * (3 / num_inputs)
            module.weight = nn.parameter.Parameter(new_weight)

        elif model_type == 'pytorch':
            self.model = UNet2dPytorch(in_channels=num_inputs)
            model_weights = torch.load('pretrained_weights/unet-e012d006.pt');
            model_weights.pop('encoder1.enc1conv1.weight')
            self.model.load_state_dict(model_weights, strict=False);


        self.final_activations = final_activations;
    def forward(
        self,
        input_images: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of Unet2d.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.

        Returns
        -------
        output: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        input_image = torch.cat([*input_images], dim = 1)
        output = self.model(input_image)

        if self.final_activations == 'relu':
            output = F.relu(output);
        elif self.final_activations == 'abs':
            output = torch.abs(output);
        elif self.final_activations == 'sigmoid':
            output = F.sigmoid(output);
        return output