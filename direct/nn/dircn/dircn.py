# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable

import torch
import torch.nn as nn

from direct.data.transforms import expand_operator, reduce_operator


from typing import Optional, List, Tuple, Union
import math

import torch.nn.functional as F

class Bottleneck(nn.Module):
    """
    Original paper:
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    Inspiration from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self,
                 channels: int,
                 mid_channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 ratio: float = 1./16,
                 activation: Union[Callable[..., nn.Module], None] = nn.ReLU(inplace=True),
                 downsample: Optional[nn.Module] = None,
                 ):
        super().__init__()

        self.stride = stride
        if stride > 1 and downsample is None:
            downsample = nn.Conv2d(
                in_channels=channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                )

        self.downsample = downsample


        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            )

        self.norm1 = nn.InstanceNorm2d(num_features=mid_channels, affine=bias)

        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            groups=groups,
            bias=False,
            padding=1,
            )

        self.norm2 = nn.InstanceNorm2d(num_features=mid_channels, affine=bias)

        self.conv3 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=False
            )
        self.norm3 = nn.InstanceNorm2d(num_features=channels, affine=bias)

        self.activation = activation

        self.se = SqueezeExcitation(channels=channels, ratio=ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = self.se(x)

        if self.stride > 1:
            identity = self.downsample(identity)

        x += identity
        x = self.activation(x)

        return x

class SqueezeExcitation(nn.Module):
    """
    Squeeze and excitation block based on:
    https://arxiv.org/abs/1709.01507
    ratio set at 1./16 as recommended by the paper
    """

    def __init__(self,
                 channels: int,
                 ratio: float = 1./16,
                 ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        squeezed_channels = max(1, int(channels*ratio))
        self.layer_1 = nn.Conv2d(in_channels=channels,
                                 out_channels=squeezed_channels,
                                 kernel_size=1,
                                 bias=True,
                                 )
        self.layer_2 = nn.Conv2d(in_channels=squeezed_channels,
                                 out_channels=channels,
                                 kernel_size=1,
                                 bias=True)
        # self.swish = MemoryEfficientSwish()
        self.act = nn.ReLU(inplace=True)
        # Could do this using linear layer aswell, but than we need to .view in forward
        # self.linear_1 = nn.Linear(in_features=channels, out_features=squeezed_channels, bias=True)
        # self.linear_2 = nn.Linear(in_features=squeezed_channels, out_features=channels, bias=True)

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        x = torch.sigmoid(x) * inputs
        return x

class BasicBlock(nn.Module):
    """
    Original paper:
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    Inspiration from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self,
                 channels: int,
                 stride: int = 1,
                 bias: bool = True,
                 groups: int = 1,
                 ratio: float = 1./16,
                 activation: Union[Callable[..., nn.Module], None] = nn.ReLU(inplace=True),
                 downsample: Optional[nn.Module] = None,
                 ):
        super().__init__()

        self.stride = stride
        if stride > 1 and downsample is None:
            downsample = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                )

        self.downsample = downsample


        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=stride,
            groups=1,
            bias=False,
            padding=1,
            )
        self.norm1 = nn.InstanceNorm2d(num_features=channels, affine=bias)

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            groups=groups,
            bias=False,
            padding=1,
            )
        self.norm2 = nn.InstanceNorm2d(num_features=channels, affine=bias)

        self.activation = activation

        self.se = SqueezeExcitation(channels=channels, ratio=ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = self.se(x)

        if self.stride > 1:
            identity = self.downsample(identity)

        x += identity
        x = self.activation(x)

        return x
    
class ResXUNet(nn.Module):

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n: int = 64,
        groups: int = 32,
        bias: bool = False,
        ratio: float = 1./8,
        activation: Union[nn.Module, None] = None,
        interconnections: bool = False,
        make_interconnections: bool = False,
        ):
        super().__init__()
        self.interconnections = interconnections
        self.make_interconnections = make_interconnections

        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.activation = nn.ReLU(inplace=True) if activation is None else activation
        self.activation = nn.ReLU(inplace=True) if activation is None else activation


        self.input = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            nn.InstanceNorm2d(num_features=n, affine=bias),
            self.activation)

        self.inc = BasicBlock(
            channels=n,
            bias=bias,
            groups=groups,
            ratio=ratio,
            activation=self.activation,
            )


        self.down1 = DownConvBlock(
            in_channels=n,
            out_channels=2*n,
            groups=1,
            bias=bias,
            activation=self.activation,
            )

        self.down1_basic = BasicBlock(
            channels=2*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            activation=self.activation,
            )


        self.down2 = DownConvBlock(
            in_channels=2*n,
            out_channels=4*n,
            groups=1,
            bias=bias,
            activation=self.activation,
            )

        self.down2_basic = BasicBlock(
            channels=4*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            activation=self.activation,
            )


        self.down3 = DownConvBlock(
            in_channels=4*n,
            out_channels=8*n,
            groups=1,
            bias=bias,
            activation=self.activation,
            )

        self.down3_basic = BasicBlock(
            channels=8*n,
            bias=bias,
            groups=4*groups,
            ratio=ratio,
            activation=self.activation,
            )


        self.down4 = DownConvBlock(
            in_channels=8*n,
            out_channels=8*n,
            groups=1,
            bias=bias,
            activation=self.activation,
            )

        self.down4_bottle = nn.Sequential(*[Bottleneck(
            channels=8*n,
            mid_channels=8*n // 2,
            groups=4*groups,
            bias=bias,
            ratio=ratio,
            activation=self.activation,
            ) for i in range(2)])


        self.up4 = TransposeConvBlock(
            in_channels=8*n,
            out_channels=8*n,
            groups=1,
            bias=bias,
            activation=self.activation
            )




        self.up3_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=3*8*n if make_interconnections else 2*8*n,
                out_channels=8*n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            nn.InstanceNorm2d(num_features=8*n, affine=bias),
            self.activation)


        self.up3_basic = BasicBlock(
            channels=8*n,
            bias=bias,
            groups=4*groups,
            ratio=ratio,
            activation=self.activation,
            )

        self.up3 = TransposeConvBlock(
            in_channels=8*n,
            out_channels=4*n,
            groups=1,
            bias=bias,
            activation=self.activation
            )


        self.up2_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=3*4*n if make_interconnections else 2*4*n,
                out_channels=4*n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            nn.InstanceNorm2d(num_features=4*n, affine=bias),
            self.activation)


        self.up2_basic = BasicBlock(
            channels=4*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            activation=self.activation,
            )

        self.up2 = TransposeConvBlock(
            in_channels=4*n,
            out_channels=2*n,
            groups=1,
            bias=bias,
            activation=self.activation
            )




        self.up1_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=3*2*n if make_interconnections else 2*2*n,
                out_channels=2*n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            nn.InstanceNorm2d(num_features=2*n, affine=bias),
            self.activation)


        self.up1_basic = BasicBlock(
            channels=2*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            activation=self.activation,
            )

        self.up1 = TransposeConvBlock(
            in_channels=2*n,
            out_channels=n,
            groups=1,
            bias=bias,
            activation=self.activation
            )



        self.out_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=3*n if make_interconnections else 2*n,
                out_channels=n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            nn.InstanceNorm2d(num_features=n, affine=bias),
            self.activation)

        self.out_1 = BasicBlock(
            channels=n,
            bias=bias,
            groups=groups,
            ratio=ratio,
            activation=self.activation,
            )

        self.final_bottle = nn.Sequential(Bottleneck(
            channels=n,
            mid_channels=n,
            groups=groups,
            bias=bias,
            ratio=ratio,
            activation=self.activation,
            ))

        self.outc = nn.Conv2d(in_channels=n, out_channels=n_classes, stride=1, kernel_size=1)

    def forward(self, x: torch.Tensor, internals: Optional[List[torch.Tensor]] = None):

        x = self.input(x)
        x1 = self.inc(x)


        x2 = self.down1(x1)
        x2 = self.down1_basic(x2)

        x3 = self.down2(x2)
        x3 = self.down2_basic(x3)

        x4 = self.down3(x3)
        x4 = self.down3_basic(x4)

        x = self.down4(x4)
        x = self.down4_bottle(x)
        x = self.up4(x)

        if self.interconnections and internals is not None:
            assert len(internals) == 4, "When using dense cascading, all layers must be given"
            # Connect conv
            x1 = torch.cat([x1, internals[0]], dim=1)
            x2 = torch.cat([x2, internals[1]], dim=1)
            x3 = torch.cat([x3, internals[2]], dim=1)
            x4 = torch.cat([x4, internals[3]], dim=1)

        internals = list()

        x = torch.cat([x, x4], dim=1)
        x = self.up3_channel(x)
        x = self.up3_basic(x)
        internals.append(x)
        x = self.up3(x)

        x = torch.cat([x, x3], dim=1)
        x = self.up2_channel(x)
        x = self.up2_basic(x)
        internals.append(x)
        x = self.up2(x)

        x = torch.cat([x, x2], dim=1)
        x = self.up1_channel(x)
        x = self.up1_basic(x)
        internals.append(x)
        x = self.up1(x)

        x = torch.cat([x, x1], dim=1)
        x = self.out_channel(x)
        x = self.out_1(x)
        x = self.final_bottle(x)
        internals.append(x)

        internals.reverse()

        if self.interconnections:
            return self.outc(x), internals
        return self.outc(x)



# Additional blocks to make making the network easier (avoid unnecessary repeated code)

class DownConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 groups: int,
                 bias: bool,
                 activation: Union[nn.Module, None] = None,
                 ):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=2,
            stride=2,
            bias=False,
            padding=0,
            )

        self.norm = nn.InstanceNorm2d(num_features=out_channels, affine=bias)
        self.act = nn.ReLU(inplace=True) if activation is None else activation

    def forward(self, x: torch.Tensor):
        x = self.down(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class TransposeConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 groups: int,
                 bias: bool,
                 activation: Union[nn.Module, None] = None,
                 ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=2,
            stride=2,
            bias=False,
            padding=0,
            )
        self.norm = nn.InstanceNorm2d(num_features=out_channels, affine=bias)
        self.act = nn.ReLU(inplace=True) if activation is None else activation

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class NormNet(nn.Module):
    """A normalized wrapper for ResXUNet or for any models in that manner.
    Each input channel is normalized independatly, and the means and stds of the first channel, i.e.,
    for the lastest cascade is used to de-normalize the output prediction for said cascade.
    """

    def __init__(self,
        n: int = 24,
        n_channels: int = 2,
        groups: int = 4,
        bias: bool = True,
        ratio: float = 1./8,
        interconnections: bool = False,
        make_interconnections: bool = False,
        ):
        """
        Args:
            n (int): the number of channels in the model
            n_channels (int): the number of input channels
            groups (int): the number of groups used in the convolutions, needs to dividable with n
            bias (bool): whether to use bias or not
            ratio (float): the ratio for squeeze and excitation
            interconnections (bool): whether to enable interconnection
            make_interconnections (bool): whether to make the model accept interconnection input
        """
        super().__init__()

        self.interconnections = interconnections

        self.model = ResXUNet(
            n_channels=n_channels,
            n_classes=2,
            n=n,
            groups=groups,
            bias=bias,
            ratio=ratio,
            activation=torch.nn.SiLU(inplace=True),
            interconnections=interconnections,
            make_interconnections=make_interconnections
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        a =  x.permute(0, 1, 4, 2, 3).reshape(b, 2 * c, h, w)
        return a

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, c, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize each channel individually
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean[:, :2], std[:, :2]

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, internals: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for the model and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        if self.interconnections:
            x, internals = self.model(x, internals)
        else:
            x = self.model(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        if self.interconnections:
            return x, internals
        return x

class DIRCN(nn.Module):
    """End-to-End Variational Network based on [1]_.

    References
    ----------

    .. [1] Sriram, Anuroop, et al. “End-to-End Variational Networks for Accelerated MRI Reconstruction.”
        ArXiv:2004.06688 [Cs, Eess], Apr. 2020. arXiv.org, http://arxiv.org/abs/2004.06688.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_cascades: int = 30,
        n: int = 20,
        groups: int = 4,
        bias: bool = True,
        ratio: float = 1./8,
        dense: bool = True,
        variational: bool = False,
        interconnections: bool = True,
        min_complex_support: bool = True,
        **kwargs,
    ):
        """Inits :class:`DIRCN`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_layers: int
            Number of cascades.
        regularizer_num_filters: int
            Regularizer model number of filters.
        regularizer_num_pull_layers: int
            Regularizer model number of pulling layers.
        regularizer_dropout: float
            Regularizer model dropout probability.
        """
        super().__init__()
        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        self.interconnections = interconnections
        self.i_cascades = nn.ModuleList(
            [ImageBlock(
                forward_operator = forward_operator,
                backward_operator=backward_operator,
                dense=dense,
                interconnections=interconnections,
                variational=variational,
                i_model=NormNet(
                    n=n,
                    n_channels=2*(i+1) if dense else 2,
                    groups=groups,
                    bias=bias,
                    ratio=ratio,
                    interconnections=interconnections,
                    make_interconnections=True if interconnections and i > 0 else False,
                    )) for i in range(num_cascades)])

        self.min_complex_support = min_complex_support

    def forward(
        self, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        """Performs the forward pass of :class:`DIRCN`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        kspace_prediction: torch.Tensor
            K-space prediction of shape (N, coil, height, width, complex=2).
        """

        kspace_pred = masked_kspace.clone()


        i_concat = None  # The first concat is always None
        interconnections = None

        for i, i_cascade in enumerate(self.i_cascades):
            if self.interconnections:
                kspace_pred, i_concat, interconnections = i_cascade(kspace_pred, masked_kspace, sampling_mask, sensitivity_map, i_concat, interconnections)
            else:
                kspace_pred, i_concat = i_cascade(kspace_pred, masked_kspace, sampling_mask, sensitivity_map, i_concat)



        return kspace_pred;


class EndToEndVarNetBlock(nn.Module):
    """End-to-End Variational Network block."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        regularizer_model: nn.Module,
    ):
        """Inits :class:`EndToEndVarNetBlock`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        regularizer_model: nn.Module
            Regularizer model.
        """
        super().__init__()
        self.regularizer_model = regularizer_model
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.learning_rate = nn.Parameter(torch.tensor([1.0]))
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the forward pass of :class:`EndToEndVarNetBlock`.

        Parameters
        ----------
        current_kspace: torch.Tensor
            Current k-space prediction of shape (N, coil, height, width, complex=2).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        torch.Tensor
            Next k-space prediction of shape (N, coil, height, width, complex=2).
        """
        kspace_error = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_kspace - masked_kspace,
        )
        regularization_term = torch.cat(
            [
                reduce_operator(
                    self.backward_operator(kspace, dim=self._spatial_dims), sensitivity_map, dim=self._coil_dim
                )
                for kspace in torch.split(current_kspace, 2, self._complex_dim)
            ],
            dim=self._complex_dim,
        ).permute(0, 3, 1, 2)
        regularization_term = self.regularizer_model(regularization_term).permute(0, 2, 3, 1)
        regularization_term = torch.cat(
            [
                self.forward_operator(
                    expand_operator(image, sensitivity_map, dim=self._coil_dim), dim=self._spatial_dims
                )
                for image in torch.split(regularization_term, 2, self._complex_dim)
            ],
            dim=self._complex_dim,
        )
        return current_kspace - self.learning_rate * kspace_error + regularization_term


class ImageBlock(nn.Module):
    """
    Model block for end-to-end variational network.
    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, 
                forward_operator: Callable,
                backward_operator: Callable,
                i_model: nn.Module, 
                dense: bool = True, 
                variational: bool = False, 
                interconnections: bool = False):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.i_model = i_model
        self.dense = dense
        self.variational = variational
        self.interconnections = interconnections
        self.dc_weight = nn.Parameter(torch.Tensor([0.01]))

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)
    
    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_conj: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, sens_conj).sum(
            dim=1, keepdim=True
        )


    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        concat: torch.Tensor,
        interconnections: Optional[List[torch.Tensor]] = None,
        ) -> torch.Tensor:


        inp = reduce_operator(self.backward_operator(current_kspace, dim=self._spatial_dims), sens_maps, dim=self._coil_dim, keep_dim=True)

        if concat is None or not self.dense:  # Check if there are any previous concats or if dense connections are on the menu
            concat = inp
        else:
            concat = torch.cat([inp, concat], dim=1)

        if self.interconnections:
            model_term, interconnections = self.i_model(concat, interconnections)
        else:
            model_term = self.i_model(concat)

        model_term_expanded = self.forward_operator(expand_operator(model_term, sens_maps, dim=self._coil_dim, unsqueeze = False), dim=self._spatial_dims)  # Expand stuff

        if self.variational:
            zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
            soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
            kspace = current_kspace - soft_dc - model_term_expanded
        else:
            kspace = torch.where(mask,
                            (ref_kspace + self.dc_weight*model_term_expanded)/(1+self.dc_weight),
                            model_term_expanded)

        if self.interconnections:
            return kspace, concat, interconnections
        return kspace, concat
