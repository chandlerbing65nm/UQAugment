from typing import Union, Tuple
import torch
from torch import Tensor
from torch.nn import Module, Conv1d, BatchNorm1d, ReLU, Dropout

class DilatedConvBLock1D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        groups=1,
        dropout_prob: float = 0.5,
    ) -> None:
        """Dilated convolution block with BatchNorm and Dropout.

        :param in_channels: Amount of input channels.
        :param out_channels: Amount of output channels.
        :param input_size: Size of the input.
        :param kernel_size: Kernel size.
        :param stride: Stride size.
        :param dilation: Dilation rate.
        :param groups: Number of groups.
        :param dropout_prob: Probability of an element to be zeroed.
        """
        super().__init__()
        assert groups == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.input_size = input_size
        padding = self.get_padding_bins(input_size, self.dilation)

        self.cnn1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
            groups=groups,
        )
        self.batch_norm1 = BatchNorm1d(num_features=out_channels)
        self.non_linearity = ReLU(inplace=True)
        self.dropout1 = Dropout(p=dropout_prob)

        self.cnn2 = Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
            groups=groups,
        )
        self.batch_norm2 = BatchNorm1d(num_features=out_channels)
        self.dropout2 = Dropout(p=dropout_prob)

        # 1x1 convolution for residual connection if input and output channels differ
        if in_channels != out_channels:
            self.residual_conv = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.residual_conv = None

    def get_padding_bins(self, input_length, dilations):
        return int(
            (
                input_length * (self.stride - 1)
                - self.stride
                + dilations * (self.kernel_size - 1)
                + 1
            )
            / 2
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the dilated convolution block.

        :param x: Input tensor.
        :return: Output tensor.
        """
        residual = x

        out = self.cnn1(x)
        out = self.batch_norm1(out)
        out = self.non_linearity(out)
        out = self.dropout1(out)

        out = self.cnn2(out)
        out = self.batch_norm2(out)
        out = self.non_linearity(out)
        out = self.dropout2(out)

        # Apply 1x1 convolution to the residual if the channels are different
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        out = out + residual  # Residual connection
        return out



class DilatedConv(Module):
    def __init__(
        self,
        in_channels: int,
        dilation_rate: int,
        input_size: int,
        kernel_size: int,
        stride: int,
        out_channels=1,
        dropout_prob: float = 0.5,
    ) -> None:
        """Dilated convolution module with multiple blocks.

        :param in_channels: Amount of input channels.
        :param dilation_rate: Dilation rate for convolution.
        :param input_size: Size of the input.
        :param kernel_size: Kernel size.
        :param stride: Stride size.
        :param out_channels: Amount of output channels.
        :param dropout_prob: Probability of an element to be zeroed.
        """
        super().__init__()

        self.blks = torch.nn.ModuleList()

        self.blks.append(
            DilatedConvBLock1D(
                in_channels,
                in_channels,
                input_size=input_size,
                kernel_size=kernel_size,
                stride=stride,
                dilation=1,
                dropout_prob=dropout_prob,
            )
        )
        self.blks.append(
            DilatedConvBLock1D(
                in_channels,
                in_channels // 2,
                input_size=input_size,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation_rate,
                dropout_prob=dropout_prob,
            )
        )
        self.blks.append(
            DilatedConvBLock1D(
                in_channels // 2,
                in_channels // 4,
                input_size=input_size,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation_rate,
                dropout_prob=dropout_prob,
            )
        )
        self.blks.append(
            DilatedConvBLock1D(
                in_channels // 4,
                in_channels // 4,
                input_size=input_size,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation_rate,
                dropout_prob=dropout_prob,
            )
        )
        self.blks.append(
            DilatedConvBLock1D(
                in_channels // 4,
                out_channels,
                input_size=input_size,
                kernel_size=kernel_size,
                stride=stride,
                dilation=1,
                dropout_prob=dropout_prob,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the dilated convolution blocks.

        :param x: Input tensor.
        :return: Output tensor.
        """
        for blk in self.blks:
            x = blk(x)
        return x
