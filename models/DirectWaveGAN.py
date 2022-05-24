import torch.utils.data
import torch.nn as nn
import torch
import contextlib
import torch.nn.functional as F
from torch.cuda.amp import autocast


class ScaledTanh(nn.Module):
    def __init__(self, scaler):
        super(ScaledTanh, self).__init__()
        self.scaler = scaler
        self.activation = nn.Tanh()

    def forward(self, input):
        out = self.activation(input)
        out = out * self.scaler
        return out


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor by a random integer
    in {-n, n} and performing reflection padding where necessary.
    Taken from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    """
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)
        # Combine sample indices into lists so that less shuffle operations need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)
        x_shuffle = x.clone()    # Make a copy of x for our output
        for k, idxs in k_map.items():   # Apply shuffle to each sample
            x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect') if k > 0 else \
                F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')
        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle


weight_init_dispatcher = {'uniform': nn.init.uniform_,
                          'normal': nn.init.normal_,
                          'xavier_uniform': nn.init.xavier_uniform_,
                          'xavier_normal': nn.init.xavier_normal_,
                          'kaiming_uniform': nn.init.kaiming_uniform_,
                          'kaiming_normal': nn.init.kaiming_normal_,
                          'orthogonal': nn.init.orthogonal_}


class G(nn.Module):
    def __init__(self,  **kwargs):
        super(G, self).__init__()
        self.__dict__.update(kwargs)  # unpack the dictionary such that each key becomes self.key
        self.input_channel_dim = self.G_channel_list[0]
        self.conv_layers = nn.ModuleList()
        if "quantized" not in self.__dict__:
            self.quantized = False
        self.build_model()
        input_size = self.lower_resolution * self.G_channel_list[0]
        self.input_layer = nn.Linear(self.latent_dim, input_size, bias=self.gan_bias)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                weight_init_dispatcher[self.weight_init](m.weight.data)

    def build_model(self):
        for layer_num in range(self.model_levels):
            input_channels = self.input_channel_dim if layer_num == 0 else self.G_channel_list[layer_num]
            output_channels = self.G_channel_list[layer_num + 1]
            if self.receptive_field_type == "kernel":
                kernel_size = self.kernel_size // self.scale_factor ** (self.model_levels - layer_num - 1)
                kernel_size = self.min_kernel_size if kernel_size < self.min_kernel_size else kernel_size
                dilation = 1
            elif self.receptive_field_type == "dilation":
                kernel_size = 8
                dilation = 2 ** layer_num
            else:
                kernel_size, dilation = self.kernel_size, 1
            output_padding = 1 if dilation > 1 or self.scale_factor == 1 else 0
            padding = ((kernel_size - 1) * dilation + output_padding - 1) // 2 - (self.scale_factor - 1) // 2
            if kernel_size % 2 != 0:
                padding += 1
                output_padding = 1
            conv_block = [nn.ConvTranspose1d(input_channels, output_channels, kernel_size, self.scale_factor,
                                             padding, output_padding, dilation=dilation, bias=self.gan_bias)]
            if self.gen_batch_norm and layer_num != self.model_levels - 1:
                bn = nn.BatchNorm1d(num_features=output_channels)
                conv_block.append(bn)
            if layer_num != self.model_levels - 1:
                activation_function = nn.ReLU()
            else:
                activation_function = nn.Tanh() if not self.quantized else ScaledTanh(self.max_abs_limit)
            conv_block.append(activation_function)
            self.conv_layers.append(nn.Sequential(*conv_block))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        x = self.input_layer(z)
        x = x.view(-1, self.G_channel_list[0], self.lower_resolution)
        for layer_number in range(self.model_levels):
            x = self.conv_layers[layer_number](x)
        return x


class D(nn.Module):
    def __init__(self,  **kwargs):
        super(D, self).__init__()
        self.__dict__.update(kwargs)  # unpack the dictionary such that each key becomes self.key
        self.output_channel_dim = self.D_channel_list[-1]
        output_size = self.lower_resolution * self.output_channel_dim
        self.output_layer = nn.Linear(output_size, 1, bias=self.gan_bias)
        self.min_kernel_size = 4
        self.conv_layers = nn.ModuleList()
        self.build_model()
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                weight_init_dispatcher[self.weight_init](m.weight.data)

    def build_model(self):
        for layer_num in range(self.model_levels):
            layer_out_dim = self.lower_resolution * self.scale_factor ** (self.model_levels - layer_num - 1)
            input_channels = self.D_channel_list[layer_num]
            output_channels = self.output_channel_dim if layer_num == self.model_levels - 1 \
                else self.D_channel_list[layer_num + 1]
            if self.receptive_field_type == "kernel":
                kernel_size = self.kernel_size // self.scale_factor ** layer_num
                kernel_size = self.min_kernel_size if kernel_size < self.min_kernel_size else kernel_size
                dilation = 1
            elif self.receptive_field_type == "dilation":
                kernel_size = 8
                dilation = 2 ** (self.model_levels - layer_num - 1)
            else:
                kernel_size, dilation = self.kernel_size, 1
            padding = ((kernel_size - 1) * dilation - 1) // 2 - (self.scale_factor - 1) // 2
            if kernel_size % 2 != 0:
                padding += 1
            conv_block = [nn.Conv1d(input_channels, output_channels, kernel_size, stride=self.scale_factor,
                                    padding=padding, dilation=dilation, bias=self.gan_bias),
                          nn.LeakyReLU(negative_slope=0.2)]
            if self.phase_shuffle and layer_out_dim > 4:
                    conv_block.append(PhaseShuffle(shift_factor=2))
            self.conv_layers.append(nn.Sequential(*conv_block))

    def forward(self, x):
        for layer in range(self.model_levels):
            x = self.conv_layers[layer](x)
        x = torch.flatten(x, start_dim=1)
        out = self.output_layer(x)
        return out
