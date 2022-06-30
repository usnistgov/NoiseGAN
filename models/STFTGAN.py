import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


weight_init_dispatcher = {'uniform': nn.init.uniform_,
                          'normal': nn.init.normal_,
                          'xavier_uniform': nn.init.xavier_uniform_,
                          'xavier_normal': nn.init.xavier_normal_,
                          'kaiming_uniform': nn.init.kaiming_uniform_,
                          'kaiming_normal': nn.init.kaiming_normal_,
                          'orthogonal': nn.init.orthogonal_}


class ScaledTanh(nn.Module):
    def __init__(self, scaler):
        super(ScaledTanh, self).__init__()
        self.scaler = scaler
        self.activation = nn.Tanh()

    def forward(self, input):
        out = self.activation(input)
        out = out * self.scaler
        return out


class G(nn.Module):
    def __init__(self,  **kwargs):
        super(G, self).__init__()
        self.__dict__.update(kwargs)  # unpack the dictionary such that each key becomes self.key
        input_size = self.lower_resolution[0] * self.lower_resolution[1] * self.G_channel_list[0]
        self.input_layer = nn.Linear(self.latent_dim, input_size, bias=True)
        self.conv_layers = nn.ModuleList()

        if "quantized" not in self.__dict__:
            self.quantized = False
        self.build_model()
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                weight_init_dispatcher[self.weight_init](m.weight.data)

    def build_model(self):
        for layer_num in range(self.model_levels):
            input_channels, output_channels = self.G_channel_list[layer_num], self.G_channel_list[layer_num + 1]
            kernel_size, dilation = (self.kernel_size, self.kernel_size), 1

            output_padding = 1 if dilation > 1 or self.scale_factor == 1 else 0
            padding = [((kernel_dim - 1) * dilation + output_padding) // 2 - (self.scale_factor - 1) // 2 for kernel_dim in kernel_size]
            output_padding = (1, 1)
            last_layer = True if layer_num == self.model_levels - 1 else False
            if layer_num == self.model_levels - 1:
                output_padding = [0 if self.input_dimension_parity[i] == "odd" else 1 for i in range(2)]
                padding = [padding[i] - 1 if self.input_dimension_parity[i] == "odd" else padding[i] for i in range(2)]
            stride = (self.scale_factor, self.scale_factor)
            conv_block = [nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding, bias=self.gan_bias)]
            if last_layer and self.use_tanh:
                activation = nn.Tanh() if not self.quantized else ScaledTanh(self.max_abs_limit)
                conv_block.append(activation)
            else:
                conv_block.append(nn.ReLU())
            self.conv_layers.append(nn.Sequential(*conv_block))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        x = self.input_layer(z)
        x = x.view(x.size(0), -1, self.lower_resolution[0], self.lower_resolution[1])
        for layer_number in range(self.model_levels):
            x = self.conv_layers[layer_number](x)
        return x


class D(nn.Module):
    def __init__(self, **kwargs):
        super(D, self).__init__()
        self.__dict__.update(kwargs)  # unpack the dictionary such that each key becomes self.key
        self.gan_bias = True
        self.input_dim = [lower_res * (self.scale_factor ** self.model_levels) for lower_res in self.lower_resolution]
        output_size = self.lower_resolution[0] * self.lower_resolution[1] * self.D_channel_list[-1]
        self.output_layer = nn.Linear(output_size, 1, bias=True)
        self.conv_layers = nn.ModuleList()
        self.build_model()
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                weight_init_dispatcher[self.weight_init](m.weight.data)

    def build_model(self):
        for layer_num in range(self.model_levels):
            input_channels = self.D_channel_list[layer_num]
            output_channels = self.D_channel_list[layer_num + 1]
            kernel_size, dilation = (self.kernel_size, self.kernel_size), 1
            padding = [((kernel_dim - 1) * dilation) // 2 - (self.scale_factor - 1) // 2 for kernel_dim in kernel_size]
            if layer_num == 0:
                padding = [padding[i] - 1 if self.input_dimension_parity[i] == "odd" else padding[i] for i in range(2)]
            stride = (self.scale_factor, self.scale_factor)
            conv_block = [nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding,
                                        bias=self.gan_bias), nn.LeakyReLU(negative_slope=0.2)]
            self.conv_layers.append(nn.Sequential(*conv_block))

    def forward(self, x):
        for layer in range(self.model_levels):
            x = self.conv_layers[layer](x)
        x = torch.flatten(x, start_dim=1)
        out = self.output_layer(x)
        return out
