%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai import *
from fastai.vision import *
from fastai.vision.gan import *
from fastai.callbacks import *
import matplotlib.animation as animation
from IPython.display import HTML

path = Path('G:/Mars/')

def get_data(bs, size, path, num_workers, noise_size=512):
    return (GANItemList.from_folder(path, noise_sz=noise_size)
               .split_none()
               .label_from_func(noop)
               .transform(tfms=[[crop_pad(size=size, row_pct=(0,1), col_pct=(0,1))], []], size=size, tfm_y=True)
               .databunch(bs=bs, num_workers=num_workers)
               .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))

data_4 = get_data(256, 4, path/'patches_4', 8)
data_8 = get_data(256, 8, path/'patches_8', 8)
data_16 = get_data(128, 16, path/'patches_16', 8)
data_32 = get_data(128, 32, path/'patches_32', 8)
data_64 = get_data(48, 64, path/'patches_64', 8)
data_128 = get_data(22, 128, path/'patches_128', 8)
data_256 = get_data(10, 256, path/'patches_256', 8)
data_512 = get_data(4, 512, path/'image_patches', 8)

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)
    
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        pixel_norm=True,
        spectral_norm=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
            EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        out = self.conv(input)

        return out
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out
    
 class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise
    
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out
class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            self.conv1 = EqualConv2d(
                in_channel, out_channel, kernel_size, padding=padding
            )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.adain1(out, style)
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.adain2(out, style)
        out = self.lrelu2(out)

        return out
class Generator(nn.Module):
    def __init__(self, code_dim):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True), # 4x4
                StyledConvBlock(512, 512, 3, 1), # 8x8
                StyledConvBlock(512, 512, 3, 1), # 16x16
                StyledConvBlock(512, 512, 3, 1), # 32x32
                StyledConvBlock(512, 256, 3, 1), # 64x64
                StyledConvBlock(256, 128, 3, 1), # 128x128
                StyledConvBlock(128, 64, 3, 1), # 256x256
                StyledConvBlock(64, 32, 3, 1), # 512x512
                StyledConvBlock(32, 16, 3, 1), # 1024x1024
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1), # 4x4
                EqualConv2d(512, 3, 1), # 8x8
                EqualConv2d(512, 3, 1), # 16x16
                EqualConv2d(512, 3, 1), # 32x32
                EqualConv2d(256, 3, 1), # 64x64
                EqualConv2d(128, 3, 1), # 128x128
                EqualConv2d(64, 3, 1), # 256x256
                EqualConv2d(32, 3, 1), # 512x512
                EqualConv2d(16, 3, 1), # 1024x1024
            ]
        )
        
    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = random.sample(list(range(step)), len(style) - 1)

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                upsample = F.interpolate(
                    out, scale_factor=2, mode='bilinear', align_corners=False
                )
                out = conv(upsample, style_step, noise[i])

            else:
                out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and alpha > 0:
                    skip_rgb = self.to_rgb[i - 1](upsample)
                    out = alpha * skip_rgb + (1 - alpha) * out

                break

        return out
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)
        
        self.step = 0
        self.alpha = 0

    def forward(self, input, noise=None, mean_style=None, style_weight=0, mixing_range=(-1, -1), mixing=True):
        
        bs, ch, _, _ = input.shape
        input = input.view(bs, ch)
        
        if self.step < 1:
            mixing=False
        
        if mixing and random.random() < 0.9:
            shuffle = torch.randperm(input.size(0)).to(input.device)
            input = [input, input[shuffle]]
        else:
            input = [input]  
        
        styles = []

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(self.step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(styles, noise, self.step, self.alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style
        
    def grow_model(self, step=None, alpha=None):
        if step:
            self.step = step
        else:
            self.step += 1
            
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = 1
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1),
                ConvBlock(32, 64, 3, 1),
                ConvBlock(64, 128, 3, 1),
                ConvBlock(128, 256, 3, 1),
                ConvBlock(256, 512, 3, 1),
                ConvBlock(512, 512, 3, 1),
                ConvBlock(512, 512, 3, 1),
                ConvBlock(512, 512, 3, 1),
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        self.from_rgb = nn.ModuleList(
            [
                EqualConv2d(3, 16, 1),
                EqualConv2d(3, 32, 1),
                EqualConv2d(3, 64, 1),
                EqualConv2d(3, 128, 1),
                EqualConv2d(3, 256, 1),
                EqualConv2d(3, 512, 1),
                EqualConv2d(3, 512, 1),
                EqualConv2d(3, 512, 1),
                EqualConv2d(3, 512, 1),
            ]
        )


        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)
        
        self.step = 0
        self.alpha = 0

    def forward(self, input, actual=False):
        if actual:
            for i in range(self.step, -1, -1):
                index = self.n_layer - i - 1

                if i == self.step:
                    out = self.from_rgb[index](input)

                if i == 0:
                    out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                    mean_std = out_std.mean()
                    mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                    out = torch.cat([out, mean_std], 1)

                out = self.progression[index](out)

                if i > 0:
                    out = F.interpolate(
                        out, scale_factor=0.5, mode='bilinear', align_corners=False
                    )

                    if i == self.step and self.alpha > 0:  #0 <= alpha < 1:
                        skip_rgb = self.from_rgb[index + 1](input)
                        skip_rgb = F.interpolate(
                            skip_rgb, scale_factor=0.5, mode='bilinear', align_corners=False
                        )

                        out = self.alpha * skip_rgb + (1 - self.alpha) * out

            out = out.squeeze(2).squeeze(2)
            out = self.linear(out)

            return (out, input)
        else:
            return (None, input)
    
    def grow_model(self, step=None, alpha=None):
        if step:
            self.step = step
        else:
            self.step += 1
            
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = 1
