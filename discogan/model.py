import torch
import torch.nn as nn

# --- Building blocks ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# --- Generator (U-Net-lite) ---
class Generator(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super().__init__()
        # encoder
        self.e1 = ConvBlock(in_c, 64, norm=False)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.e5 = ConvBlock(512, 512)

        # decoder
        self.d1 = DeconvBlock(512, 512, dropout=True)              # OK
        self.d2 = DeconvBlock(512 + 512, 256, dropout=True)        # 1024 -> 256
        self.d3 = DeconvBlock(256 + 256, 128)                      # FIX: 512 -> 128
        self.d4 = DeconvBlock(128 + 128, 64)                       # FIX: 256 -> 64
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_c, 4, 2, 1),           # FIX: 128 -> out_c
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        d1 = self.d1(e5)
        d2 = self.d2(torch.cat([d1, e4], dim=1))
        d3 = self.d3(torch.cat([d2, e3], dim=1))
        d4 = self.d4(torch.cat([d3, e2], dim=1))
        out = self.out(torch.cat([d4, e1], dim=1))
        return out

# --- Discriminator (PatchGAN) ---
class Discriminator(nn.Module):
    def __init__(self, in_c=3):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_c, 64, norm=False),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1)
        )
    def forward(self, x): return self.net(x)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
