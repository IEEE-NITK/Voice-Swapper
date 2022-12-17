from .unet_parts import *

class Generator(nn.Module):
  def __init__(self, in_channels, out_channels, bilinear=False):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.bilinear = bilinear

    self.inc = (DoubleConv(in_channels, 64))
    self.encode1 = (Encode(64, 128))
    self.encode2 = (Encode(128, 256))
    self.encode3 = (Encode(256, 512))
    factor = 2 if bilinear else 1
    self.encode4 = (Encode(512, 1024 // factor))
    self.decode1 = (Decode(1024, 512 // factor, bilinear))
    self.decode2 = (Decode(512, 256 // factor, bilinear))
    self.decode3 = (Decode(256, 128 // factor, bilinear))
    self.decode4 = (Decode(128, 64, bilinear))
    self.outc = (OutConv(64, out_channels))
  
  def forward(self, x):
    x1 = self.inc(x)
    x2 = self.encode1(x1)
    x3 = self.encode2(x2)
    x4 = self.encode3(x3)
    x5 = self.encode4(x4)
    x = self.decode1(x5, x4)
    x = self.decode2(x, x3)
    x = self.decode3(x, x2)
    x = self.decode4(x, x1)
    logits = self.outc(x)
    return logits
