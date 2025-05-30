import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Align spatial dimensions before concatenation if needed
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNet(nn.Module):
    def __init__(self, out_channels=1):
        super(SwinUNet, self).__init__()
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            img_size=512,
        )
        enc_channels = [f["num_chs"] for f in self.encoder.feature_info]  # [96, 192, 384, 768]

        self.decoder3 = DecoderBlock(
            in_channels=enc_channels[3], skip_channels=enc_channels[2], out_channels=enc_channels[2]
        )
        self.decoder2 = DecoderBlock(
            in_channels=enc_channels[2], skip_channels=enc_channels[1], out_channels=enc_channels[1]
        )
        self.decoder1 = DecoderBlock(
            in_channels=enc_channels[1], skip_channels=enc_channels[0], out_channels=enc_channels[0]
        )
        
        # Adjust decoder0: Remove the second ConvTranspose2d to reduce extra upsampling
        self.decoder0 = nn.Sequential(
            nn.ConvTranspose2d(enc_channels[0], 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        enc_feats = self.encoder(x)
        # Sometimes encoder outputs in NHWC; permute if needed:
        enc_feats = [f.permute(0, 3, 1, 2).contiguous() if f.dim() == 4 else f for f in enc_feats]

        x = self.decoder3(enc_feats[3], enc_feats[2])
        x = self.decoder2(x, enc_feats[1])
        x = self.decoder1(x, enc_feats[0])
        x = self.decoder0(x)

        x = self.out_conv(x)

        # Upsample to input size dynamically to ensure output size == input size
        x = F.interpolate(x, size=(input_h, input_w), mode='bilinear', align_corners=False)

        return x


def build_swin_unet(input_shape=(3, 512, 512), out_channels=1):
    return SwinUNet(out_channels=out_channels)