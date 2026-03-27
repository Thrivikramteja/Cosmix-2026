import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp # Make sure you pip installed this!

# ==========================================
# ADVANCED MODEL SUB-COMPONENTS
# ==========================================
class PretrainedCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) 
        self.layer1 = resnet.layer1 
        self.layer2 = resnet.layer2 
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4 
        
        self.up_align = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

    def forward(self, x):
        f0 = self.initial(x)  
        f1 = self.layer1(f0)  
        f2 = self.layer2(f1)  
        f3 = self.layer3(f2)  
        f4 = self.layer4(f3)  
        f_out = self.up_align(f4) 
        return f0, f1, f2, f_out

class CrossFusionBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.norm1_sar = nn.LayerNorm(embed_dim)
        self.norm1_opt = nn.LayerNorm(embed_dim)
        
        self.cross_attn_sar = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_opt = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm2_sar = nn.LayerNorm(embed_dim)
        self.norm2_opt = nn.LayerNorm(embed_dim)
        
        self.mlp_sar = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )
        self.mlp_opt = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, sar_feat, opt_feat):
        B, C, H, W = sar_feat.shape
        sar_flat = sar_feat.flatten(2).transpose(1, 2)
        opt_flat = opt_feat.flatten(2).transpose(1, 2)
        
        sar_norm = self.norm1_sar(sar_flat)
        opt_norm = self.norm1_opt(opt_flat)
        
        attn_sar, _ = self.cross_attn_sar(sar_norm, opt_norm, opt_norm)
        sar_out = sar_flat + attn_sar
        
        attn_opt, _ = self.cross_attn_opt(opt_norm, sar_norm, sar_norm)
        opt_out = opt_flat + attn_opt
        
        sar_out = sar_out + self.mlp_sar(self.norm2_sar(sar_out))
        opt_out = opt_out + self.mlp_opt(self.norm2_opt(opt_out))
        
        sar_out = sar_out.transpose(1, 2).view(B, C, H, W)
        opt_out = opt_out.transpose(1, 2).view(B, C, H, W)
        
        return sar_out, opt_out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ==========================================
# MAIN ADVANCED MODEL
# ==========================================
class FusionHeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_opt = PretrainedCNNEncoder()
        self.enc_sar = PretrainedCNNEncoder()
        self.cross_fusion = CrossFusionBlock(embed_dim=256)
        self.fusion_conv = nn.Conv2d(512, 256, 1) 
        
        self.foot_dec1 = DecoderBlock(256, 128, 128) 
        self.foot_dec2 = DecoderBlock(128, 64, 64)   
        self.foot_dec3 = DecoderBlock(64, 64, 64)    
        self.foot_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'), 
            nn.Conv2d(32, 2, 1) 
        )
        
        self.height_dec1 = DecoderBlock(256, 128, 128)
        self.height_dec2 = DecoderBlock(128, 64, 64)
        self.height_dec3 = DecoderBlock(64, 64, 64)
        self.height_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid() 
        )

    def forward(self, opt, sar):
        o_f0, o_f1, o_f2, o_fout = self.enc_opt(opt)
        s_f0, s_f1, s_f2, s_fout = self.enc_sar(sar)
        
        s_fused, o_fused = self.cross_fusion(s_fout, o_fout)
        f_fusion = self.fusion_conv(torch.cat([o_fused, s_fused], dim=1))
        
        d_foot = self.foot_dec1(f_fusion, o_f2)
        d_foot = self.foot_dec2(d_foot, o_f1)
        d_foot = self.foot_dec3(d_foot, o_f0)
        pred_footprint = self.foot_head(d_foot)
        
        d_height = self.height_dec1(f_fusion, s_f2)
        d_height = self.height_dec2(d_height, s_f1)
        d_height = self.height_dec3(d_height, s_f0)
        base_height = self.height_head(d_height)
        
        return pred_footprint, base_height


# ==========================================
# MAIN BASIC MODEL
# ==========================================
class WinningFusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = smp.Unet(
            encoder_name="resnet34",        
            encoder_weights="imagenet",     
            in_channels=7,                  
            classes=64, 
        )
        self.footprint_head = nn.Conv2d(64, 1, kernel_size=1)
        self.height_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        features = self.base_model(x)
        footprint_prob = torch.sigmoid(self.footprint_head(features))
        raw_height = torch.relu(self.height_head(features))
        refined_height = raw_height * footprint_prob
        return footprint_prob, refined_height