import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
# --- CELL 3: MULTI-TASK U-NET ARCHITECTURE ---
class WinningFusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-Trained Backbone (Takes 7 channels, outputs 64-channel features)
        self.base_model = smp.Unet(
            encoder_name="resnet34",        
            encoder_weights="imagenet",     
            in_channels=7,                  
            classes=64, 
        )
        
        # The Two Output Heads
        self.footprint_head = nn.Conv2d(64, 1, kernel_size=1)
        self.height_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        features = self.base_model(x)
        
        # Head 1: Probability of being a building (0.0 to 1.0)
        footprint_prob = torch.sigmoid(self.footprint_head(features))
        
        # Head 2: Raw height prediction in meters
        raw_height = torch.relu(self.height_head(features))
        
        # SEMANTIC REFINEMENT: Multiply height by footprint probability
        refined_height = raw_height * footprint_prob
        
        return footprint_prob, refined_height