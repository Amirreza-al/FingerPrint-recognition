import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_name="resnet18", embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        # Load a pretrained model as feature extractor
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            
            # Modify the first layer to accept 1-channel input instead of 3
            original_layer = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, 64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
            
            # Initialize the new layer with weights from the pretrained model
            # by averaging across the channel dimension
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    original_layer.weight.mean(dim=1, keepdim=True)
                )
                
            num_ftrs = self.backbone.fc.in_features
            # Remove the last FC layer
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError("Only 'resnet18' is implemented here for demonstration.")
        
        # Add a new FC layer to reduce to the embedding dimension
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward_once(self, x):
        # Extract features from backbone
        x = self.backbone(x)
        # Reduce to embedding
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        # Extract features using shared weights
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)
        return emb1, emb2

class ContrastiveLoss(nn.Module):
    """ Contrastive loss """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # Calculate Euclidean distance
        dist = nn.functional.pairwise_distance(emb1, emb2)
        # Contrastive loss formula
        loss = 0.5 * (label * dist.pow(2) + (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0), 2))
        return loss.mean()