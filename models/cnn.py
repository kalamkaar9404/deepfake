# EfficientNet + Vision Transformer block for explainable attention
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel

class CNNTransformerAttention(nn.Module):
    def __init__(self):
        super(CNNTransformerAttention, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Identity()

        self.transformer = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        self.fc = nn.Linear(self.transformer.config.hidden_size, 512)

    def forward(self, x):
        cnn_features = self.efficientnet(x)  # shape: (B, 1280)
        vit_input = nn.functional.interpolate(x, size=(224, 224))
        transformer_outputs = self.transformer(pixel_values=vit_input)
        transformer_features = transformer_outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.fc(transformer_features)
        return x, transformer_outputs.attentions