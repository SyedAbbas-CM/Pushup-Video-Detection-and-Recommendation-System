import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class EnhancedConvTransformer(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedConvTransformer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.embedding_dim = 256 * 4 * 4
        self.num_heads = 8
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers=12)

        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.conv(x)
        x = x.view(batch_size, timesteps, -1)

        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)

        x = self.fc(x[:, -1, :])
        return x


num_classes = 2  # Exercise and Non-Exercise
model = EnhancedConvTransformer(num_classes)
print(model)
