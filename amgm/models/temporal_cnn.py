import torch.nn as nn
import torch

class TemporalCNNTrendModel(nn.Module):
    def __init__(self, input_length, n_features, conv_channels=64, kernel_size=5, conv_layers=3, output_size=1, dropout=0.1, pooling=2):
        """
        """
        # Building layers
        super().__init__()
        self.conv_blocks = []

        for l in range(conv_layers):            
            if l == 0:
                self.conv_blocks.append(nn.Conv1d(n_features, conv_channels, kernel_size, padding=kernel_size//2))
            else:
                self.conv_blocks.append(nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size//2))
            self.conv_blocks.append(nn.BatchNorm1d(conv_channels))
            self.conv_blocks.append(nn.MaxPool1d(pooling))
        # Creating the nn sequential model
        self.conv_stack = nn.Sequential(*self.conv_blocks)
        
        # Dynamically compute the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_length, n_features)
            dummy = dummy.transpose(1, 2)  # (1, n_features, input_length)
            dummy = self.conv_stack(dummy)
            flat_size = dummy.flatten(1).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
        self.conv_blocks.append(nn.Flatten())   # flatten all time features
        self.conv_blocks.append(self.fc)   # flatten all time features
        self.classifier = nn.Sequential(*self.conv_blocks)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, feats, time)
        out = self.classifier(x)
        return out  # match target shape