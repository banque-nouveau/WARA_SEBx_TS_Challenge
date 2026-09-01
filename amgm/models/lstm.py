import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0,
       
    ):
        """
        The input to every LSTM cell is a 3D tensor of shape (batch_size, n_features).
        """
        super().__init__()
        self.n_features = n_features
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_size, output_size)
        self.output_shape = (output_size)
        

    def forward(self, x):
        """
        x: (batch, input_length, n_features)
        returns:
          -  (batch, output_size)
        """
        out, _ = self.lstm(x)          # (batch, input_length, hidden)
        last = out[:, -1, :]           # (batch, hidden)
        y = self.head(last)            # (batch, output_size)
        return y