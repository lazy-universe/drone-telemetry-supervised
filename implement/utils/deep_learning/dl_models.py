import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        logits = self.fc(out[:, -1, :])
        return logits

class TCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(TCNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.conv_res = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x_trans = x.transpose(1, 2)
        
        # Main path
        out = self.conv1(x_trans)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        
        # Residual path
        res = self.conv_res(x_trans)
        
        # Combine
        out = out + res
        
        out = self.global_pool(out)
        out = out.squeeze(-1)
        logits = self.fc(out)
        return logits

class CNNGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(CNNGRUClassifier, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        x_trans = x.transpose(1, 2)
        c_out = self.conv(x_trans)
        c_out = self.relu(c_out)
        c_out = c_out.transpose(1, 2)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(c_out, h0)
        logits = self.fc(out[:, -1, :])
        return logits

class CNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        x_trans = x.transpose(1, 2)
        out = self.conv1(x_trans)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.global_pool(out).squeeze(-1)
        logits = self.fc(out)
        return logits
