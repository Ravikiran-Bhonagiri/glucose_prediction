"""
models_classifier.py

This file contains definitions for various neural network models designed for sequential
data processing tasks, such as time series and NLP applications. These models are designed
to capture complex spatial and temporal dependencies in data through combinations of
convolutional, LSTM, and Transformer layers.

Models included:
    1. LSTMModel: A basic LSTM-based model with stacked LSTM layers.
    2. DeepLSTMModel: A deeper LSTM model with multiple layers and configurable dropout.
    3. TransformerModel: A Transformer-based model for time series data.
    4. CNNLSTMModel: A hybrid model combining CNN and LSTM layers.
    5. CNNModel: A simple CNN model for spatial feature extraction.
    6. DeepCNNModel: A deeper CNN model with multiple convolutional layers.
    7. DeepCNNLSTMModel: A hybrid model combining multiple CNN and LSTM layers for deep spatial and temporal feature extraction.

Author: Ravikiran Bhonagiri
Date: 5th November, 2024
"""

import torch
import torch.nn as nn
import math


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])  # Use the last output of the LSTM
        return self.fc2(out)


class DeepLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_classes):
        super(DeepLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, batch_first=True, dropout=dropout)
        self.lstm4 = nn.LSTM(hidden_size // 2, hidden_size // 4, num_layers=1, batch_first=True, dropout=dropout)
        
        self.fc1 = nn.Linear(hidden_size // 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        
        out = out[:, -1, :]
        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.dropout(torch.relu(self.fc2(out)))
        return self.fc3(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout_rate=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_encoder_layers, dim_feedforward, dropout, max_seq_length, num_classes):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, lstm_hidden_size, dropout, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, dropout, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x.permute(0, 2, 1)))
        x = self.pool(x)
        x = x.mean(dim=2)
        x = self.dropout(x)
        return self.fc(x)


class DeepCNNModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, dropout, num_classes):
        super(DeepCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=kernel_size, padding=1)
        self.conv4 = nn.Conv1d(in_channels=num_filters * 4, out_channels=num_filters * 8, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * 8, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x.permute(0, 2, 1)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.mean(dim=2)
        x = self.dropout(x)
        return self.fc(x)


class DeepCNNLSTMModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, lstm_hidden_size, dropout, num_classes):
        super(DeepCNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=kernel_size, padding=1)
        self.conv4 = nn.Conv1d(in_channels=num_filters * 4, out_channels=num_filters * 8, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout_cnn = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=num_filters * 8, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.dropout_cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
