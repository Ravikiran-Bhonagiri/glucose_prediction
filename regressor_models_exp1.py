"""
models.py

This file contains definitions for various neural network models designed for regression
tasks on sequential data, such as time series forecasting. These models are built to capture
complex spatial and temporal dependencies using combinations of convolutional, LSTM, and
Transformer layers.

Models included:
    1. LSTMModel: A basic LSTM-based model with stacked LSTM layers.
    2. TransformerModel: A Transformer-based model for time series regression.
    3. CNNLSTMModel: A hybrid model combining CNN and LSTM layers.
    4. CNNModel: A simple CNN model for spatial feature extraction in regression tasks.

Author: Ravikiran Bhonagiri
Date: 17th November, 2024
"""

import torch
import torch.nn as nn
import math


# -------------------------------------------------------------------------------------
# LSTMModel: A basic LSTM model with two stacked LSTM layers.
# -------------------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=dropout, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use the last output
        out = torch.relu(self.fc1(out))
        out = self.dropout(torch.relu(self.fc2(out)))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        return self.fc5(out)


# -------------------------------------------------------------------------------------
# PositionalEncoding: Adds positional information to input embeddings for Transformer layers.
# -------------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
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


# -------------------------------------------------------------------------------------
# TransformerModel: A model utilizing Transformer encoder layers to process sequential data.
# -------------------------------------------------------------------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads=4, num_encoder_layers=4, d_model=128, dim_feedforward=512, dropout=0.1, max_seq_length=500):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # Fully connected layers
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        out = x[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        return self.fc5(out)


# -------------------------------------------------------------------------------------
# CNNLSTMModel: A hybrid model combining CNN and LSTM layers for spatial and temporal dependencies.
# -------------------------------------------------------------------------------------
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_size=3, lstm_hidden_size=128, num_lstm_layers=2, dropout=0.2):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        self.lstm = nn.LSTM(num_filters, lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True, dropout=dropout)
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        out = hn[-1]
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        return self.fc5(out)


# -------------------------------------------------------------------------------------
# CNNModel: A simple CNN model for spatial feature extraction.
# -------------------------------------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_size=3, dropout=0.2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x.permute(0, 2, 1)))
        x = self.pool(x)
        x = x.mean(dim=2)
        x = self.dropout(x)
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        return self.fc5(out)
