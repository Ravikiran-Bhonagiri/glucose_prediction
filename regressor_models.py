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
Date: 9th November, 2024
"""

import torch
import torch.nn as nn
import math


class LSTMModel(nn.Module):
    """
    LSTMModel: A basic LSTM model with two stacked LSTM layers.

    Purpose:
        This model is suitable for capturing temporal dependencies in sequential data.
        It uses two LSTM layers to learn time-based patterns and a fully connected layer
        for regression.

    Parameters:
    - input_size (int): Dimension of the input features per timestep.
    - hidden_size (int): Number of hidden units in the LSTM layers.
    
    Architecture:
    - 2 LSTM layers with dropout for regularization.
    - Fully connected layers with ReLU activation to map LSTM output to a regression output.
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)  # Single output for regression
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])  # Use the last output of the LSTM
        return self.fc2(out)


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding: Adds positional information to input embeddings for Transformer layers.

    Purpose:
        Positional encoding provides sequence order information to the Transformer model,
        allowing it to process temporal or ordered data.

    Parameters:
    - d_model (int): Dimensionality of the model embeddings.
    - max_seq_length (int): Maximum length of the input sequence.
    - dropout_rate (float): Dropout rate for regularization of positional encoding.
    """
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
    """
    TransformerModel: A model utilizing Transformer encoder layers to process sequential data.

    Purpose:
        This model captures long-range dependencies in sequential data through self-attention mechanisms.

    Parameters:
    - input_size (int): Number of input features per timestep.
    - num_heads (int): Number of attention heads in each encoder layer.
    - num_encoder_layers (int): Number of Transformer encoder layers.
    - d_model (int): Dimensionality of the model's embeddings.
    - dim_feedforward (int): Size of the feedforward layer in each encoder block.
    - dropout_rate (float): Dropout rate for regularization.
    - max_seq_length (int): Maximum length of input sequences.

    Architecture:
    - Input projection and positional encoding to enhance sequence representation.
    - Transformer encoder layers with self-attention to capture dependencies.
    - Fully connected layer to output a single regression value.
    """
    def __init__(self, input_size, num_heads=4, num_encoder_layers=4, 
                 d_model=128, dim_feedforward=512, dropout_rate=0.1, max_seq_length=500):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, 
            dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)  # Single output for regression
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])


class CNNLSTMModel(nn.Module):
    """
    CNNLSTMModel: A hybrid model combining CNN and LSTM layers for spatial and temporal dependencies.

    Purpose:
        This model first applies a CNN layer to extract spatial features, followed by LSTM
        layers to capture temporal dependencies, making it effective for structured sequence data.

    Parameters:
    - input_size (int): Number of input features per timestep.
    - num_filters (int): Number of filters in the CNN layer.
    - kernel_size (int): Size of the convolutional kernel.
    - lstm_hidden_size (int): Number of hidden units in the LSTM layer.
    - num_lstm_layers (int): Number of stacked LSTM layers.
    - dropout_rate (float): Dropout rate for regularization.

    Architecture:
    - CNN layer with ReLU and pooling for spatial feature extraction.
    - LSTM layers for capturing temporal patterns in the sequence.
    - Fully connected layer for regression.
    """
    def __init__(self, input_size, num_filters=64, kernel_size=3,
                 lstm_hidden_size=128, num_lstm_layers=2, dropout_rate=0.2):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                               kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(lstm_hidden_size, 1)  # Single output for regression
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class CNNModel(nn.Module):
    """
    CNNModel: A simple CNN model for spatial feature extraction.

    Purpose:
        This model is used to extract spatial features from sequence data with one convolutional
        layer, followed by global pooling and a fully connected layer for regression.

    Parameters:
    - input_size (int): Dimension of input features per timestep.
    - num_filters (int): Number of filters in the CNN layer.
    - kernel_size (int): Size of the convolutional kernel.
    - dropout_rate (float): Dropout rate for regularization.

    Architecture:
    - Single CNN layer for basic spatial feature extraction.
    - Global average pooling and dropout.
    - Fully connected layer for final regression output.
    """
    def __init__(self, input_size, num_filters=64, kernel_size=3, dropout_rate=0.2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters, 1)  # Single output for regression
    
    def forward(self, x):
        x = torch.relu(self.conv1(x.permute(0, 2, 1)))
        x = self.pool(x)
        x = x.mean(dim=2)  # Global average pooling
        x = self.dropout(x)
        return self.fc(x)
