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
    """
    LSTMModel: A basic LSTM model with two stacked LSTM layers.

    Purpose:
        This model is suitable for capturing temporal dependencies in sequential data.
        It uses two LSTM layers to learn time-based patterns and a fully connected layer
        for classification.

    Parameters:
    - input_size (int): Dimension of the input features per timestep.
    - hidden_size (int): Number of hidden units in the LSTM layers.
    - num_classes (int): Number of output classes for the final classification.

    Architecture:
    - 2 LSTM layers with dropout for regularization.
    - Fully connected layers with ReLU activation to map LSTM output to class logits.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])  # Use the last output of the LSTM
        return self.fc2(out)


class DeepLSTMModel(nn.Module):
    """
    DeepLSTMModel: A deeper LSTM model with four LSTM layers and configurable dropout.

    Purpose:
        This model captures more complex temporal dependencies with four stacked LSTM layers,
        each reducing the hidden dimension to compress information progressively.

    Parameters:
    - input_size (int): Dimension of the input features per timestep.
    - hidden_size (int): Initial hidden size for the first LSTM layer, progressively reduced.
    - num_classes (int): Number of output classes.
    - dropout_rate (float): Dropout rate for regularization across layers.

    Architecture:
    - 4 LSTM layers with decreasing hidden sizes to compress and retain essential temporal information.
    - Fully connected layers with dropout and ReLU activation for classification.
    """
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2):
        super(DeepLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.lstm4 = nn.LSTM(hidden_size // 2, hidden_size // 4, num_layers=1, batch_first=True, dropout=dropout_rate)
        
        self.fc1 = nn.Linear(hidden_size // 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    
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
    - num_classes (int): Number of output classes.
    - num_heads (int): Number of attention heads in each encoder layer.
    - num_encoder_layers (int): Number of Transformer encoder layers.
    - d_model (int): Dimensionality of the model's embeddings.
    - dim_feedforward (int): Size of the feedforward layer in each encoder block.
    - dropout_rate (float): Dropout rate for regularization.
    - max_seq_length (int): Maximum length of input sequences.

    Architecture:
    - Input projection and positional encoding to enhance sequence representation.
    - Transformer encoder layers with self-attention to capture dependencies.
    - Fully connected layer to output class logits.
    """
    def __init__(self, input_size, num_classes, num_heads=4, num_encoder_layers=4, 
                 d_model=128, dim_feedforward=512, dropout_rate=0.1, max_seq_length=500):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, 
            dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)
    
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
    - num_classes (int): Number of output classes.
    - num_filters (int): Number of filters in the CNN layer.
    - kernel_size (int): Size of the convolutional kernel.
    - lstm_hidden_size (int): Number of hidden units in the LSTM layer.
    - num_lstm_layers (int): Number of stacked LSTM layers.
    - dropout_rate (float): Dropout rate for regularization.

    Architecture:
    - CNN layer with ReLU and pooling for spatial feature extraction.
    - LSTM layers for capturing temporal patterns in the sequence.
    - Fully connected layer for classification.
    """
    def __init__(self, input_size, num_classes, num_filters=64, kernel_size=3,
                 lstm_hidden_size=128, num_lstm_layers=2, dropout_rate=0.2):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                               kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, batch_first=True, dropout=dropout_rate)
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
    """
    CNNModel: A simple CNN model for spatial feature extraction.

    Purpose:
        This model is used to extract spatial features from sequence data with one convolutional
        layer, followed by global pooling and a fully connected layer for classification.

    Parameters:
    - input_size (int): Dimension of input features per timestep.
    - num_classes (int): Number of output classes.
    - num_filters (int): Number of filters in the CNN layer.
    - kernel_size (int): Size of the convolutional kernel.
    - dropout_rate (float): Dropout rate for regularization.

    Architecture:
    - Single CNN layer for basic spatial feature extraction.
    - Global average pooling and dropout.
    - Fully connected layer for final classification.
    """
    def __init__(self, input_size, num_classes, num_filters=64, kernel_size=3, dropout_rate=0.2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x.permute(0, 2, 1)))
        x = self.pool(x)
        x = x.mean(dim=2)  # Global average pooling
        x = self.dropout(x)
        return self.fc(x)


class DeepCNNModel(nn.Module):
    """
    DeepCNNModel: A deeper CNN model with four convolutional layers for complex spatial feature extraction.

    Purpose:
        This model captures spatial features at multiple levels of abstraction by using four
        convolutional layers with progressively increasing filters, pooling, and dropout.

    Parameters:
    - input_size (int): Dimension of input features per timestep.
    - num_classes (int): Number of output classes.
    - num_filters (int): Initial number of filters, doubled in each subsequent layer.
    - kernel_size (int): Size of the convolutional kernel.
    - dropout_rate (float): Dropout rate for regularization.

    Architecture:
    - 4 convolutional layers with ReLU and max pooling to capture deep spatial features.
    - Global average pooling and dropout for regularization.
    - Fully connected layer for classification.
    """
    def __init__(self, input_size, num_classes, num_filters=64, kernel_size=3, dropout_rate=0.2):
        super(DeepCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=kernel_size, padding=1)
        self.conv4 = nn.Conv1d(in_channels=num_filters * 4, out_channels=num_filters * 8, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
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
    """
    DeepCNNLSTMModel: A hybrid model combining multiple CNN and LSTM layers for deep feature extraction.

    Purpose:
        This model extracts spatial features using multiple CNN layers and then captures temporal
        dependencies with LSTM layers, providing a comprehensive approach for sequential data.

    Parameters:
    - input_size (int): Dimension of input features per timestep.
    - num_classes (int): Number of output classes.
    - num_filters (int): Initial number of filters, increased in each convolutional layer.
    - kernel_size (int): Size of the convolutional kernel.
    - lstm_hidden_size (int): Number of hidden units in the LSTM layer.
    - num_lstm_layers (int): Number of stacked LSTM layers.
    - dropout_rate (float): Dropout rate for regularization.

    Architecture:
    - 4 CNN layers for complex spatial feature extraction.
    - LSTM layers for capturing temporal dependencies.
    - Fully connected layer for classification.
    """
    def __init__(self, input_size, num_classes, num_filters=64, kernel_size=3,
                 lstm_hidden_size=128, num_lstm_layers=2, dropout_rate=0.2):
        super(DeepCNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=kernel_size, padding=1)
        self.conv4 = nn.Conv1d(in_channels=num_filters * 4, out_channels=num_filters * 8, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout_cnn = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=num_filters * 8, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, batch_first=True, dropout=dropout_rate)
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
