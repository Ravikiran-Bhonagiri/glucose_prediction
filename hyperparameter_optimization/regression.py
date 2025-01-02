import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd

from config import heart_rate_features_1, heart_rate_features_2, \
                    sleep_features_1, sleep_features_2, \
                    intensity_features_1, intensity_features_2, \
                    steps_features_1, steps_features_2,  temporal_features \
                    
features_1 =  heart_rate_features_1 + sleep_features_1 + intensity_features_1 + steps_features_1 + temporal_features
features =  features_1 + heart_rate_features_2 + sleep_features_2 + intensity_features_2 + steps_features_2


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
    def __init__(self, input_size, num_heads, num_encoder_layers, d_model, dim_feedforward,
                 dropout, max_seq_length):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward,
                                                   dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
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
# Custom Loss Function: MAPE (Mean Absolute Percentage Error)
# -------------------------------------------------------------------------------------
def mape_loss(output, target):
    """
    Custom MAPE loss function.
    Computes Mean Absolute Percentage Error between output and target.
    """
    epsilon = 1e-8  # Small value to prevent division by zero
    return torch.mean(torch.abs((target - output) / (target + epsilon))) * 100


# -------------------------------------------------------------------------------------
# Training Function: Includes advanced hyperparameter handling.
# -------------------------------------------------------------------------------------
optimizer_mapping = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop
}

def train_transformer_model(config, data_loader, val_loader):
    # Initialize model
    model = TransformerModel(
        input_size=10,
        num_heads=config["num_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        d_model=config["d_model"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_seq_length=config["max_seq_length"]
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Select optimizer
    optimizer_class = optimizer_mapping[config["optimizer"]]
    optimizer = optimizer_class(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        momentum=config.get("momentum", 0.0)
    )

    # Learning rate scheduler
    if config["lr_scheduler"] == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif config["lr_scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    elif config["lr_scheduler"] == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    # Select loss function
    loss_function_mapping = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.SmoothL1Loss(),
        "mape": mape_loss  # Custom MAPE loss
    }
    criterion = loss_function_mapping[config["loss_function"]]

    # Training loop
    for epoch in range(100):  # Adjust as needed
        model.train()
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()

            # Gradient clipping
            if config["gradient_clipping"] > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])

            optimizer.step()
        scheduler.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_mape = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = model(inputs)
            
            # Accumulate validation loss (based on the chosen criterion)
            val_loss += criterion(outputs.squeeze(), targets).item()
            
            # Compute MAPE for evaluation
            epsilon = 1e-7  # Prevent division by zero
            mape = torch.mean(torch.abs((targets - outputs.squeeze()) / (targets + epsilon))) * 100
            val_mape += mape.item()

    # Report MAPE to Ray Tune
    tune.report(validation_loss=val_loss / len(val_loader), validation_mape=val_mape / len(val_loader))



# -------------------------------------------------------------------------------------
# Hyperparameter Search Space: Includes advanced parameters for fine-grained tuning.
# -------------------------------------------------------------------------------------
search_space = {
    # Core Transformer hyperparameters
    "num_heads": tune.choice([2, 4, 8, 16, 32]),
    "num_encoder_layers": tune.choice([2, 4, 6, 8, 12, 16]),
    "d_model": tune.choice([64, 128, 192, 256, 320, 512]),
    "dim_feedforward": tune.sample_from(lambda spec: spec.config["d_model"] * tune.choice([2, 3, 4])),
    "dropout": tune.loguniform(0.01, 0.5),
    "learning_rate": tune.loguniform(1e-5, 5e-2),
    "batch_size": tune.choice([16, 32, 64, 128, 256]),
    "weight_decay": tune.loguniform(1e-6, 1e-1),
    "max_seq_length": tune.choice([50, 100, 200, 500, 1000, 2000]),

    # Advanced exploration
    "activation_function": tune.choice(["relu", "gelu", "tanh", "elu", "leaky_relu"]),
    "initialization": tune.choice(["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]),
    "lr_scheduler": tune.choice(["constant", "linear", "cosine_annealing", "exponential", "step"]),
    "gradient_clipping": tune.uniform(0.0, 1.0),

    # Optimizer selection
    "optimizer": tune.choice(["adam", "adamw", "sgd", "rmsprop"]),
    "momentum": tune.uniform(0.0, 0.9),

    # Loss function
    "loss_function": tune.choice(["mse", "mae", "huber", "mape"]),

    # Layer Normalization vs Batch Normalization
    "normalization": tune.choice(["layer_norm", "batch_norm", "group_norm"]),
    "num_groups": tune.sample_from(lambda spec: 4 if spec.config["normalization"] == "group_norm" else 1),

    # Attention mechanism enhancements
    "attention_dropout": tune.loguniform(0.01, 0.4),
    "attention_type": tune.choice(["scaled_dot_product", "multi_head_attention", "linear_attention"]),
    "key_dim_scaling": tune.choice([1.0, 0.5, 0.25]),

    # Regularization enhancements
    "label_smoothing": tune.uniform(0.0, 0.2),
    "stochastic_depth_rate": tune.uniform(0.0, 0.2),

    # Positional encoding options
    "positional_encoding_type": tune.choice(["absolute", "relative"]),
    "relative_position_clip": tune.choice([16, 32, 64]),

    # Multi-headed Attention variants
    "multi_head_type": tune.choice(["vanilla", "cosformer", "performer", "reformer"]),
    "attn_kernel_size": tune.choice([3, 5, 7]),

    # Deep architecture tweaks
    "depthwise_separable_conv": tune.choice([True, False]),
    "shared_weights": tune.choice([True, False]),

    # Training loop controls
    "warmup_steps": tune.choice([100, 500, 1000, 2000]),
    "total_steps": tune.choice([10000, 20000, 50000, 100000]),
}


# Function to split data into intervals
def split_into_intervals(data, interval_size, stride):
    intervals = []
    num_intervals = (data.shape[0] - interval_size) // stride + 1
    for i in range(num_intervals):
        start_ix = i * stride
        end_ix = start_ix + interval_size
        interval = data[start_ix:end_ix]
        intervals.append(interval)
    return np.array(intervals)

# Function to load data from .npy files
def load_dataframe_from_npy(file_path):
    data_dict = np.load(file_path, allow_pickle=True).item()
    columns = data_dict['columns']
    data = data_dict['data']
    return pd.DataFrame(data, columns=columns)

def get_data_loaders(id_):

    features_data_path = f'/home/rxb2495/Glucose-Forecasting/final_data/{id_}_combined_data.npy'
    labels_data_path = f'/home/rxb2495/Glucose-Forecasting/final_data/{id_}_main_data.npy'
    
    intervals = load_dataframe_from_npy(features_data_path)
    output_data = load_dataframe_from_npy(labels_data_path)

    intervals = intervals[features]

    intervals = intervals.loc[:, ~intervals.columns.duplicated()]


    output_data = output_data["Historic Glucose mg/dL"].values.astype(np.float32)
    
    # Convert intervals to numpy array
    intervals = intervals.astype(np.float32).values
    
    intervals = split_into_intervals(intervals, 96, 96)
    

    last_48 = intervals[:, -48:, :]  # Last interval_split entries

    # Further split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(intervals, output_data, test_size=0.3, random_state=42)
    
    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    return train_loader, val_loader


# -------------------------------------------------------------------------------------
# Main Script: Executes Bayesian optimization with Ray Tune.
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders("MMCS0002")

    analysis = tune.run(
        tune.with_parameters(train_transformer_model, data_loader=train_loader, val_loader=val_loader),
        config=search_space,
        num_samples=50,  # Number of trials
        
        scheduler=ASHAScheduler(
            metric="validation_mape",  # Use MAPE as the primary metric
            mode="min",
            max_t=100,
            grace_period=10,
            reduction_factor=2
        ),
        
        search_alg=BayesOptSearch(
            metric="validation_mape",  # Optimize for MAPE
            mode="min",
            random_search_steps=10
        ),
        resources_per_trial={"cpu": 2, "gpu": 1}  # Adjust based on your hardware
    )


    print("Best hyperparameters found: ", analysis.best_config)
