import logging
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score
import numpy as np

from regressor_models import LSTMModel, TransformerModel, CNNLSTMModel, CNNModel
from config import classification_config, \
                    heart_rate_features_1, heart_rate_features_2, heart_rate_features_3, heart_rate_features_4, heart_rate_features_5, \
                    sleep_features_1, sleep_features_2, sleep_features_3, sleep_features_4, sleep_features_5,  \
                    intensity_features_1, intensity_features_2, intensity_features_3, intensity_features_4, intensity_features_5, \
                    steps_features_1, steps_features_2, steps_features_3, steps_features_4, steps_features_5,  temporal_features, m_epochs, ids


# Ignore warnings
warnings.filterwarnings("ignore")

# Set up logging with timestamp
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_filename = f'/home/rxb2495/logs/model_processing_forecasting_{current_time}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Set device for GPU use
device = 'cuda' if torch.cuda.is_available() else 'cpu'

features_1 =  heart_rate_features_1 + sleep_features_1 + intensity_features_1 + steps_features_1 + temporal_features

features_2 =  features_1 + heart_rate_features_2 + sleep_features_2 + intensity_features_2 + steps_features_2

features_3 =  features_2 + heart_rate_features_3 + sleep_features_3 + intensity_features_3 + steps_features_3

features_4 =  features_3 + heart_rate_features_4 + sleep_features_4 + intensity_features_4 + steps_features_4

features_5 =  features_4 + heart_rate_features_5 + sleep_features_5 + intensity_features_5 + steps_features_5


# Function to split data into intervals
def split_into_intervals(data, interval_size, stride):
    logging.info("Splitting data into intervals.")
    intervals = []
    num_intervals = (data.shape[0] - interval_size) // stride + 1
    for i in range(num_intervals):
        start_ix = i * stride
        end_ix = start_ix + interval_size
        interval = data[start_ix:end_ix]
        intervals.append(interval)
    return np.array(intervals)



# Function to capture and log metrics for regression
def capture_metrics(y_test, y_pred, model_name, model_results):
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if np.all(y_test) else float('inf')
    
    # R² Score (Coefficient of Determination)
    r2 = r2_score(y_test, y_pred)
    
    # Median Absolute Error
    median_ae = median_absolute_error(y_test, y_pred)
    
    # Explained Variance Score
    explained_var = explained_variance_score(y_test, y_pred)
    
    # Append metrics to the results dictionary
    model_results[model_name]['MAE'].append(mae)
    model_results[model_name]['MSE'].append(mse)
    model_results[model_name]['RMSE'].append(rmse)
    model_results[model_name]['MAPE'].append(mape)
    model_results[model_name]['R2_Score'].append(r2)
    model_results[model_name]['Median_AE'].append(median_ae)
    model_results[model_name]['Explained_Variance'].append(explained_var)
    
    # Log the metrics
    logging.info(
        f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, "
        f"MAPE: {mape:.4f}, R2: {r2:.4f}, Median AE: {median_ae:.4f}, Explained Variance: {explained_var:.4f}"
    )
    
    return model_results


# Pipeline function for training models
def pipeline_run(intervals, output_data, m_epochs, model_results, classification_config):
    """
    Main pipeline function for training deep learning models using K-Fold cross-validation.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    logging.info("Initialized K-Fold cross-validation")

    # Define deep learning models with configurations
    deep_models = {
        'LSTM': (LSTMModel, classification_config['LSTMModel']),
        'Transformer': (TransformerModel, classification_config['TransformerModel']),
        'CNN-LSTM': (CNNLSTMModel, classification_config['CNNLSTMModel']),
        'CNN': (CNNModel, classification_config['CNNModel'])
    }

    # Loop through each model
    for model_name, (model_class, model_params) in deep_models.items():
        logging.info(f"Starting training for model: {model_name}")

        # Perform K-Fold cross-validation
        for fold, (train_index, test_index) in enumerate(kf.split(intervals)):
            logging.info(f"Processing Fold {fold + 1} for model: {model_name}")
            
            # Split data into training and test sets
            X_train, X_test = intervals[train_index], intervals[test_index]
            y_train, y_test = output_data[train_index], output_data[test_index]

            # Further split training data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
            
            # Convert data to PyTorch tensors
            train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
            val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
            test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)

            # Initialize model, loss function, and optimizer
            model = model_class(**model_params).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters())

            # Training loop
            for epoch in range(m_epochs):
                model.train()
                running_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * X_batch.size(0)
                
                train_loss = running_loss / len(train_loader.dataset)
                logging.info(f"Epoch [{epoch + 1}/{m_epochs}] - Train Loss: {train_loss:.4f}")

            # Test evaluation
            model.eval()
            test_pred, test_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch).squeeze()
                    test_pred.extend(outputs.cpu().numpy())
                    test_true.extend(y_batch.cpu().numpy())

            # Capture metrics
            model_results = capture_metrics(np.array(test_true), np.array(test_pred), model_name, model_results)

    return model_results

# Function to load data from .npy files
def load_dataframe_from_npy(file_path):
    data_dict = np.load(file_path, allow_pickle=True).item()
    columns = data_dict['columns']
    data = data_dict['data']
    return pd.DataFrame(data, columns=columns)

# Initialize model metrics dictionary
def initialize_model_metrics(model_names):
    return {model_name: {'MAE': [], 'MSE': [], 'MAPE': []} for model_name in model_names}

# Main pipeline execution
logging.info("Running forecasting pipeline")
model_names = ['LSTM', 'Transformer', 'CNN-LSTM', 'CNN']
model_results = initialize_model_metrics(model_names)

for id_ in ids:
    logging.info(f"Processing ID: {id_}")

    # Load features and labels
    features_data_path = f'/home/rxb2495/Glucose-Forecasting/final_data/{id_}_combined_data.npy'
    labels_data_path = f'/home/rxb2495/Glucose-Forecasting/final_data/{id_}_main_data.npy'
    
    intervals = load_dataframe_from_npy(features_data_path)[features]
    output_data = load_dataframe_from_npy(labels_data_path)["Historic Glucose mg/dL"].values.astype(np.float32)
    
    # Convert intervals to numpy array
    intervals = intervals.astype(np.float32).values
    interval_size = 96
    intervals = split_into_intervals(intervals, interval_size, interval_size)
    
    # Run the pipeline for forecasting
    model_results = pipeline_run(intervals, output_data, m_epochs, model_results, classification_config)

# Save results to CSV
csv_data = []
for model_name, metrics in model_results.items():
    csv_data.append({
        "Model": model_name,
        "Mean MAE": np.mean(metrics['MAE']),
        "Mean MSE": np.mean(metrics['MSE']),
        "Mean MAPE": np.mean(metrics['MAPE']),
        "MAE List": metrics['MAE'],
        "MSE List": metrics['MSE'],
        "MAPE List": metrics['MAPE']
    })

df = pd.DataFrame(csv_data)
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
df.to_csv(f"/home/rxb2495/forecasting_results_{m_epochs}_epochs_{current_time}.csv", index=False)

logging.info("Forecasting pipeline completed successfully!")
