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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score, mean_absolute_percentage_error
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


from regressor_models import LSTMModel, TransformerModel, CNNLSTMModel, CNNModel
from config import regression_config, \
                    heart_rate_features_1, heart_rate_features_2, heart_rate_features_3, heart_rate_features_4, heart_rate_features_5, \
                    sleep_features_1, sleep_features_2, sleep_features_3, sleep_features_4, sleep_features_5,  \
                    intensity_features_1, intensity_features_2, intensity_features_3, intensity_features_4, intensity_features_5, \
                    steps_features_1, steps_features_2, steps_features_3, steps_features_4, steps_features_5,  temporal_features, m_epochs, ids

# Ignore warnings
warnings.filterwarnings("ignore")

# Set up logging with timestamp
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_filename = f'/home/rxb2495/logs/model_processing_forecasting_experiment_plots_{current_time}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Set device for GPU use
device = 'cuda' if torch.cuda.is_available() else 'cpu'

features_1 =  heart_rate_features_1 + sleep_features_1 + intensity_features_1 + steps_features_1 + temporal_features

features_2 =  features_1 + heart_rate_features_2 + sleep_features_2 + intensity_features_2 + steps_features_2

features_3 =  features_2 + heart_rate_features_3 + sleep_features_3 + intensity_features_3 + steps_features_3

features_4 =  features_3 + heart_rate_features_4 + sleep_features_4 + intensity_features_4 + steps_features_4

features_5 =  features_4 + heart_rate_features_5 + sleep_features_5 + intensity_features_5 + steps_features_5

# List to store the models for evaluation
model_names = ['LSTM', 'Transformer', 'CNN-LSTM', 'CNN', 'XGBoost', 'RandomForest']

# Modified model names with Validation and Test tags
model_names_with_tags = [f"{name} (Test)" for name in model_names]


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


# Create a PDF file to save all the plots
pdf_filename = "/home/rxb2495/glucose_predictions_plots.pdf"
pdf_pages = PdfPages(pdf_filename)

# CSV file to save actual vs predicted values
csv_filename = "/home/rxb2495/glucose_predictions_values.csv"
csv_data = []

# Function to capture and log metrics for regression
def capture_metrics(y_test, y_pred, model_name, model_results):
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
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
def pipeline_run(intervals, output_data, m_epochs, model_results, regression_config):
    """
    Main pipeline function for training deep learning models using K-Fold cross-validation.
    This includes training, validation after every epoch, and a final test evaluation after all epochs.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    logging.info("Initialized K-Fold cross-validation")

    # ----------------------------
    # Classical Regression Models
    # ----------------------------

    X_full_flattened = intervals.reshape(intervals.shape[0], -1)
    y_full = output_data.flatten()
  
    classical_models = {
        'XGBoost': XGBRegressor(**regression_config['XGBoost']),
        'RandomForest': RandomForestRegressor(**regression_config['RandomForest'])
    }

    for model_name, model in classical_models.items():
        logging.info(f"Starting training for classical model: {model_name}")

        for fold, (train_index, test_index) in enumerate(kf.split(intervals)):
            logging.info(f"Processing Fold {fold + 1} for model: {model_name}")

            X_train, X_test = X_full_flattened[train_index], X_full_flattened[test_index]
            y_train, y_test = y_full[train_index], y_full[test_index]

            # Train classical model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate test metrics
            # Calculate test metrics
            test_mae = mean_absolute_error(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            test_rmse = np.sqrt(test_mse)
            test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            test_r2 = r2_score(y_test, y_pred)

            # Log the metrics
            logging.info(f"{model_name} - MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}, R2: {test_r2:.4f}")
            print(f"Model-Name: {model_name} - MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}, R2: {test_r2:.4f}")


            # Capture metrics for the current classical model
            model_results = capture_metrics(
                np.array(y_test), np.array(y_pred), f"{model_name} (Test)", model_results
            )

            # Save actual and predicted values to the CSV data list
            for actual, predicted in zip(y_test, y_pred):
                csv_data.append({
                    "Model": model_name,
                    "Fold": fold + 1,
                    "Actual": actual,
                    "Predicted": predicted
                })

            # Plot Actual vs Predicted values
            plt.figure(figsize=(10, 6))
            # Plotting the actual glucose levels with a solid blue line
            plt.plot(y_test, label='Actual', linestyle='-', color='blue')

            # Plotting the predicted glucose levels with a solid red line
            plt.plot(y_pred, label='Predicted', linestyle='-', color='red')

            # Set plot title with model name and fold number
            plt.title(f"Actual vs Predicted Glucose Levels\nModel: {model_name}, Fold: {fold + 1}", fontsize=14)
            plt.xlabel("Input Interval")
            plt.ylabel("Glucose Level")
            plt.legend()
            plt.grid(True)

            # Add text box with metrics in the plot
            plt.figtext(0.15, -0.1, f"MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}",
                        fontsize=10, ha='left')

            # Save the current plot to the PDF
            pdf_pages.savefig()
            plt.close()

    return model_results


# Function to load data from .npy files
def load_dataframe_from_npy(file_path):
    data_dict = np.load(file_path, allow_pickle=True).item()
    columns = data_dict['columns']
    data = data_dict['data']
    return pd.DataFrame(data, columns=columns)


def set_input_size(config, input_size):
    """ Set the input size in the configuration based on input data shape """
    for model in config:
        if 'input_size' in config[model]:
            config[model]['input_size'] = input_size
    return config


def initialize_model_metrics(model_names_with_tags):
    model_results = {model_name: {
        'MAE': [],          # Mean Absolute Error
        'MSE': [],          # Mean Squared Error
        'RMSE': [],         # Root Mean Squared Error
        'MAPE': [],         # Mean Absolute Percentage Error
        'R2_Score': [],     # Coefficient of Determination (R²)
        'Median_AE': [],    # Median Absolute Error
        'Explained_Variance': []  # Explained Variance Score
    } for model_name in model_names_with_tags}
    
    return model_results
# Main pipeline execution
logging.info("Running forecasting pipeline")


list_of_features = [features_2]

results = {}

logging.info("Running all experiments for all interval splits and couple of IDs")
# Loop through each ID and perform the data processing and model evaluation
for index, features in enumerate(list_of_features):
    logging.info(f"Processing features: {features} \n")

    regression_config = set_input_size(regression_config, len(features))
    model_results = initialize_model_metrics(model_names_with_tags)
    
    print(f"features length {len(features)}")

    for interval_split in [96]:

        print(f"interval split {interval_split}")
        # Load data for the current features and interval split

        for id_ in ["MMCS0007"]:
            logging.info(f"Processing ID: {id_}")

            # Load features and labels
            features_data_path = f'/home/rxb2495/Glucose-Forecasting/final_data/{id_}_combined_data.npy'
            labels_data_path = f'/home/rxb2495/Glucose-Forecasting/final_data/{id_}_main_data.npy'
            
            intervals = load_dataframe_from_npy(features_data_path)
            output_data = load_dataframe_from_npy(labels_data_path)

            print(output_data.columns)

            # Handle missing values
            if output_data.isnull().values.any():
                logging.warning(f"The output_data DataFrame for {id_} contains NaN values.")
            if intervals.isnull().values.any():
                logging.warning(f"The intervals DataFrame for {id_} contains NaN values.")

            
            intervals = intervals[features]

            intervals = intervals.loc[:, ~intervals.columns.duplicated()]

            output_data = output_data["Historic Glucose mg/dL"].values.astype(np.float32)
            
            # Convert intervals to numpy array
            intervals = intervals.astype(np.float32).values
            interval_size = 96
            intervals = split_into_intervals(intervals, interval_size, interval_size)
            
            last_48 = intervals[:, -interval_split:, :]  # Last interval_split entries

            intervals = last_48.astype(np.float32)

            # Run the pipeline for forecasting
            model_results = pipeline_run(intervals, output_data, m_epochs, model_results, regression_config)

            results[f"id_{id_}_{index}_{interval_split}"] = model_results

# Save all the plots to the PDF file
pdf_pages.close()

# Save the actual vs predicted data to a CSV file
df = pd.DataFrame(csv_data)
df.to_csv(csv_filename, index=False)

# Save results to CSV
csv_data = []
for model_name, metrics in model_results.items():
    csv_data.append({
        "Model": model_name,
        "Mean MAE": np.mean(metrics['MAE']),
        "Mean MSE": np.mean(metrics['MSE']),
        "Mean RMSE": np.mean(metrics['RMSE']),
        "Mean MAPE": np.mean(metrics['MAPE']),
        "Mean R2 Score": np.mean(metrics['R2_Score']),
        "Mean Median AE": np.mean(metrics['Median_AE']),
        "Mean Explained Variance": np.mean(metrics['Explained_Variance']),
        "MAE List": metrics['MAE'],
        "MSE List": metrics['MSE'],
        "RMSE List": metrics['RMSE'],
        "MAPE List": metrics['MAPE'],
        "R2_Score List": metrics['R2_Score'],
        "Median_AE List": metrics['Median_AE'],
        "Explained_Variance List": metrics['Explained_Variance']
    })


df = pd.DataFrame(csv_data)
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
df.to_csv(f"/home/rxb2495/forecasting_results_baseline_{m_epochs}_epochs_{current_time}.csv", index=False)

logging.info("Forecasting pipeline completed successfully!")
