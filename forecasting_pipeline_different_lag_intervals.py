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
model_names = ['XGBoost']

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
pdf_filename = "/home/rxb2495/glucose_predictions_all_subjects_lag_plots_{current_time}.pdf"
pdf_pages = PdfPages(pdf_filename)

# CSV file to save actual vs predicted values
csv_filename = "/home/rxb2495/glucose_predictions_lag_values_{current_time}.csv"
csv_data = []


# Pipeline function for training models
def pipeline_run(intervals, output_data, m_epochs, regression_config, id_, index, lag_val):
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
    }
    print(f"Lag: {lag_val}")
    logging.info(f"Lag: {lag_val}")
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


            # Save actual and predicted values to the CSV data list
            for actual, predicted in zip(y_test, y_pred):
                csv_data.append({
                    "Model": model_name,
                    "Fold": fold + 1,
                    "Actual": actual,
                    "Predicted": predicted,
                    "id": id_,
                    "feature_set": index, 
                    "lag": lag_val*15
                })

            # Create a new figure with a specified size
            plt.figure(figsize=(10, 6))
            
            # Plot the actual glucose levels with a solid blue line
            plt.plot(y_test, label='Actual', linestyle='-', color='blue')
        
            # Plot the predicted glucose levels with a solid red line
            plt.plot(y_pred, label='Predicted', linestyle='-', color='red')
        
            # Set plot title with model name and fold number
            plt.title(f"Actual vs Predicted Glucose Levels\n ID: {id_}, Model: {model_name}, Fold: {fold + 1}", fontsize=14)
            plt.xlabel("Input Interval")
            plt.ylabel("Glucose Level")
            plt.legend()
            plt.grid(True)
            
            # Adjust layout to fit the metrics text within the figure
            plt.tight_layout()
        
            # Add a text box with metrics inside the plot area to avoid getting cut off
            metrics_text = f"MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}, LAG: {lag_val*15}"
            plt.figtext(0.5, -0.1, metrics_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
            # Save the current plot to the PDF
            pdf_pages.savefig()
            plt.close()

    return ""


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


# Main pipeline execution
logging.info("Running forecasting pipeline")


list_of_features = [features_2]

results = {}

logging.info("Running all experiments for all interval splits and couple of IDs")
# Loop through each ID and perform the data processing and model evaluation
for index, features in enumerate(list_of_features):
    logging.info(f"Processing features: {features} \n")

    regression_config = set_input_size(regression_config, len(features))
    
    print(f"features length {len(features)}")

    for lag_val in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        
        interval_split = 48
        interval_size = 96
        
        print(f"interval split {interval_split}")
        # Load data for the current features and interval split

        for id_ in ids:
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

            logging.info(f"Final Features : {intervals.columns}")
            logging.info(f"Final Features Length : {len(intervals.columns)}")

            output_data = output_data["Historic Glucose mg/dL"].values.astype(np.float32)
            
            # Convert intervals to numpy array
            intervals = intervals.astype(np.float32).values
            
            intervals = split_into_intervals(intervals, interval_size, interval_size)
            
            if lag_val:
                last_48 = intervals[:, -(interval_split+lag_val):-lag_val, :]  # Last interval_split entries
            else:
                last_48 = intervals[:, -interval_split:, :]  # Last interval_split entries

            intervals = last_48.astype(np.float32)

            print(f"intervals shape is {intervals.shape}")
            logging.info(f"intervals shape is {intervals.shape}")

            # Run the pipeline for forecasting
            pipeline_run(intervals, output_data, m_epochs, regression_config, id_, index, lag_val)


# Save all the plots to the PDF file
pdf_pages.close()

# Save the actual vs predicted data to a CSV file
df = pd.DataFrame(csv_data)
df.to_csv(csv_filename, index=False)

logging.info("Forecasting pipeline completed successfully!")
