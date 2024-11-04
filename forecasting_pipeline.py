import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import gc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, Callback
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import warnings

import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC



# Ignore all warnings
warnings.filterwarnings("ignore")

# Set global seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

# Define the list of IDs
# ids = ["MMCS0002", "MMCS0003", "MMCS0005", "MMCS0007", "MMCS0008", "MMCS0009", "MMCS0010", "MMCS0011", "MMCS0016"]

ids = ["MMCS0002"]

# Define a dictionary to store results for each model and each ID
#time_values = [12, 24, 48, 96]

time_values = [48]

# Initialize the results dictionary
results = {f"{id_}_{t}": {} for id_ in ids for t in time_values}

m_epochs = 1

#========================================================================================================================================================

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

# Function to capture and log metrics
def capture_metrics(y_test, y_pred, y_pred_probs, model_name, model_results):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Confusion Matrix and Classification Report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # AUC-ROC Calculation
    try:
        auc_roc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc_roc = 'N/A'

    # Append metrics to the results dictionary
    model_results[model_name]['Accuracy'].append(accuracy)
    model_results[model_name]['Precision'].append(precision)
    model_results[model_name]['Recall'].append(recall)
    model_results[model_name]['F1_Score'].append(f1)
    model_results[model_name]['AUC_ROC'].append(auc_roc)
    model_results[model_name]['Confusion_Matrix'].append(conf_matrix)
    model_results[model_name]['Classification_Report'].append(class_report)

    # Log the metrics
    logging.info(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-ROC: {auc_roc}")
    return model_results


# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

# Transformer Model
def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(2, activation='softmax')(x)
    return Model(inputs, outputs)



current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Set up logging with timestamp in filename
log_filename = f'/home/rxb2495/logs/model_processing_5_benchmarking_forecasting_{current_time}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Custom callback for logging per epoch
class EpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Epoch {epoch + 1} - loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}, "
                     f"val_loss: {logs['val_loss']:.4f}, val_accuracy: {logs['val_accuracy']:.4f}")

#========================================================================================================================================================

# Heart Rate Features
heart_rate_features_1 = [
    'HeartRate' 
]

heart_rate_features_2 = [
    'HeartRate_15_Mean', 'HeartRate_15_Std', 'HeartRate_30_Mean', 'HeartRate_30_Std','HeartRate_60_Mean', 'HeartRate_60_Std', 'HeartRate_90_Mean', 
    'HeartRate_90_Std', 'HeartRate_180_Mean', 'HeartRate_180_Std', 'HeartRate_360_Mean', 'HeartRate_360_Std', 'HeartRate_720_Mean', 'HeartRate_720_Std', 
    'HeartRate_1440_Mean', 'HeartRate_1440_Std', 'HeartRate_Diff_Lag_15_Mean_15', 'HeartRate_Diff_Lag_15_Std_15'
]

heart_rate_features_3 = [
    'HeartRate_Diff_Lag_30_Mean_30', 'HeartRate_Diff_Lag_30_Std_30', 'HeartRate_Diff_Lag_60_Mean_60',
    'HeartRate_Diff_Lag_60_Std_60', 'HeartRate_Diff_Lag_90_Mean_90', 'HeartRate_Diff_Lag_90_Std_90',
    'HeartRate_EWMA_15_Mean', 'HeartRate_EWMA_15_Std', 'HeartRate_EWMA_30_Mean', 'HeartRate_EWMA_30_Std',
    'HeartRate_EWMA_60_Mean', 'HeartRate_EWMA_60_Std', 'HeartRate_EWMA_90_Mean', 'HeartRate_EWMA_90_Std',
    'HeartRate_EWMA_180_Mean', 'HeartRate_EWMA_180_Std']

heart_rate_features_4 = [
    'HeartRate_RoC_1', 'HeartRate_RoC_5', 'HeartRate_RoC_15',
    'HeartRate_RoC_30', 'HeartRate_RoC_60', 'HeartRate_RoC_120', 'HeartRate_RoC_180', 'HeartRate_Autocorr_lag_5_15',
    'HeartRate_Autocorr_lag_5_30', 'HeartRate_Autocorr_lag_5_60', 'HeartRate_Autocorr_lag_5_120',
    'HeartRate_Autocorr_lag_30_180', 'HeartRate_Autocorr_lag_30_240', 'HeartRate_PSD_15', 'HeartRate_PSD_30',
    'HeartRate_PSD_60', 'HeartRate_PSD_90']

# Sleep Features
sleep_features_1 = [
    'Sleep'
]

sleep_features_2 = [
    'Sleep_15min_Mean', 'Sleep_15min_Std', 'Sleep_30min_Mean', 'Sleep_30min_Std', 'Sleep_60min_Mean', 'Sleep_60min_Std', 
    'Sleep_90min_Mean', 'Sleep_90min_Std', 'Sleep_180min_Mean', 'Sleep_180min_Std', 'Sleep_240min_Mean', 'Sleep_240min_Std', 
    'Sleep_360min_Mean', 'Sleep_360min_Std', 'Sleep_720min_Mean', 'Sleep_720min_Std', 'Sleep_1440min_Mean', 'Sleep_1440min_Std'
]

sleep_features_3 = [
    'Sleep_15min_Skew', 'Sleep_15min_Kurt', 'Sleep_30min_Skew', 'Sleep_30min_Kurt', 'Sleep_60min_Skew',
    'Sleep_60min_Kurt', 'Sleep_90min_Skew', 'Sleep_90min_Kurt', 'Sleep_15min_Sum', 'Sleep_30min_Sum',
    'Sleep_60min_Sum', 'Sleep_90min_Sum', 'Sleep_180min_Sum', 'Sleep_240min_Sum'
]

sleep_features_4 = [
    'Sleep_RoC_1min', 'Sleep_RoC_5min', 'Sleep_RoC_15min', 'Sleep_RoC_30min', 'Sleep_RoC_60min', 'Sleep_RoC_120min',
    'Sleep_RoC_180min', 'Sleep_PSD_15min', 'Sleep_PSD_30min', 'Sleep_PSD_60min', 'Sleep_PSD_90min'
]

# Intensity Features
intensity_features_1 = [
    'Intensity', 
]

intensity_features_2 = [
    'Intensity_15min_Mean', 'Intensity_15min_Std', 'Intensity_30min_Mean', 'Intensity_30min_Std', 
    'Intensity_60min_Mean', 'Intensity_60min_Std', 'Intensity_90min_Mean', 'Intensity_90min_Std', 'Intensity_180min_Mean',
    'Intensity_180min_Std', 'Intensity_240min_Mean', 'Intensity_240min_Std', 'Intensity_360min_Mean', 'Intensity_360min_Std', 
    'Intensity_720min_Mean', 'Intensity_720min_Std', 'Intensity_1440min_Mean', 'Intensity_1440min_Std'
]

intensity_features_3 = [
    'Intensity_15min_Sum', 'Intensity_30min_Sum', 'Intensity_60min_Sum', 'Intensity_90min_Sum', 'Intensity_180min_Sum',
    'Intensity_240min_Sum'
]

intensity_features_4 = [
    'Intensity_RoC_1min', 'Intensity_RoC_5min', 'Intensity_RoC_15min', 'Intensity_RoC_30min',
    'Intensity_RoC_60min', 'Intensity_RoC_120min', 'Intensity_RoC_180min'
    'Intensity_Switch_Count_15min', 'Intensity_Switch_Count_30min', 'Intensity_Switch_Count_60min', 'Intensity_Switch_Count_90min',
    'Intensity_Switch_Count_180min', 'Intensity_PSD_15min', 'Intensity_PSD_30min', 'Intensity_PSD_60min', 'Intensity_PSD_90min'
]

# Steps Features
steps_features_1 = [
    'Steps', 
]

steps_features_2 = [
    'Steps_Lag_1min', 'Steps_Lag_5min', 'Steps_Lag_15min', 'Steps_Lag_30min', 'Steps_Lag_60min', 
    'Steps_Lag_120min', 'Steps_Lag_180min', 'Steps_Lag_240min', 'Steps_Lag_360min', 'Steps_Lag_720min', 'Steps_Lag_1440min',
    'Steps_15min_Mean', 'Steps_15min_Std', 'Steps_30min_Mean', 'Steps_30min_Std', 'Steps_60min_Mean',
    'Steps_60min_Std', 'Steps_90min_Mean', 'Steps_90min_Std', 'Steps_180min_Mean', 'Steps_180min_Std'
]

steps_features_3 = [
    'Steps_Lag_Diff_5min', 'Steps_Lag_Diff_15min', 'Steps_Lag_Diff_30min',
    'Steps_Lag_Diff_60min', 'Steps_Lag_Diff_120min', 'Steps_Lag_Diff_180min', 'Steps_Lag_Diff_240min', 
    'Steps_Lag_Diff_360min'
]

steps_features_4 = [
    'Steps_RoC_5min', 'Steps_RoC_15min', 'Steps_RoC_30min', 'Steps_RoC_60min', 'Steps_RoC_120min',
    'Steps_RoC_180min', 'Steps_Lag_Diff_5min_15min_Mean', 'Steps_Lag_Diff_5min_15min_Std',
    'Steps_Lag_Diff_5min_30min_Mean', 'Steps_Lag_Diff_5min_30min_Std', 'Steps_Lag_Diff_5min_60min_Mean',
    'Steps_Lag_Diff_5min_60min_Std'
]


temporal_features = [
    'minute',
     'hour',
     'day_of_week',
     'day_of_month',
     'SecondOfMinute_Sin',
     'SecondOfMinute_Cos',
     'MinuteOfHour_Sin',
     'MinuteOfHour_Cos',
     'HourOfDay_Sin',
     'HourOfDay_Cos',
     'DayOfWeek_Sin',
     'DayOfWeek_Cos',
     'Is_Weekend'
]


features_1 =  heart_rate_features_1 + sleep_features_1 + intensity_features_1 + steps_features_1 + temporal_features

features_2 =  features_1 + heart_rate_features_2 + sleep_features_2 + intensity_features_2 + steps_features_2

features_3 =  features_2 + heart_rate_features_3 + sleep_features_3 + intensity_features_3 + steps_features_3

features_4 =  features_3 + heart_rate_features_4 + sleep_features_4 + intensity_features_4 + steps_features_4


list_of_features = [ features_1, features_2, features_3, features_4]

interval_split = 48

logging.info("Started experiment for 12 hrs interval")

# Loop through each ID and perform the data processing and model evaluation
for index, features in enumerate(list_of_features):
    logging.info(f"Processing features: {features} \n")
    for id_ in ids:
        logging.info(f"Processing ID: {id_}")



# Save results to CSV
csv_data = []
for id_, model_results in results.items():
    for model_name, metrics in model_results.items():
        # Convert confusion matrix and classification report to strings (for storage in CSV)
        confusion_matrix_str = str(metrics['Confusion_Matrix'])  # Store confusion matrix as string
        classification_report_str = metrics['Classification_Report'][0] if isinstance(metrics['Classification_Report'], list) else str(metrics['Classification_Report'])
        
        csv_data.append({
            "ID": id_,
            "Model": model_name,
            "Mean Accuracy": np.mean(metrics['Accuracy']),
            "Mean Precision": np.mean(metrics['Precision']),
            "Mean Recall": np.mean(metrics['Recall']),
            "Mean F1_Score": np.mean(metrics['F1_Score']),
            #"Mean AUC-ROC": np.mean(metrics['AUC_ROC']) if metrics['AUC_ROC'][0] != 'N/A' else 'N/A',
            "AUC-ROC List": metrics['AUC_ROC'] if metrics['AUC_ROC'][0] != 'N/A' else 'N/A',
            "Accuracy List": metrics['Accuracy'],
            "Precision List": metrics['Precision'],
            "Recall List": metrics['Recall'],
            "F1_Score List": metrics['F1_Score'],
            "Confusion Matrix": confusion_matrix_str,  # Adding confusion matrix as a string
            "Classification Report": classification_report_str  # Adding classification report as a string
        })

# Convert to DataFrame and save as CSV
df = pd.DataFrame(csv_data)
df.to_csv(f"/home/rxb2495/binary_classification_benchmarking_results_normal_500_epochs_{current_time}.csv", index=False)