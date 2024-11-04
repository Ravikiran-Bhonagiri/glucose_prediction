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
log_filename = f'/home/rxb2495/logs/model_processing_5_benchmarking_classification_{current_time}.log'
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
        
        # Load the combined data tuple from the .npy file for the current ID
        file_path = f'/home/rxb2495/data/{id_}_all_data.npy'
        loaded_data = np.load(file_path, allow_pickle=True)

        # Reconstruct DataFrames from the loaded data
        intervals = pd.DataFrame(data=loaded_data[0][1], columns=loaded_data[0][0])
        output_data = pd.DataFrame(data=loaded_data[1][1], columns=loaded_data[1][0])

        # Handle missing values
        if output_data.isnull().values.any():
            logging.warning(f"The output_data DataFrame for {id_} contains NaN values.")
        if intervals.isnull().values.any():
            logging.warning(f"The intervals DataFrame for {id_} contains NaN values.")

        intervals = intervals[features]
        
        # Prepare output data for classification
        output_data = output_data[["Historic Glucose mg/dL"]]
        
        # Reduce memory usage by converting to appropriate data types
        intervals = intervals.astype(np.float32)
        output_data = output_data.astype(np.float32)

        # Create categorical bins for binary classification
        bins = [0, 100, float('inf')]  # 0-100 is one class, >100 is the second class
        labels = [0, 1]  # Class 0: 0-100, Class 1: >100

        # Create the new 'Glucose_Category' based on the specified bins
        output_data['Glucose_Category'] = pd.cut(output_data['Historic Glucose mg/dL'], bins=bins, labels=labels, right=True)

        # Encode the categories using LabelEncoder (though not strictly necessary since labels are already 0 and 1)
        label_encoder = LabelEncoder()
        output_data['Glucose_Label'] = label_encoder.fit_transform(output_data['Glucose_Category'])

        glucose_label_counts = output_data['Glucose_Label'].value_counts()

        # Convert value counts to dictionary
        glucose_label_counts_dict = glucose_label_counts.to_dict()

        # Log the value counts as a dictionary
        logging.info("Glucose_Label value counts (as dictionary):")
        logging.info(f"{glucose_label_counts_dict}")

        # Perform classification
        output_data_scaled = output_data[["Glucose_Label"]].values.astype(np.float32)

        logging.info(f"Total size of data: {output_data_scaled.shape[0]}")

        # Split data into intervals
        interval_size = 96
        intervals = split_into_intervals(intervals, interval_size, interval_size)

        last_12 = intervals[:, -interval_split:, :]  # Last 24 entries

        intervals = last_12.astype(np.float32)
        output_data_scaled = output_data_scaled.astype(np.float32)

        feature_size = intervals.shape[2]
        num_classes = 2

        # Initialize stratified 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # List to store the models for evaluation
        model_names = ['LSTM', 'Transformer', 'XGBoost', 'DecisionTree', 'RandomForest', 'CNN_LSTM']

        # Modified model names with Validation and Test tags
        model_names_with_tags = [f"{name} (Test)" for name in model_names]

        # Dictionary to store metrics for each model
        model_results = {model_name: {
            'Accuracy': [], 
            'Precision': [], 
            'Recall': [], 
            'F1_Score': [], 
            'AUC_ROC': [], 
            'Confusion_Matrix': [], 
            'Classification_Report': []} for model_name in model_names_with_tags}

        ############################ LSTM Model ############################
        logging.info(f"Starting LSTM model training")

        for fold, (train_index, val_index) in enumerate(skf.split(intervals, output_data_scaled.flatten())):
            logging.info(f"Processing Fold {fold + 1}")

            # Split into train and test sets (90% train, 10% test)
            X_train_fold, X_test_fold = intervals[train_index], intervals[val_index]
            y_train_fold, y_test_fold = output_data_scaled[train_index], output_data_scaled[val_index]

            # Further split the training set into train and validation sets (90% train, 10% validation)
            X_train, X_val, y_train, y_val = train_test_split(X_train_fold, y_train_fold, test_size=0.1, stratify=y_train_fold, random_state=42)

            # Define LSTM model
            lstm_model = Sequential([
                LSTM(128, input_shape=(interval_split, feature_size), return_sequences=True),
                Dropout(0.2),
                LSTM(128, return_sequences=True),
                Dropout(0.2),
                LSTM(64),
                Dense(64, activation='relu'),
                Dense(num_classes, activation='softmax')  # Multi-class classification
            ])

            lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Use the custom EpochLogger callback and early stopping
            epoch_logger = EpochLogger()

            # Train the model on the train-validation split for the current fold
            lstm_model.fit(X_train, y_train, epochs=m_epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0, callbacks=[epoch_logger])

            # Evaluate the model on the validation set
            y_val_pred_probs = lstm_model.predict(X_val)
            y_val_pred_classes = np.argmax(y_val_pred_probs, axis=1)
            y_val_classes = y_val.flatten()


            # After training, test the model on the test set (held out earlier in the fold)
            y_test_pred_probs = lstm_model.predict(X_test_fold)
            y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)
            y_test_classes = y_test_fold.flatten()

            # Capture and log test metrics
            model_results = capture_metrics(y_test_classes, y_test_pred_classes, y_test_pred_probs, f'LSTM (Test)', model_results)

            # Clean up memory
            del lstm_model, X_train, X_val, y_train, y_val, y_val_pred_probs, y_val_pred_classes, y_test_pred_probs, y_test_pred_classes
            gc.collect()

            logging.info(f"LSTM model training and evaluation complete for all folds.")

        ############################ Transformer Model ############################
        logging.info(f"Starting Transformer model training")

        for train_index, test_index in skf.split(intervals, output_data_scaled.flatten()):
            X_train, X_test = intervals[train_index], intervals[test_index]
            y_train, y_test = output_data_scaled[train_index], output_data_scaled[test_index]

            # Assuming X_train and y_train are your training data and labels
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


            # Define Transformer model for each fold
            transformer_model = build_transformer_model(
                input_shape=(interval_split, feature_size),
                head_size=256,
                num_heads=4,
                ff_dim=128,
                num_transformer_blocks=4,
                mlp_units=[128],
                dropout=0.2,
                mlp_dropout=0.2
            )

            # Use the custom EpochLogger callback and early stopping
            epoch_logger = EpochLogger()

            transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            history = transformer_model.fit(X_train_split, y_train_split, 
                                    validation_data=(X_val_split, y_val_split), 
                                    epochs=m_epochs, batch_size=32, callbacks=[epoch_logger])


            # Evaluate the model
            y_pred_probs = transformer_model.predict(X_test)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            y_test_classes = y_test.flatten()

            model_results = capture_metrics(y_test_classes, y_pred_classes, y_pred_probs, f'Transformer (Test)', model_results)

        logging.info(f"Transformer model training and evaluation complete for all folds.")

        ############################ XGBoost Model ############################
        # Flatten the input data for compatibility with the models
        X_full_flattened = intervals.reshape(intervals.shape[0], -1)
        y_full = output_data_scaled.flatten()

        ############################ XGBoost Model ############################
        logging.info(f"Starting XGBoost model training with Stratified K-Fold Cross-Validation")
        for fold, (train_index, val_index) in enumerate(skf.split(X_full_flattened, y_full)):
            logging.info(f"Processing Fold {fold + 1}")

            # Split into train and validation sets for the current fold
            X_train, X_val = X_full_flattened[train_index], X_full_flattened[val_index]
            y_train, y_val = y_full[train_index], y_full[val_index]

            # Define and train the XGBoost model
            xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')

            xgb_model.fit(X_train, y_train)

            # Capture and log validation metrics
            y_val_pred = xgb_model.predict(X_val)
            y_val_pred_probs = xgb_model.predict_proba(X_val)
            model_results = capture_metrics(y_val, y_val_pred, y_val_pred_probs, f'XGBoost (Test)', model_results)

            # Clean up memory
            del xgb_model, X_train, X_val, y_train, y_val, y_val_pred, y_val_pred_probs
            gc.collect()

        logging.info(f"XGBoost model training and testing complete for all folds.")

        ############################ Decision Tree Model ############################
        logging.info(f"Starting Decision Tree model training with Stratified K-Fold Cross-Validation")
        for fold, (train_index, val_index) in enumerate(skf.split(X_full_flattened, y_full)):
            logging.info(f"Processing Fold {fold + 1}")

            # Split into train and validation sets for the current fold
            X_train, X_val = X_full_flattened[train_index], X_full_flattened[val_index]
            y_train, y_val = y_full[train_index], y_full[val_index]

            # Define and train the Decision Tree model
            dt_model = DecisionTreeClassifier(random_state=42)

            dt_model.fit(X_train, y_train)

            # Capture and log validation metrics
            y_val_pred = dt_model.predict(X_val)
            y_val_pred_probs = dt_model.predict_proba(X_val)
            model_results = capture_metrics(y_val, y_val_pred, y_val_pred_probs, f'DecisionTree (Test)', model_results)

            # Clean up memory
            del dt_model, X_train, X_val, y_train, y_val, y_val_pred, y_val_pred_probs
            gc.collect()

        logging.info(f"Decision Tree model training and testing complete for all folds.")

        ############################ Random Forest Model ############################
        logging.info(f"Starting Random Forest model training with Stratified K-Fold Cross-Validation")
        for fold, (train_index, val_index) in enumerate(skf.split(X_full_flattened, y_full)):
            logging.info(f"Processing Fold {fold + 1}")

            # Split into train and validation sets for the current fold
            X_train, X_val = X_full_flattened[train_index], X_full_flattened[val_index]
            y_train, y_val = y_full[train_index], y_full[val_index]

            # Define and train the Random Forest model
            rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

            rf_model.fit(X_train, y_train)

            # Capture and log validation metrics
            y_val_pred = rf_model.predict(X_val)
            y_val_pred_probs = rf_model.predict_proba(X_val)
            model_results = capture_metrics(y_val, y_val_pred, y_val_pred_probs, f'RandomForest (Test)', model_results)

            # Clean up memory
            del rf_model, X_train, X_val, y_train, y_val, y_val_pred, y_val_pred_probs
            gc.collect()

        logging.info(f"Random Forest model training and testing complete for all folds.")


        ############################ CNN-LSTM Model ############################
        logging.info(f"Starting CNN-LSTM model training")

        for fold, (train_index, val_index) in enumerate(skf.split(intervals, output_data_scaled.flatten())):
            logging.info(f"Processing Fold {fold + 1}")

            # Split into train and test sets (90% train, 10% test)
            X_train_fold, X_test_fold = intervals[train_index], intervals[val_index]
            y_train_fold, y_test_fold = output_data_scaled[train_index], output_data_scaled[val_index]

            # Further split the training set into train and validation sets (90% train, 10% validation)
            X_train, X_val, y_train, y_val = train_test_split(X_train_fold, y_train_fold, test_size=0.1, stratify=y_train_fold, random_state=42)

            # Define CNN-LSTM model
            cnn_lstm_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(interval_split, feature_size)),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(64),
                Dense(32, activation='relu'),
                Dense(num_classes, activation='softmax')  # Multi-class classification
            ])

            cnn_lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Use the custom EpochLogger callback and early stopping
            epoch_logger = EpochLogger()

            # Train the model on the train-validation split for the current fold
            cnn_lstm_model.fit(X_train, y_train, epochs=m_epochs, batch_size=64, validation_data=(X_val, y_val), verbose=0, callbacks=[epoch_logger])

            # Evaluate the model on the validation set
            y_val_pred_probs = cnn_lstm_model.predict(X_val)
            y_val_pred_classes = np.argmax(y_val_pred_probs, axis=1)
            y_val_classes = y_val.flatten()

            # After training, test the model on the test set (held out earlier in the fold)
            y_test_pred_probs = cnn_lstm_model.predict(X_test_fold)
            y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)
            y_test_classes = y_test_fold.flatten()

            # Capture and log test metrics
            model_results = capture_metrics(y_test_classes, y_test_pred_classes, y_test_pred_probs, f'CNN_LSTM (Test)', model_results)

            # Clean up memory
            del cnn_lstm_model, X_train, X_val, y_train, y_val, y_val_pred_probs, y_val_pred_classes, y_test_pred_probs, y_test_pred_classes
            gc.collect()

        logging.info(f"CNN-LSTM model training and evaluation complete for all folds.")


        # Logging the results at the end of each ID processing
        logging.info("------------------------------------------------------------------")
        logging.info(f"Processing completed for ID: {id_}")
        
        # Manually format and log model results with mean and std deviation
        for model_name, metrics in model_results.items():
            logging.info(f"Model: {model_name}")
            
            # Log each metric with comma-separated values and compute mean and std
            for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1_Score']:
                metric_values = metrics[metric_name]
                
                # Skip calculations for 'AUC_ROC' if it contains invalid data
                if metric_name == 'AUC_ROC' and (not metric_values or any(val is None for val in metric_values)):
                    logging.info(f"AUC_ROC: Not available due to null values")
                    continue
                
                # Compute mean and std only for valid data
                mean_value = np.mean(metric_values) if metric_values else 0
                std_value = np.std(metric_values) if metric_values else 0
                
                # Log the metric values, mean, and std deviation
                logging.info(f"{metric_name}: {', '.join(map(str, metric_values))}")
                logging.info(f"{metric_name} (Mean): {mean_value:.4f}")
                logging.info(f"{metric_name} (Std): {std_value:.4f}")
            
            # Confusion Matrix and Classification Report can be tricky to format, but here's one way to handle them:
            for i, conf_matrix in enumerate(metrics['Confusion_Matrix']):
                logging.info(f"Confusion Matrix {i+1}: {conf_matrix}")
            for i, class_report in enumerate(metrics['Classification_Report']):
                logging.info(f"Classification Report {i+1}: {class_report}")
        
        logging.info("------------------------------------------------------------------")
        
        # Store results for the current ID
        results[f"id_{index}_{interval_split}"] = model_results

        logging.info("---------------------------completed------------------------------")



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