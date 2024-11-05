# pipeline.py

import logging
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from classifier_models import LSTMModel, DeepLSTMModel, TransformerModel, CNNLSTMModel, CNNModel, DeepCNNModel, DeepCNNLSTMModel
from config import classification_config, \
                    heart_rate_features_1, heart_rate_features_2, heart_rate_features_3, heart_rate_features_4, \
                    sleep_features_1, sleep_features_2, sleep_features_3, sleep_features_4, \
                    intensity_features_1, intensity_features_2, intensity_features_3, intensity_features_4, \
                    steps_features_1, steps_features_2, steps_features_3, steps_features_4, temporal_features, m_epochs, ids

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler


                    
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Set up logging with timestamp in filename
log_filename = f'/home/rxb2495/logs/model_processing_5_benchmarking_classification_{current_time}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# Set device for GPU use
device = 'cuda' if torch.cuda.is_available() else 'cpu'

features_1 =  heart_rate_features_1 + sleep_features_1 + intensity_features_1 + steps_features_1 + temporal_features

features_2 =  features_1 + heart_rate_features_2 + sleep_features_2 + intensity_features_2 + steps_features_2

features_3 =  features_2 + heart_rate_features_3 + sleep_features_3 + intensity_features_3 + steps_features_3

features_4 =  features_3 + heart_rate_features_4 + sleep_features_4 + intensity_features_4 + steps_features_4


list_of_features = [ features_1, features_2, features_3, features_4]

# Code to split into intervals
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

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

def evaluate_model(model, data_loader, criterion):
    model.eval()
    predictions, actuals, probs = [], [], []
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds = torch.round(torch.sigmoid(outputs)).cpu().numpy()
            predictions.extend(preds)
            actuals.extend(y_batch.cpu().numpy())
            probs.extend(torch.sigmoid(outputs).cpu().numpy())
    epoch_loss = running_loss / len(data_loader.dataset)
    logging.info(f"Evaluation Loss: {epoch_loss:.4f}")
    return predictions, actuals, probs

def process_deep_model(X_train, X_val, y_train, y_val, model_class, model_params, batch_size, epochs):
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = model_class(**model_params).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    train_model(model, train_loader, criterion, optimizer, epochs)
    y_val_pred, y_val_true, y_val_probs = evaluate_model(model, val_loader, criterion)
    return y_val_pred, y_val_true, y_val_probs

def process_classical_model(X_train, X_val, y_train, y_val, model):
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_val_probs = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
    return y_val_pred, y_val_probs

def pipeline_run(intervals, output_data_scaled, interval_split, feature_size, m_epochs, model_results):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    deep_models = {
        'LSTM': (LSTMModel, classification_config['LSTMModel']),
        'DeepLSTM': (DeepLSTMModel, classification_config['DeepLSTMModel']),
        'Transformer': (TransformerModel, classification_config['TransformerModel']),
        'CNN-LSTM': (CNNLSTMModel, classification_config['CNNLSTMModel']),
        'CNN': (CNNModel, classification_config['CNNModel']),
        'DeepCNN': (DeepCNNModel, classification_config['DeepCNNModel']),
        'DeepCNN-LSTM': (DeepCNNLSTMModel, classification_config['DeepCNNLSTMModel'])
    }

    classical_models = {
        'XGBoost': XGBClassifier(**classification_config['XGBoost']),
        'DecisionTree': DecisionTreeClassifier(**classification_config['DecisionTree']),
        'RandomForest': RandomForestClassifier(**classification_config['RandomForest'])
    }

    for model_name, (model_class, model_params) in deep_models.items():
        logging.info(f"Starting {model_name} model training")
        for fold, (train_index, val_index) in enumerate(skf.split(intervals, output_data_scaled.flatten())):
            logging.info(f"Processing Fold {fold + 1}")
            X_train, X_val = intervals[train_index], intervals[val_index]
            y_train, y_val = output_data_scaled[train_index], output_data_scaled[val_index]
            
            y_val_pred, y_val_true, y_val_probs = process_deep_model(X_train, X_val, y_train, y_val, model_class, model_params, batch_size=32, epochs=m_epochs)
            model_results = capture_metrics(y_val_true, y_val_pred, y_val_probs, f"{model_name} (Test)", model_results)
            logging.info(f"{model_name} model training and evaluation complete for Fold {fold + 1}")
        logging.info(f"{model_name} model cross-validation complete.")
    
    X_full_flattened = intervals.reshape(intervals.shape[0], -1)
    y_full = output_data_scaled.flatten()

    for model_name, model in classical_models.items():
        logging.info(f"Starting {model_name} model training")
        for fold, (train_index, val_index) in enumerate(skf.split(X_full_flattened, y_full)):
            logging.info(f"Processing Fold {fold + 1}")
            X_train, X_val = X_full_flattened[train_index], X_full_flattened[val_index]
            y_train, y_val = y_full[train_index], y_full[val_index]

            y_val_pred, y_val_probs = process_classical_model(X_train, X_val, y_train, y_val, model)
            model_results = capture_metrics(y_val, y_val_pred, y_val_probs, f"{model_name} (Test)", model_results)
            logging.info(f"{model_name} model training and evaluation complete for Fold {fold + 1}")
        logging.info(f"{model_name} model cross-validation complete.")
    
    return model_results


list_of_features = [ features_1, features_2, features_3, features_4]

interval_split = 48

logging.info("Started experiment for 12 hrs interval")

# List to store the models for evaluation
model_names = ['LSTM', 'DeepLSTM', 'Transformer', 'CNN-LSTM', 'CNN', 'DeepCNN', 'DeepCNN-LSTM', 'XGBoost', 'DecisionTree', 'RandomForest']

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

results = {}

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

        model_results = pipeline_run(intervals, output_data_scaled, interval_split, feature_size, m_epochs, model_results)

        # Logging the results at the end of each ID processing
        logging.info("------------------------------------------------------------------")
        logging.info(f"Processing completed for ID: {id_}")
        
        # Manually format and log model results with mean and std deviation
        for model_name, metrics in model_results.items():
            logging.info(f"Model: {model_name}")
            print(f"Model: {model_name}")
            
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
        results[f"id_{id_}_{index}_{interval_split}"] = model_results

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
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
df.to_csv(f"/home/rxb2495/binary_classification_benchmarking_results_{m_epochs}_epochs_{current_time}.csv", index=False)
