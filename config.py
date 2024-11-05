# Configuration parameters for each model, including classical machine learning models

classification_config = {
    # Deep Learning Models
    "LSTMModel": {
        "input_size": 48,       # Number of features per timestep in the input
        "hidden_size": 128,     # Hidden size for LSTM layers
        "dropout": 0.2,         # Dropout rate for regularization
        "num_classes": 2        # Number of output classes
    },
    "DeepLSTMModel": {
        "input_size": 48,       # Number of features per timestep in the input
        "hidden_size": 256,     # Initial hidden size, progressively reduced in deeper layers
        "dropout": 0.3,         # Dropout rate for regularization across layers
        "num_classes": 2        # Number of output classes
    },
    "TransformerModel": {
        "input_size": 48,             # Number of features per timestep
        "d_model": 128,               # Dimension of the model embeddings
        "num_heads": 4,               # Number of attention heads
        "num_encoder_layers": 4,      # Number of Transformer encoder layers
        "dim_feedforward": 512,       # Size of the feedforward layer in each encoder block
        "dropout": 0.1,               # Dropout rate for regularization
        "max_seq_length": 500,        # Maximum sequence length for positional encoding
        "num_classes": 2              # Number of output classes
    },
    "CNNLSTMModel": {
        "input_size": 48,             # Number of features per timestep
        "num_filters": 64,            # Number of filters in the CNN layer
        "kernel_size": 3,             # Size of the convolutional kernel
        "lstm_hidden_size": 128,      # Hidden size for the LSTM layers
        "dropout": 0.2,               # Dropout rate for regularization
        "num_classes": 2              # Number of output classes
    },
    "CNNModel": {
        "input_size": 48,             # Number of features per timestep
        "num_filters": 64,            # Number of filters in the CNN layer
        "kernel_size": 3,             # Size of the convolutional kernel
        "dropout": 0.2,               # Dropout rate for regularization
        "num_classes": 2              # Number of output classes
    },
    "DeepCNNModel": {
        "input_size": 48,             # Number of features per timestep
        "num_filters": 64,            # Initial number of filters, doubled with each convolutional layer
        "kernel_size": 3,             # Size of the convolutional kernel
        "dropout": 0.3,               # Dropout rate for regularization
        "num_classes": 2              # Number of output classes
    },
    "DeepCNNLSTMModel": {
        "input_size": 48,             # Number of features per timestep
        "num_filters": 64,            # Initial number of filters in the CNN layers
        "kernel_size": 3,             # Size of the convolutional kernel
        "lstm_hidden_size": 128,      # Hidden size for LSTM layers
        "dropout": 0.3,               # Dropout rate for regularization
        "num_classes": 2              # Number of output classes
    },

    # Classical Machine Learning Models
    "XGBoost": {
        "n_estimators": 100,          # Number of trees in the ensemble
        "max_depth": 6,               # Maximum depth of each tree
        "learning_rate": 0.1,         # Step size shrinkage to prevent overfitting
        "subsample": 0.8,             # Subsample ratio of the training instances
        "colsample_bytree": 0.8,      # Subsample ratio of columns when constructing each tree
        "objective": "binary:logistic", # Binary classification objective
        "use_label_encoder": False,    # Disable the use of label encoder for binary classification
        "eval_metric": "logloss"      # Evaluation metric
    },
    "DecisionTree": {
        "criterion": "gini",          # Criterion for splitting (options: 'gini', 'entropy')
        "max_depth": None,            # Maximum depth of the tree; None means nodes expand until all leaves are pure
        "min_samples_split": 2,       # Minimum number of samples required to split an internal node
        "min_samples_leaf": 1,        # Minimum number of samples required to be at a leaf node
        "random_state": 42            # Seed for reproducibility
    },
    "RandomForest": {
        "n_estimators": 200,          # Number of trees in the forest
        "criterion": "gini",          # Criterion for splitting (options: 'gini', 'entropy')
        "max_depth": None,            # Maximum depth of the tree; None means nodes expand until all leaves are pure
        "min_samples_split": 2,       # Minimum number of samples required to split an internal node
        "min_samples_leaf": 1,        # Minimum number of samples required to be at a leaf node
        "bootstrap": True,            # Whether bootstrap samples are used when building trees
        "random_state": 42            # Seed for reproducibility
    }
}


m_epochs = 100

# Define the list of IDs
# ids = ["MMCS0002", "MMCS0003", "MMCS0005", "MMCS0007", "MMCS0008", "MMCS0009", "MMCS0010", "MMCS0011", "MMCS0016"]

ids = ["MMCS0002"]

# Define a dictionary to store results for each model and each ID
#time_values = [12, 24, 48, 96]

time_values = [48]

#========================================================================================================================================================

# Heart Rate Features
heart_rate_features_1 = [
    'HeartRate' 
]

heart_rate_features_2 = [
    'HeartRate_15_Mean', 'HeartRate_15_Std', 'HeartRate_30_Mean', 'HeartRate_30_Std','HeartRate_60_Mean', 'HeartRate_60_Std', 'HeartRate_90_Mean', 
    'HeartRate_90_Std', 'HeartRate_180_Mean', 'HeartRate_180_Std', 'HeartRate_360_Mean', 'HeartRate_360_Std', 'HeartRate_720_Mean', 'HeartRate_720_Std', 
    'HeartRate_1440_Mean', 'HeartRate_1440_Std'
]

heart_rate_features_3 = [
    'HeartRate_Diff_Lag_15_Mean_15', 'HeartRate_Diff_Lag_15_Std_15', 'HeartRate_Diff_Lag_30_Mean_30', 'HeartRate_Diff_Lag_30_Std_30', 
    'HeartRate_Diff_Lag_60_Mean_60', 'HeartRate_Diff_Lag_60_Std_60', 'HeartRate_Diff_Lag_90_Mean_90', 'HeartRate_Diff_Lag_90_Std_90']

heart_rate_features_4 = ['HeartRate_EWMA_15_Mean', 'HeartRate_EWMA_15_Std', 'HeartRate_EWMA_30_Mean', 'HeartRate_EWMA_30_Std',
    'HeartRate_EWMA_60_Mean', 'HeartRate_EWMA_60_Std', 'HeartRate_EWMA_90_Mean', 'HeartRate_EWMA_90_Std',
    'HeartRate_EWMA_180_Mean', 'HeartRate_EWMA_180_Std']

heart_rate_features_5 = [
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


###===============================================================================================================================
