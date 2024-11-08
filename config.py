# Configuration parameters for each model, including classical machine learning models

classification_config = {
    # Deep Learning Models
    "LSTMModel": {
        "input_size": 17,       # Number of features per timestep in the input
        "hidden_size": 128,     # Hidden size for LSTM layers
        "dropout": 0.2,         # Dropout rate for regularization
        "num_classes": 2        # Number of output classes
    },
    "DeepLSTMModel": {
        "input_size": 17,       # Number of features per timestep in the input
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


m_epochs = 1

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
    'HeartRate_15_Mean', 'HeartRate_15_Std', 
    'HeartRate_30_Mean', 'HeartRate_30_Std', 
    'HeartRate_60_Mean', 'HeartRate_60_Std', 
    'HeartRate_90_Mean', 'HeartRate_90_Std', 
    'HeartRate_180_Mean', 'HeartRate_180_Std', 
    'HeartRate_240_Mean', 'HeartRate_240_Std', 
    'HeartRate_360_Mean', 'HeartRate_360_Std', 
    'HeartRate_720_Mean', 'HeartRate_720_Std', 
    'HeartRate_1440_Mean', 'HeartRate_1440_Std',
]

heart_rate_features_3 = [
    'HeartRate_Lag_5', 'HeartRate_Lag_15', 'HeartRate_Lag_30', 'HeartRate_Lag_60', 'HeartRate_Lag_120',
    'HeartRate_Lag_180', 'HeartRate_Lag_240', 'HeartRate_Lag_360', 'HeartRate_Lag_720', 'HeartRate_Lag_1440',
    'HeartRate_Diff_Lag_15', 'HeartRate_Diff_Lag_30', 'HeartRate_Diff_Lag_60', 'HeartRate_Diff_Lag_90',
    'HeartRate_Diff_Lag_180', 'HeartRate_Diff_Lag_240', 'HeartRate_Diff_Lag_360', 'HeartRate_Diff_Lag_720', 'HeartRate_Diff_Lag_1440']

heart_rate_features_4 = [ 'HeartRate_Diff_Lag_15_Mean_15', 'HeartRate_Diff_Lag_15_Std_15', 'HeartRate_Diff_Lag_30_Mean_30', 'HeartRate_Diff_Lag_30_Std_30',
    'HeartRate_Diff_Lag_60_Mean_60', 'HeartRate_Diff_Lag_60_Std_60', 'HeartRate_Diff_Lag_90_Mean_90', 'HeartRate_Diff_Lag_90_Std_90',
    'HeartRate_Diff_Lag_180_Mean_180', 'HeartRate_Diff_Lag_180_Std_180', 'HeartRate_Diff_Lag_240_Mean_240', 'HeartRate_Diff_Lag_240_Std_240',
    'HeartRate_Diff_Lag_360_Mean_360', 'HeartRate_Diff_Lag_360_Std_360', 'HeartRate_Diff_Lag_720_Mean_720', 'HeartRate_Diff_Lag_720_Std_720',
    'HeartRate_Diff_Lag_1440_Mean_1440', 'HeartRate_Diff_Lag_1440_Std_1440']

heart_rate_features_5 = [
    'HeartRate_RoC_5', 'HeartRate_RoC_15', 'HeartRate_RoC_30', 'HeartRate_RoC_60', 'HeartRate_RoC_120',
    'HeartRate_RoC_180', 'HeartRate_RoC_240', 'HeartRate_RoC_360', 'HeartRate_RoC_720', 'HeartRate_RoC_1440',
]

# Sleep Features

sleep_features_1 = [
    'Sleep'
]

sleep_features_2 = [
     'Sleep_15min_Mean', 'Sleep_15min_Std',
    'Sleep_30min_Mean', 'Sleep_30min_Std',
    'Sleep_60min_Mean', 'Sleep_60min_Std',
    'Sleep_90min_Mean', 'Sleep_90min_Std',
    'Sleep_180min_Mean', 'Sleep_180min_Std',
    'Sleep_240min_Mean', 'Sleep_240min_Std',
    'Sleep_360min_Mean', 'Sleep_360min_Std',
    'Sleep_720min_Mean', 'Sleep_720min_Std',
    'Sleep_1440min_Mean', 'Sleep_1440min_Std',
]

sleep_features_3 = [
   'Sleep_Lag_5min', 'Sleep_Lag_15min', 'Sleep_Lag_30min',
    'Sleep_Lag_60min', 'Sleep_Lag_120min', 'Sleep_Lag_180min', 'Sleep_Lag_240min',
    'Sleep_Lag_360min', 'Sleep_Lag_720min', 'Sleep_Lag_1440min', 'Sleep_Lag_Diff_15min',
    'Sleep_Lag_Diff_30min', 'Sleep_Lag_Diff_60min', 'Sleep_Lag_Diff_90min', 'Sleep_Lag_Diff_180min',
    'Sleep_Lag_Diff_240min', 'Sleep_Lag_Diff_360min', 'Sleep_Lag_Diff_720min', 'Sleep_Lag_Diff_1440min'
]

sleep_features_4 = [
     'Sleep_Lag_Diff_15min_15_Mean', 'Sleep_Lag_Diff_15min_15_Std',
     'Sleep_Lag_Diff_30min_30_Mean', 'Sleep_Lag_Diff_30min_30_Std',
     'Sleep_Lag_Diff_60min_60_Mean', 'Sleep_Lag_Diff_60min_60_Std',
     'Sleep_Lag_Diff_90min_90_Mean', 'Sleep_Lag_Diff_90min_90_Std',
    'Sleep_Lag_Diff_180min_180_Mean', 'Sleep_Lag_Diff_180min_180_Std',
     'Sleep_Lag_Diff_240min_240_Mean', 'Sleep_Lag_Diff_240min_240_Std',
     'Sleep_Lag_Diff_360min_360_Mean', 'Sleep_Lag_Diff_360min_360_Std',
     'Sleep_Lag_Diff_720min_720_Mean', 'Sleep_Lag_Diff_720min_720_Std',
     'Sleep_Lag_Diff_1440min_1440_Mean', 'Sleep_Lag_Diff_1440min_1440_Std',
]

sleep_features_5 = [
    'Sleep_RoC_5min', 'Sleep_RoC_15min', 'Sleep_RoC_30min',
    'Sleep_RoC_60min', 'Sleep_RoC_120min', 'Sleep_RoC_180min', 'Sleep_RoC_240min',
    'Sleep_RoC_360min', 'Sleep_RoC_720min', 'Sleep_RoC_1440min',
]

# Intensity Features

intensity_features_1 = [
    'Intensity', 
]

intensity_features_2 = [
    'Intensity_15min_Mean', 'Intensity_15min_Std', 
    'Intensity_30min_Mean', 'Intensity_30min_Std', 
    'Intensity_60min_Mean', 'Intensity_60min_Std',  
    'Intensity_90min_Mean', 'Intensity_90min_Std', 
    'Intensity_180min_Mean', 'Intensity_180min_Std', 
    'Intensity_240min_Mean', 'Intensity_240min_Std', 
    'Intensity_360min_Mean', 'Intensity_360min_Std',
    'Intensity_720min_Mean', 'Intensity_720min_Std', 
    'Intensity_1440min_Mean', 'Intensity_1440min_Std',
]

intensity_features_3 = [
     'Intensity_Lag_5min', 'Intensity_Lag_15min', 'Intensity_Lag_30min', 'Intensity_Lag_60min', 'Intensity_Lag_120min', 
    'Intensity_Lag_180min', 'Intensity_Lag_240min', 'Intensity_Lag_360min', 'Intensity_Lag_720min', 'Intensity_Lag_1440min',
]

intensity_features_4 = [
   'Intensity_Lag_Diff_15min', 'Intensity_Lag_Diff_15min_Mean', 'Intensity_Lag_Diff_15min_Std',
    'Intensity_Lag_Diff_30min', 'Intensity_Lag_Diff_30min_Mean', 'Intensity_Lag_Diff_30min_Std',
    'Intensity_Lag_Diff_60min', 'Intensity_Lag_Diff_60min_Mean', 'Intensity_Lag_Diff_60min_Std',
    'Intensity_Lag_Diff_90min', 'Intensity_Lag_Diff_90min_Mean', 'Intensity_Lag_Diff_90min_Std',
    'Intensity_Lag_Diff_180min', 'Intensity_Lag_Diff_180min_Mean', 'Intensity_Lag_Diff_180min_Std',
    'Intensity_Lag_Diff_240min', 'Intensity_Lag_Diff_240min_Mean', 'Intensity_Lag_Diff_240min_Std',
    'Intensity_Lag_Diff_360min', 'Intensity_Lag_Diff_360min_Mean', 'Intensity_Lag_Diff_360min_Std',
    'Intensity_Lag_Diff_720min', 'Intensity_Lag_Diff_720min_Mean', 'Intensity_Lag_Diff_720min_Std',
    'Intensity_Lag_Diff_1440min', 'Intensity_Lag_Diff_1440min_Mean', 'Intensity_Lag_Diff_1440min_Std',
]

intensity_features_5 = [
    'Intensity_RoC_5min', 'Intensity_RoC_15min', 'Intensity_RoC_30min', 
    'Intensity_RoC_60min', 'Intensity_RoC_120min', 'Intensity_RoC_180min', 'Intensity_RoC_240min', 
    'Intensity_RoC_360min', 'Intensity_RoC_720min', 'Intensity_RoC_1440min',
]

# Steps Features

steps_features_1 = [
    'Steps', 
]

steps_features_2 = [
    'Steps_15min_Mean', 'Steps_15min_Std', 'Steps_30min_Mean', 'Steps_30min_Std', 
    'Steps_60min_Mean', 'Steps_60min_Std', 'Steps_90min_Mean', 'Steps_90min_Std', 
    'Steps_180min_Mean', 'Steps_180min_Std', 'Steps_240min_Mean', 'Steps_240min_Std', 
    'Steps_360min_Mean', 'Steps_360min_Std', 'Steps_720min_Mean', 'Steps_720min_Std', 
    'Steps_1440min_Mean', 'Steps_1440min_Std',
]

steps_features_3 = [
    'Steps_Lag_5min', 'Steps_Lag_15min', 'Steps_Lag_30min', 'Steps_Lag_60min', 
    'Steps_Lag_120min', 'Steps_Lag_180min', 'Steps_Lag_240min', 'Steps_Lag_360min', 'Steps_Lag_720min', 'Steps_Lag_1440min',
    'Steps_Lag_Diff_5min', 'Steps_Lag_Diff_15min', 'Steps_Lag_Diff_30min', 'Steps_Lag_Diff_60min', 
    'Steps_Lag_Diff_120min', 'Steps_Lag_Diff_180min', 'Steps_Lag_Diff_240min', 'Steps_Lag_Diff_360min', 'Steps_Lag_Diff_720min', 'Steps_Lag_Diff_1440min',
]

steps_features_4 = [
    'Steps_Lag_Diff_15min_15min_Mean', 'Steps_Lag_Diff_15min_15min_Std',
    'Steps_Lag_Diff_30min_30min_Mean', 'Steps_Lag_Diff_30min_30min_Std',
    'Steps_Lag_Diff_60min_60min_Mean', 'Steps_Lag_Diff_60min_60min_Std',
    'Steps_Lag_Diff_90min_90min_Mean', 'Steps_Lag_Diff_90min_90min_Std',
    'Steps_Lag_Diff_180min_180min_Mean', 'Steps_Lag_Diff_180min_180min_Std',
    'Steps_Lag_Diff_240min_240min_Mean', 'Steps_Lag_Diff_240min_240min_Std',
    'Steps_Lag_Diff_360min_360min_Mean', 'Steps_Lag_Diff_360min_360min_Std',
    'Steps_Lag_Diff_720min_720min_Mean', 'Steps_Lag_Diff_720min_720min_Std',
    'Steps_Lag_Diff_1440min_1440min_Mean', 'Steps_Lag_Diff_1440min_1440min_Std',
]

steps_features_5 = [
    'Steps_RoC_5min', 'Steps_RoC_15min', 'Steps_RoC_30min', 
    'Steps_RoC_60min', 'Steps_RoC_120min', 'Steps_RoC_180min', 'Steps_RoC_240min', 
    'Steps_RoC_360min', 'Steps_RoC_720min', 'Steps_RoC_1440min'
]


temporal_features = [
     'MinuteOfDay_Sin', 
     'MinuteOfDay_Cos',
     'DayOfWeek_Sin',
     'DayOfWeek_Cos',
     'Is_Weekend'
]


###===============================================================================================================================
