import numpy as np
import pandas as pd

def load_dataframe_from_npy(file_path):
    """
    Loads a .npy file containing a dictionary with 'columns' and 'data' keys, 
    and converts it into a pandas DataFrame.
    
    Parameters:
        file_path (str): The path to the .npy file.
    
    Returns:
        pd.DataFrame: DataFrame constructed from the data in the .npy file.
    """
    # Load the .npy file, assuming it contains a dictionary
    data_dict = np.load(file_path, allow_pickle=True).item()
    
    # Check if 'columns' and 'data' keys are in the dictionary
    if 'columns' not in data_dict or 'data' not in data_dict:
        raise ValueError("The .npy file must contain 'columns' and 'data' keys.")
    
    columns = data_dict['columns']
    data = data_dict['data']
    
    # Ensure data is in the correct format for DataFrame creation
    if not isinstance(columns, list):
        raise TypeError("'columns' should be a list of column names.")
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("'data' should be a list or numpy array of rows.")
    
    # Create and return the DataFrame
    return pd.DataFrame(data, columns=columns)

ids = ["MMCS0002", "MMCS0003", "MMCS0005", "MMCS0007", "MMCS0008", "MMCS0009", "MMCS0010", "MMCS0011", "MMCS0016"]

def categorize_glucose_levels(df, bins, labels, label_column_name):
    """
    Categorizes the glucose levels in the DataFrame into specified bins and adds a new label column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing glucose levels.
        bins (list): List of bin edges.
        labels (list): List of labels corresponding to the bins.
        label_column_name (str): The name of the new column to store the labels.
        
    Returns:
        pd.DataFrame: The DataFrame with the new label column added.
    """
    df[label_column_name] = pd.cut(
        df['Historic Glucose mg/dL'],
        bins=bins,
        labels=labels,
        right=True
    ).astype(int)
    return df

# Loop through the ids and categorize glucose levels using different bin sets
for id_ in ids:
    labels_data_path = f'/home/rxb2495/Glucose-Forecasting/final_data/{id_}_main_data.npy'
    output_data = load_dataframe_from_npy(labels_data_path)

    # Prepare output data for classification
    output_data = output_data[["Historic Glucose mg/dL"]]
    output_data = output_data.astype(np.float32)

    # Define different sets of bins and labels
    binning_scenarios = {
        "Scenario 1": {
            "bins": [0, 100, 140, float('inf')],
            "labels": [0, 1, 2]
        },
        "Scenario 2": {
            "bins": [0, 80, 140, float('inf')],
            "labels": [0, 1, 2]
        },
        "Scenario 3": {
            "bins": [0, 100, float('inf')],
            "labels": [0, 1]
        }
    }

    # Apply each binning scenario and log the results
    for scenario, params in binning_scenarios.items():
        bins = params["bins"]
        labels = params["labels"]
        label_column_name = f'Glucose_Label_{scenario.replace(" ", "_")}'

        # Categorize glucose levels
        categorized_data = categorize_glucose_levels(output_data.copy(), bins, labels, label_column_name)
        
        # Check value counts for the new labels
        glucose_label_counts = categorized_data[label_column_name].value_counts()
        glucose_label_counts_dict = glucose_label_counts.to_dict()

        # Log the value counts as a dictionary
        print(f"\n{scenario} - Glucose_Label value counts (as dictionary):")
        print(glucose_label_counts_dict)
