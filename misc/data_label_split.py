import numpy as np
import pandas as pd


ids = ["MMCS0002", "MMCS0003", "MMCS0005", "MMCS0007", "MMCS0008", "MMCS0009", "MMCS0010", "MMCS0011", "MMCS0016"]


def load_dataframe_from_npy(file_path):
    """
    Loads a .npy file containing a dictionary with 'columns' and 'data' keys,
    and converts it into a pandas DataFrame.
    """
    data_dict = np.load(file_path, allow_pickle=True).item()
    columns = data_dict['columns']
    data = data_dict['data']
    return pd.DataFrame(data, columns=columns)

def categorize_glucose_levels(df, bins, labels, label_column_name):
    """
    Categorizes the glucose levels in the DataFrame into specified bins and adds a new label column.
    """
    df[label_column_name] = pd.cut(
        df['Historic Glucose mg/dL'],
        bins=bins,
        labels=labels,
        right=True
    ).astype(int)
    return df

def get_interval_labels(bins):
    """
    Generates interval labels based on the bins provided.
    """
    interval_labels = []
    for i in range(len(bins) - 1):
        if bins[i + 1] == float('inf'):
            interval_labels.append(f">{bins[i]}")
        else:
            interval_labels.append(f"{bins[i]}-{bins[i + 1]}")
    return interval_labels

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

# Dictionary to store results for each scenario
results = {scenario: [] for scenario in binning_scenarios}

# Loop through each ID and categorize data according to each scenario
for id_ in ids:
    labels_data_path = f'/home/rxb2495/Glucose-Forecasting/final_data/{id_}_main_data.npy'
    
    # Load the data for the current ID
    output_data = load_dataframe_from_npy(labels_data_path)
    
    # Prepare output data for classification
    output_data = output_data[["Historic Glucose mg/dL"]].astype(np.float32)

    # Apply each binning scenario and store the result
    for scenario, params in binning_scenarios.items():
        bins = params["bins"]
        labels = params["labels"]
        interval_labels = get_interval_labels(bins)
        label_column_name = f'Glucose_Label_{scenario.replace(" ", "_")}'

        # Categorize glucose levels
        categorized_data = categorize_glucose_levels(output_data.copy(), bins, labels, label_column_name)
        
        # Check value counts for the new labels
        glucose_label_counts = categorized_data[label_column_name].value_counts()
        glucose_label_counts_dict = glucose_label_counts.to_dict()

        # Rename the keys in the dictionary to use interval labels
        interval_counts_dict = {interval_labels[key]: value for key, value in glucose_label_counts_dict.items()}
        
        # Store the result in the results dictionary
        results[scenario].append({
            'ID': id_,
            **interval_counts_dict
        })

# Convert the results to DataFrames and save to CSV files
for scenario, data in results.items():
    df = pd.DataFrame(data)
    print(f"\n{scenario} DataFrame:")
    print(df.head())

    # Save to CSV file
    df.to_csv(f"{scenario.replace(" ", "_")}_glucose_classification_counts.csv", index=False)
    print(f"Saved {scenario} results to CSV.")
