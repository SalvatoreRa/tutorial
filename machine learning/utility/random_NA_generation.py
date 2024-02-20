import numpy as np

def introduce_missing_values(dataset, missing_percentage):
    """
    Introduces random missing values into a dataset.

    Parameters:
    dataset (numpy.ndarray): The input dataset.
    missing_percentage (float): The percentage of missing values to introduce, expressed as a float between 0 and 1.

    Returns:
    numpy.ndarray: The dataset with missing values introduced.
    """
    dataset = dataset.astype(float)
    # Calculate the number of missing values to introduce
    num_missing = int(dataset.size * missing_percentage)

    # Generate random indices to place missing values
    missing_indices = np.random.choice(dataset.size, num_missing, replace=False)

    # Flatten the dataset and introduce missing values at random indices
    flattened_dataset = dataset.flatten()
    flattened_dataset[missing_indices] = np.nan

    # Reshape the dataset back to its original shape
    dataset_with_missing = flattened_dataset.reshape(dataset.shape)

    return dataset_with_missing