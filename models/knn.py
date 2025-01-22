"""
This script demonstrates how to use the KNN model for anomaly detection.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import sys

from anomaly_detection import AnomalyDetection


# Apply moving average to time series data
def moving_average(data, window_size):
    """
    Apply moving average to the time series data.

    Parameters:
    data (pd.Series): Time series data.
    window_size (int): Size of the moving average window.

    Returns:
    pd.Series: Time series data after applying moving average.
    """
    return data - data.rolling(window=window_size).mean().dropna()


def normalize_data(data):
    """
    Normalize the time series data to the range [0, 1].

    Parameters:
    data (np.ndarray): Array of time series data.

    Returns:
    np.ndarray: Normalized time series data.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()


def extract_windows(time_series, window_size):
    """
    Extract overlapping windows from the time series data.

    Parameters:
    time_series (np.ndarray): Array of time series data.
    window_size (int): Size of each window.

    Returns:
    np.ndarray: Array of windows.
    """
    return np.array([time_series[i:i + window_size] for i in range(len(time_series) - window_size + 1)])


def extract_features(windows):
    """
    Extract features (mean and standard deviation) from each window.

    Parameters:
    windows (np.ndarray): Array of windows.

    Returns:
    np.ndarray: Array of feature vectors.
    """
    return np.array([[np.mean(window), np.std(window)] for window in windows])


def knn_anomaly_detection(features, k):
    """
    Perform KNN anomaly detection on the feature vectors.

    Parameters:
    features (np.ndarray): Array of feature vectors.
    k (int): Number of neighbors for KNN.

    Returns:
    np.ndarray: Distances to the k-nearest neighbors.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(features)
    distances, _ = nbrs.kneighbors(features)
    return distances


def get_indices(distances, threshold=3):
    """
    Detect anomalies based on the distances to the k-nearest neighbors.

    Parameters:
    distances (np.ndarray): Distances to the k-nearest neighbors.
    threshold (int): Threshold for anomaly detection.

    Returns:
    np.ndarray: Indices of detected anomalies.
    """
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    return np.where(distances > mean_distance + threshold * std_distance)[0]


def preprocess(data):
    data = moving_average(pd.Series(data), window_size=400).dropna().values
    data = normalize_data(data)
    return data


def detect_anomalies(data, window_size=440, k=145, threshold=3):
    """
    Detect anomalies in the time series data using KNN.
    :param data:
    :param window_size:
    :param k:
    :param threshold:
    :return: score for each data point
    """
    ts_data = preprocess(data)
    windows = extract_windows(ts_data, window_size)
    features = extract_features(windows)
    distances = knn_anomaly_detection(features, k)
    anomaly_indices = get_indices(distances[:, -1], threshold)

    # Create prediction array
    pred = np.zeros(len(data))
    adjusted_anomaly_indices = [i + window_size for i in anomaly_indices]
    pred[adjusted_anomaly_indices] = 1
    return pred


def main():
    """
    Main function to run the anomaly detection and save results to a CSV file.

    Parameters:
    file_path (str): Path to the input CSV file.
    output_file (str): Path to the output CSV file.
    window_size (int): Size of each window.
    k (int): Number of neighbors for KNN.
    threshold (int): Threshold for anomaly detection.
    """

    if len(sys.argv) != 2:
        print("Usage: python script.py <input_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    data = pd.read_csv(input_file, header=None)
    data = data.values.flatten().astype(float)
    output = knn(data)

    output_file = "predictions-group10.csv"
    output = pd.DataFrame(output)
    output.to_csv(output_file, index=False, header=False)

    print(f"Anomaly scores saved to {output_file}")


ad = AnomalyDetection('knn', detect_anomalies)
ad.test()
