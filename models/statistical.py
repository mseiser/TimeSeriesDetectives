"""
This script demonstrates how to use the Isolation Forest algorithm for anomaly detection.
"""
import numpy as np
from anomaly_detection import AnomalyDetection


def detect_anomalies(time_series, threshold=3, window_size=200):
    """
    Detect anomalies in a time series data.
    
    Parameters:
    - time_series: List or array of time series data points
    - threshold: Number of standard deviations to consider a point as an anomaly
    - window_size: Number of points to consider for collective anomaly detection
    
    Returns:
    - anomaly_scores: List of anomaly scores for each time step
    """
    time_series = np.array(time_series)
    # Calculate mean and standard deviation of the time series
    mean = np.mean(time_series)
    std_dev = np.std(time_series)

    # Initialize list to store anomaly scores
    anomaly_scores = np.zeros(len(time_series))

    # Detect individual anomalies
    for i, point in enumerate(time_series):
        score = abs(point - mean) / std_dev
        anomaly_scores[i] = min(score / threshold, 1.0)

    # Detect collective anomalies
    for i in range(len(time_series) - window_size + 1):
        window = time_series[i:i + window_size]
        if np.sum(np.abs(window - mean) > threshold * std_dev) > 0:
            anomaly_scores[i:i + window_size] = 1.0
            break

    return anomaly_scores


ad = AnomalyDetection('statistical', detect_anomalies)
ad.test()
