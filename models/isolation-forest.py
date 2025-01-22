"""
This script demonstrates how to use the Isolation Forest algorithm for anomaly detection.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from anomaly_detection import AnomalyDetection


def extract_features(windows):
    return np.array([[np.mean(window), np.std(window)] for window in windows])


def plot_anomalies(time_series_data, anomaly_indices):
    plt.figure(figsize=(20, 6))
    plt.plot(time_series_data, label='Time Series Data')
    plt.scatter(anomaly_indices, [time_series_data[i] for i in anomaly_indices], color='red',
                label='Detected Anomalies', zorder=5)
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Time Series Data with Detected Anomalies')
    plt.legend()
    plt.show()


class IsolationForestAnomalyDetector:
    def __init__(self, window_size=100, threshold=-0.001, contamination=0.01):
        self.window_size = window_size
        self.threshold = threshold
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination)

    def extract_windows(self, time_series):
        return np.array([time_series[i:i + self.window_size] for i in range(len(time_series) - self.window_size + 1)])

    def fit_isolation_forest(self, features):
        self.model.fit(features)

    def make_predictions(self, features):
        scores = self.model.decision_function(features)
        return scores

    def get_anomaly_indices(self, scores):
        anomalies = np.where(scores < self.threshold)[0]
        return anomalies

    def detect_anomalies(self, data):
        windows = self.extract_windows(data)
        features = extract_features(windows)

        self.fit_isolation_forest(features)
        scores = self.make_predictions(features)

        anomaly_indices = self.get_anomaly_indices(scores)

        results = np.zeros(len(data))
        for idx in anomaly_indices:
            results[idx] = 1
        return results


def detect_anomalies(data):
    detector = IsolationForestAnomalyDetector(window_size=100, threshold=-0.01)
    return detector.detect_anomalies(data)


ad = AnomalyDetection(model_name='isolation-forest', detection_method=detect_anomalies)
ad.test()
