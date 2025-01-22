"""
Following code implements/runs baselines for the anomaly detection task.
"""

from anomaly_detection import AnomalyDetection
from kan import KANBaseline
import numpy as np


def baseline_anomal(data):
    return np.ones(len(data))


ad = AnomalyDetection('baseline-anomal', baseline_anomal)
ad.test()


kan = KANBaseline()
ad = AnomalyDetection('baseline-kan', kan.detect_anomalies)
ad.test()
