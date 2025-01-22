import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from anomaly_detection import AnomalyDetection

def create_dataset(X,y,time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xv = X[i:(i + time_steps)]
        Xs.append(Xv)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def recreate_series(X):
    X_out = []
    max_epochs = X.shape[0]
    for index in range(0, max_epochs):
        X_out.append(X[index][-1][0])
    X_out = np.array(X_out)
    return X_out

# Plot the time series data with detected anomalies
def plot_anomalies(time_series_data, anomaly_indices):
    plt.figure(figsize=(20, 6))
    plt.plot(time_series_data, label='Time Series Data')
    plt.scatter(anomaly_indices, [time_series_data[i] for i in anomaly_indices], color='red', label='Detected Anomalies', zorder=5)
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Time Series Data with Detected Anomalies')
    plt.legend()
    plt.show()

# Load the time series data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.values.flatten().astype(float)

def LSTM_AE(time_series_data):
    time_series_data = np.array(time_series_data)
    prediction = np.zeros(time_series_data.shape)
    model = tf.keras.models.load_model('lstm_ae_128_64_32.keras')
    print("Model Loaded!!!!")

    scaler = MinMaxScaler()
    data_points = scaler.fit_transform(time_series_data.reshape(-1,1))
    data_points = data_points.flatten()

    X_TEST_SEQ, Y_TEST_SEQ = create_dataset(data_points,data_points,30)

    Y_TEST_SEQ_PRED = model.predict(X_TEST_SEQ, verbose=1)
    Y_TEST_SEQ_PRED = recreate_series(Y_TEST_SEQ_PRED)

    threshold = np.mean(np.abs(Y_TEST_SEQ.flatten(), Y_TEST_SEQ_PRED.flatten())) + 1.5*np.std(np.abs(Y_TEST_SEQ.flatten(), Y_TEST_SEQ_PRED.flatten()))
    testMAE = np.abs(Y_TEST_SEQ.flatten(), Y_TEST_SEQ_PRED.flatten())
    anomalies_indices = testMAE>threshold
    anomalies_indices = list(np.where(anomalies_indices==True)[0])
    anomalies_indices = sorted(anomalies_indices)

    for index in anomalies_indices: prediction[index] = 1

    return anomalies_indices, prediction

def main():
    time_series_data = load_data('X_test_2.csv')
    anomalies_indices, prediction = LSTM_AE(time_series_data)
    plot_anomalies(time_series_data, anomalies_indices)


ad = AnomalyDetection('lstm-ae', LSTM_AE)
ad.test()