"""
This module contains the implementation of the HotSAX algorithm for anomaly detection in time series data.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt

from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from anomaly_detection import AnomalyDetection


class InputError(Exception):
    def __init__(self, message):
        self.message = message


# Load the time series data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.values.flatten().astype(float)


def fit(data, x_length, window_size):
    nb_of_segments = x_length - window_size + 1
    x_segments = [data[:, i:i + window_size] for i in range(nb_of_segments)]
    return x_segments


def euclidean_distance(input1, input2, window_size, X_length):
    """
    Calculate the Euclidean distance between two numpy arrays
    :param input1: a numpy array
    :param input2: a numpy array
    :return: the Euclidean distance between the two input arrays multiplied by the square root of the ratio of
    the window length to the length of the time series.
    """
    if isinstance(input1, np.ndarray) and isinstance(input2, np.ndarray):
        if input1.shape != input2.shape:
            raise InputError(
                f"Mismatch in the Euclidean distance calculation: first input has shape {input1.shape} "
                f"and second input has shape {input2.shape}")

        distance = np.linalg.norm(input1 - input2, ord=2)
        distance *= np.sqrt(window_size / X_length)
        return distance
    else:
        raise InputError("Can only calculate distance between numpy arrays.")


def brute_forcing_multiprocessing(input_data):
    x_segments, i, p, window_size, best_dist, X_length = input_data
    x_segments = np.array(x_segments)
    print(f"AT ITERATION: {i}")
    nearest_dist = np.inf
    for j, q in enumerate(x_segments):
        if np.abs(i - j) >= window_size:
            dist = np.linalg.norm(p - q)
            # dist = euclidean_distance(p, q, window_size, X_length)
            if dist < nearest_dist:
                nearest_dist = dist
        if nearest_dist > best_dist: best_dist = nearest_dist
    return (i, nearest_dist)


def brute_force_ad_detection(x_segments, window_size, best_dist, best_loc, all_discords, X_length):
    print("INSIDE")
    print(f"USING {5} WORKERS....... DANGER TO USE MORE THAN 10!!!")
    inputs = [(x_segments, index, value, window_size, best_dist, X_length) for index, value in enumerate(x_segments)]
    with ProcessPoolExecutor(max_workers=10) as executor: results = executor.map(brute_forcing_multiprocessing, inputs)
    results = [r for r in results]
    print(len(results))
    for i, nearest_dist in results: all_discords[i] = nearest_dist
    print(len(all_discords))
    return all_discords


def transform(x_segments, window_size, best_dist, best_loc, all_discords, X_length):
    all_discords = brute_force_ad_detection(x_segments, window_size, best_dist, best_loc, all_discords, X_length)
    all_discords = {k: v for k, v in sorted(all_discords.items(), key=lambda x: -x[1])}
    return all_discords


def list_anomalies(all_discords, nb_discords):
    return_list = []
    for d in enumerate(all_discords):
        if d[0] <= nb_discords - 1:
            # print(f"Discord {(d[0] + 1)} located at index {d[1]}")
            return_list.append(d[1])
        else:
            break
    return return_list


def find_top_n_consecutive_sequences_pandas(nums, top_n=3):
    if not nums:
        return []

    df = pd.DataFrame({'num': nums})
    df['diff'] = df['num'] - df['num'].shift(1)
    df['new_seq'] = (df['diff'] != 1).cumsum()

    grouped = df.groupby('new_seq')['num'].apply(list)
    # Sort the groups by the length of sequences in descending order
    grouped_sorted = grouped.sort_values(key=lambda x: x.str.len(), ascending=False)
    # Retrieve top N sequences
    top_sequences = grouped_sorted.head(top_n).tolist()

    return top_sequences


def find_longest_consecutive_sequences(nums):
    if not nums:
        return []

    sequences = []
    current_sequence = [nums[0]]

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            current_sequence.append(nums[i])
        else:
            sequences.append(current_sequence)
            current_sequence = [nums[i]]

    # Append the last sequence
    sequences.append(current_sequence)

    # Find the maximum length
    max_length = max(len(seq) for seq in sequences)

    # Retrieve all sequences with the maximum length
    longest_sequences = [seq for seq in sequences if len(seq) == max_length]

    return longest_sequences


# Plot the time series data with detected anomalies
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


def HotSax(time_series_data):
    # time_series_data = np.array(pd.Series(pd.Series(time_series_data) - pd.Series(time_series_data).rolling(window=400).mean().dropna()).dropna())
    scaler = MinMaxScaler()
    data_points = scaler.fit_transform(time_series_data.reshape(-1, 1))
    data_points = data_points.flatten()
    print(data_points.shape)

    best_dist = 0
    best_loc = np.nan
    window_size = 300
    nb_discords = 600
    all_discords = defaultdict(float)
    data_points = data_points.reshape(1, -1)
    X_length = data_points.shape[1]
    anomaly_indices = np.zeros(time_series_data.shape)

    print("CURRENT ANOMLAY INDICES ARE:", len(anomaly_indices))

    x_segments = fit(data_points, X_length, window_size)
    all_discords = transform(x_segments, window_size, best_dist, best_loc, all_discords, X_length)
    anomalies = list_anomalies(all_discords, nb_discords)

    anomalies = sorted(anomalies)
    longest_seqs = find_top_n_consecutive_sequences_pandas(anomalies)
    filtered_anomalies = []
    print("Longest consecutive sequences:")
    for seq in longest_seqs:
        for index in range(seq[0], seq[-1] + 1):
            filtered_anomalies.append(index)
            anomaly_indices[index] = 1
        print(seq[0], seq[-1] + 1)

    return filtered_anomalies, anomaly_indices


# def main(file_path, window_size=100, nb_discords=200):
#     time_series_data = load_data(file_path)

#     scaler = StandardScaler()
#     data_points = scaler.fit_transform(time_series_data.reshape(-1,1))
#     data_points = data_points.flatten()
#     print(data_points.shape)

#     best_dist = 0
#     best_loc = np.nan
#     all_discords = defaultdict(float)
#     NUM_WORKERS = multiprocessing.cpu_count() - 1

#     print("NUMBER OF CPU AVIALABLE: ", NUM_WORKERS)
#     data_points = data_points.reshape(1,-1)
#     X_length = data_points.shape[1]

#     x_segments = fit(data_points, X_length, window_size)
#     all_discords = transform(x_segments, window_size, best_dist, best_loc, all_discords, NUM_WORKERS, X_length)
#     anomalies = list_anomalies(all_discords, nb_discords)

#     print(anomalies)

#if __name__ == "__main__":
#    time_series_data = load_data('X_test_2.csv')
#
#    anomalies, prediction = HotSax(time_series_data)
#    plot_anomalies(time_series_data, anomalies)
#
#    print(np.where(prediction == 1))

if __name__ == "__main__":
    ad = AnomalyDetection('hotsax', HotSax)
    ad.test()