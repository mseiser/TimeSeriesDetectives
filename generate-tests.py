"""
This script generates test data for the anomaly detection model.
"""
import glob
import os

import numpy as np
import pandas as pd

# Define the paths for tests here
gt_path = "data/ground-truth/"
test_data_path = "data/generated-tests/"

n_single_anomalies = 150
n_collective_anomaly_min_size = 100


def store_data(x_test, y_test_solution, filename):
    x_test = pd.DataFrame(x_test)
    x_test.to_csv(test_data_path + 'X_' + filename + '.csv', index=False, header=False)
    y_test_solution = pd.DataFrame(y_test_solution)
    y_test_solution.to_csv(gt_path + 'gt_' + filename + '.csv', index=False, header=False)
    print(y_test_solution.sum()[0], ' anomalies generated.')


# -----------------------------------
# Naive Test Cases
# -----------------------------------

# Add block of zeros to the dataset
def exchange_with_zero(x_train, n):
    x_test = x_train.copy()
    y_test_solution = np.zeros(len(x_test))
    for i in range(n):
        row = np.random.randint(0, len(x_test))
        x_test.iloc[row] = 0
        y_test_solution[row] = 1
    return x_test, y_test_solution


def add_zero_block(x_train, start_index=100, n=1500):
    x_test = x_train.copy()
    y_test_solution = np.zeros(len(x_test))
    end_block = start_index + n
    for i in range(start_index, end_block):
        row = i
        x_test.iloc[row] = 0
        y_test_solution[row] = 1
    return x_test, y_test_solution


def add_peaks(x_train, n):
    x_test = x_train.copy()
    y_test_solution = np.zeros(len(x_test))
    for i in range(n):
        row = np.random.randint(i, i + 100)
        x_test.iloc[row] = ((np.random.uniform(0.75, 1.5, 1) * float(x_test.max().iloc[0]))
                            .astype(x_test.dtypes[0]))
        y_test_solution[row] = 1
    return x_test, y_test_solution


def add_valleys(x_train, n):
    x_test = x_train.copy()
    y_test_solution = np.zeros(len(x_test))
    for i in range(n):
        row = np.random.randint(i, i + 100)
        x_test.iloc[row] = ((np.random.uniform(0.25, -0.5, 1) * float(x_test.min().iloc[0]))
                            .astype(x_test.dtypes[0]))
        y_test_solution[row] = 1
    return x_test, y_test_solution


def add_peak_block(x_train, start_index=100):
    x_test = x_train.copy()
    y_test_solution = np.zeros(len(x_test))
    end_block = start_index + 1500
    for i in range(start_index, end_block):
        row = i
        x_test.iloc[row] = (x_test.max() * 0.75).astype(x_test.dtypes[0])
        y_test_solution[row] = 1
    return x_test, y_test_solution


def add_valley_block(x_train, start_index=100):
    x_test = x_train.copy()
    y_test_solution = np.zeros(len(x_test))
    start_block = start_index
    end_block = start_index + 1500
    for i in range(start_block, end_block):
        row = i
        x_test.iloc[row] = (x_test.min() * 0.25).astype(x_test.dtypes[0])
        y_test_solution[row] = 1
    return x_test, y_test_solution


# Duplicate values from the +100 index
# n = size of anomaly block
def duplicate_neighbour_block(x_train, n, start_index=100):
    x_test = x_train.copy()
    y_test_solution = np.zeros(len(x_test))
    end_block = start_index + n
    for i in range(start_index, end_block):
        row = i
        x_test.iloc[row] = x_test.iloc[row + 100]
        y_test_solution[row] = 1
    store_data(x_test, y_test_solution, 'dn')
    return x_test, y_test_solution


def duplicate_window_block(x_train, size, index, copy_index):
    x_test = x_train.copy()
    y_test_solution = np.zeros(len(x_test))
    end_block = index + size
    for i in range(index, end_block):
        row = i
        x_test.iloc[row] = x_test.iloc[copy_index]
        y_test_solution[row] = 1
    return x_test, y_test_solution


# -----------------------------------
def generate_single_anomalies(x_train, n):
    data, label1 = exchange_with_zero(x_train, n)
    data, label2 = add_peaks(data, n)
    data, label3 = add_valleys(data, n)
    labels = merge_labels(label1, label2, label3)
    return data, labels


def generate_multiple_constant_collective_anomalies(x_train):
    data, label1 = add_zero_block(x_train, start_index=100)
    data, label2 = add_valley_block(data, start_index=2000)
    data, label3 = add_peak_block(data, start_index=4000)
    labels = merge_labels(label1, label2, label3)
    return data, labels


def merge_labels(label1, label2, label3, label4=None):
    labels = np.zeros(len(label1))
    for i in range(len(label1)):
        if label1[i] == 1 or label2[i] == 1 or label3[i] == 1:
            labels[i] = 1
    return labels


def generate_multiple_collective_anomalies(x_train, n):
    data, label1 = add_valley_block(x_train, start_index=20)
    data, label2 = duplicate_neighbour_block(data, 350, start_index=1500)
    data, label3 = add_peak_block(data, start_index=2500)

    labels = merge_labels(label1, label2, label3)
    return data, labels


def generate_multiple_anomalies(x_train, n):
    data, label1 = add_peaks(x_train, 20)
    data, label2 = add_valleys(data, 10)
    data, label3 = duplicate_neighbour_block(data, 350, start_index=1500)
    data, label4 = add_valley_block(data, start_index=30000)
    labels = merge_labels(label1, label2, label3, label4)
    return data, labels


def generate_short_but_many_anomalies(x_train, n):
    # adds 600 anomalies to the dataset of 1000 entries
    test_data = x_train.copy()[0:1000]
    data, label1 = duplicate_neighbour_block(test_data, 250, start_index=500)
    data, label2 = add_zero_block(data, 200, 200)
    data, label3 = add_peaks(data, 200)
    data, label4 = add_valleys(data, 200)
    labels = merge_labels(label1, label2, label3, label4)
    return data, labels


#%%
def validate_csvs():
    for file in glob.glob(gt_path + '/*.csv'):
        gt_data = np.array(pd.read_csv(file, header=None))

        gt_data_anomalies = np.where(gt_data == 1)[0]
        print(f'Anomalies in {file}: {len(gt_data_anomalies)}')

        if len(gt_data_anomalies) == 0:
            print(f'No anomalies in {file}')
            continue

        if len(gt_data_anomalies) < n_single_anomalies:
            print(f'Less than minimum anomalies in {file}')


def main():
    input_file = "data/X_train.csv"
    # Read the input CSV file
    df = pd.read_csv(input_file, header=None)

    # Create folders
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)

    print("Created new folders.")

    # Generate Collective Anomaly Test Cases
    zero1 = add_zero_block(df, 90)
    store_data(zero1[0], zero1[1], 'zero_block')

    # Test Point anomalies
    x_test, y_test_solution = generate_single_anomalies(df, n_single_anomalies)
    store_data(x_test, y_test_solution, 'sa')

    # Test Single Constant Collective Anomaly
    x_test, y_test_solution = generate_multiple_constant_collective_anomalies(df)
    store_data(x_test, y_test_solution, 'cca')

    # Test Multiple Collective Anomalies
    x_test, y_test_solution = generate_multiple_collective_anomalies(df, n_collective_anomaly_min_size)
    store_data(x_test, y_test_solution, 'mca')

    # Test Multiple Anomalies
    x_test, y_test_solution = generate_multiple_anomalies(df, n_collective_anomaly_min_size)
    store_data(x_test, y_test_solution, 'ma')

    # Test Frequency-Based Collective Anomalies
    x_test, y_test_solution = duplicate_window_block(df, 350, 100, 3000)
    store_data(x_test, y_test_solution, 'fca')

    # Test Short but Many Anomalies
    x_test, y_test_solution = generate_short_but_many_anomalies(df, 400)
    store_data(x_test, y_test_solution, 'sbma')

    validate_csvs()
    print('Test data generated successfully.')



validate_csvs()