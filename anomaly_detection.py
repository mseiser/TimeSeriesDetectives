"""
    This script handles all general anomaly detection tasks,
    not specific to any model. For example useful test methods to automatically generate test results,
    or general preprocessing methods, like normalization.
"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class AnomalyDetection:

    def __init__(self, model_name, detection_method, window_size=200):
        """
        The parameters for intializing the class are:
        :param model_name: The name of the model, as a string
        :param detection_method: The method used for detecting anomalies
        """
        self.data = None
        self.window_size = 200
        self.adf_statistic = None
        self.model_name = model_name
        self.detection_method = detection_method
        self.anomalies = []

    def prepare_data(self, data):
        """
            The method applies following preprocessing steps to the data:
            - Normalization
            :param data: The time series data, as a pandas DataFrame
        """
        time_series_data = data.values.flatten().astype(float)
        scaler = StandardScaler()
        time_series_data = scaler.fit_transform(time_series_data.reshape(-1, 1)).flatten()
        self.data = time_series_data
        return time_series_data

    def testfile(self, input_file):
        """
        The method tests the model on a single file.
        :param input_file: path to the input file
        :return:
        """
        data = pd.read_csv(input_file, header=None)
        self.prepare_data(data)
        self.anomalies = self.detection_method(self.data)
        output_df = pd.DataFrame(self.anomalies)
        return output_df

    def test_with_labels(self, input_file, index):
        """
        The method tests the model on a single file with known labels.
        :param input_file: path to the input file
        :param labels: path to the labels file
        :return:
        """

        try:
            data = pd.read_csv(input_file, header=None)[0]

            labels = glob.glob("../data/ground-truth/*.csv")
            if len(labels) == 0:
                print("No labels found")
                labels = pd.DataFrame(np.ones(len(data)))
            else:
                labels = labels[index]

            labels = pd.read_csv(labels, header=None)[0]
            self.prepare_data(data)
            self.anomalies = self.detection_method(self.data, labels)
            output_df = pd.DataFrame(self.anomalies)
        except:
            print("An error occurred while testing with labels")
            output_df = pd.DataFrame(np.zeros(len(data)))

        return output_df

    def test(self, test_file_path="../data/generated-tests/"):
        """
        The method tests the model on all the test files found in the test_file_path directory.
        :param test_file_path: The path to the directory containing the test files
        :return:
        """
        output_file_path = "../data/model-results/" + self.model_name + "/"

        for index, file in enumerate(glob.glob(test_file_path + '/*.csv')):
            if self.model_name == "baseline-kan":
                output = self.test_with_labels(file, index)
            else:
                output = self.testfile(file)
            output_file = output_file_path + file.split('\\')[-1]
            output.to_csv(output_file, index=False, header=False)
            print(f"Anomaly scores saved to {output_file}")

    def exe_main(self):
        """
        This is a general main method to be used for testing models when building executables or
        testing with the terminal.
        :return:
        """
        if len(sys.argv) != 2:
            print("Usage: python script.py <input_csv_file>")
            sys.exit(1)

        input_file = sys.argv[1]

        output = self.testfile(input_file)

        output_file = "predictions-group10.csv"
        output.to_csv(output_file, index=False, header=False)

        print(f"Anomaly scores saved to {output_file}")
