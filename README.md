# KDDM2 Challenge WS 24/25: Time Series Anomaly Detection

This repository contains the submission for the Time Series Anomaly Detection Challenge of Group 10.
The final model and the respective source code is located in the directory *final_model*, all models 
tested can be found under *models*.

The models were evaluated against custom test cases, including mostly naive tests as well as some 
advanced cases. The evaluation and test cases can be located in the directory *evaluation*

## Content of this repository
- */final_model*: Contains the final model and the respective source code
- */models*: Contains all models tested during the development
- */evaluation*: Contains the evaluation of the models
- */data*: Contains the input data and the generated test data
- *generate_test_files.py*: Script to generate test data
- *anomaly_detector.py*: Class to test models against test data
- *requirements.txt*: Required packages to run the code



### Guide for setting up the environment on Windows
To ensure that all files will run correctly, please use follow this guide.
1. Optional: Create a virtual environment using the following command:
```bash
python -m venv <name_of_virtual_environment>
```
2. Activate the virtual environment using the following command:
```bash
<name_of_virtual_environment>\Scripts\activate
```
3. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```



### Guide for creating the executable
The executable in final_model was created by using the following command:
```bash 
pip install pyinstaller
```
```bash
pyinstaller --onefile final_model.py
```

This creates an executable in a *output* directory. To run the executable on Windows, navigate to the directory and use the following 
command:
```bash
start ./final_model <path_to_input_file>
```

To regenerate the test-files for the evaluation, run the following command:
```bash
python generate_test_files.py
```
This will create the test data based on the normal input data and store the anomal data under *data/generated_tests*, 
as well as the respective labels under *data/ground-truth*.



### Testing the models
Each model implemented in the *models* directory can be tested using the functions provided by the AnomalyDetector class located in *anomaly_detector.py*.
To run a modela against all provided tests, simply run the following command:
```bash
python <name_of_model>.py 
```

### Running the evaluation
To run the evaluation, navigate to the *evaluation* directory use the *evaluation.ipynb*.
A summary of the PATE Scores of models regarding each test case can be found in the *results.csv*.
The evaluation features:
- PATE score calculation of all tested models
- Sensitivity Analysis of KNN Parameters

#### Authors
- Maria Seiser, seiser@student.tugraz.at    
- Vinayak Lal, vinayak.lal@student.tugraz.at   
- Bolis Hakim, bolis.hakim@student.tugraz.at
