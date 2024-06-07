
# Cancer Detection Model

This repository contains scripts to prepare data, train a machine learning model, and test the model for predicting whether a tumor is malignant or benign based on certain features.

## Files

1. `prepare_data.py` - Prepares the dataset and splits it into training and testing sets.
2. `train_model.py` - Trains the model using the prepared data and saves the trained model.
3. `test_model.py` - Loads the trained model and evaluates it on the test data.

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- TensorFlow
- joblib

You can install the required packages using the following command:

```
pip install pandas scikit-learn tensorflow joblib
```

## Usage

### Step 1: Prepare Data

First, prepare the dataset by running the `prepare_data.py` script. This script loads the dataset, splits it into training and testing sets, and saves these sets to a file.

```
python prepare_data.py --input_file cancer.csv --output_file data.pkl
```

### Step 2: Train the Model

Next, train the model using the prepared data by running the `train_model.py` script. This script loads the training data, trains a neural network model, and saves the trained model to a file.

```
python train_model.py --data_file data.pkl --model_file cancer_model.h5
```

### Step 3: Test the Model

Finally, evaluate the trained model on the test data by running the `test_model.py` script. This script loads the test data and the trained model, and then evaluates the model's performance.

```
python test_model.py --data_file data.pkl --model_file cancer_model.h5
```

## Detailed Explanation

### `prepare_data.py`

This script performs the following steps:
1. Loads the dataset from `cancer.csv`.
2. Prepares the input (`x`) and output (`y`) data.
3. Splits the data into training and testing sets.
4. Saves the training and testing sets to a file named `data.pkl`.

### `train_model.py`

This script performs the following steps:
1. Loads the training and testing sets from `data.pkl`.
2. Defines and compiles a neural network model using TensorFlow.
3. Trains the model on the training data.
4. Saves the trained model to a file named `cancer_model.h5`.

### `test_model.py`

This script performs the following steps:
1. Loads the training and testing sets from `data.pkl`.
2. Loads the trained model from `cancer_model.h5`.
3. Evaluates the model on the test data and prints the accuracy and loss.

## Example Output

After running `test_model.py`, you should see output similar to the following:

```
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9498 - loss: 0.0816
Loss: 0.0816, Accuracy: 0.9498
```

This indicates that the model achieved an accuracy of approximately 94.98% on the test data with a loss of 0.0816.

