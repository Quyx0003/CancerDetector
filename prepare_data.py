import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import argparse

def prepare_data(input_file, output_file):
    # Load dataset
    dataset = pd.read_csv(input_file)

    # Prepare input and output data
    x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
    y = dataset["diagnosis(1=m, 0=b)"]

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Save the datasets to disk
    joblib.dump((x_train, x_test, y_train, y_test), output_file)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training and testing")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the prepared data")
    
    args = parser.parse_args()
    prepare_data(args.input_file, args.output_file)
