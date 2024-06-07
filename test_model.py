import joblib
import tensorflow as tf
import argparse

def test_model(data_file, model_file):
    # Load the datasets
    x_train, x_test, y_train, y_test = joblib.load(data_file)

    # Load the trained model
    model = tf.keras.models.load_model(model_file)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained model")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the prepared data file")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the trained model file")
    
    args = parser.parse_args()
    test_model(args.data_file, args.model_file)
