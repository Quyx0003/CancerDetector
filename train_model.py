import joblib
import tensorflow as tf
import argparse

def train_model(data_file, model_file):
    # Load the datasets
    x_train, x_test, y_train, y_test = joblib.load(data_file)

    # Create the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
    model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=1000)

    # Save the trained model
    model.save(model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the prepared data file")
    parser.add_argument("--model_file", type=str, required=True, help="Path to save the trained model")
    
    args = parser.parse_args()
    train_model(args.data_file, args.model_file)
