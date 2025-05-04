import tensorflow as tf
import numpy as np


def convert_h5_to_tflite(h5_model_path, tflite_model_path, enable_quantization=False):
    """
    Convert a Keras .h5 model file to TensorFlow Lite (.tflite) format, with optional quantization.

    Args:
        h5_model_path (str): Path to the .h5 model file.
        tflite_model_path (str): Path to save the converted .tflite model file.
        enable_quantization (bool): Whether to enable model quantization. Default is False.
    """
    # Step 1: Load the .h5 model
    model = tf.keras.models.load_model(h5_model_path)
    print(f"Model loaded from: {h5_model_path}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # Step 2: Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Step 3: Enable quantization (optional)
    if enable_quantization:
        print("Enabling quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Provide a representative dataset for quantization (dummy data used here)
        def representative_data_gen():
            for _ in range(100):
                # Replace this with actual input data matching the model's input shape
                yield [np.random.random((1,) + model.input_shape[1:]).astype(np.float32)]

        converter.representative_dataset = representative_data_gen

    # Step 4: Convert the model to TFLite format
    tflite_model = converter.convert()
    print("Model successfully converted to TensorFlow Lite format.")

    # Step 5: Save the TFLite model to the specified path
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {tflite_model_path}")

    # Step 6: Test the converted TFLite model (optional)
    print("Testing the TFLite model...")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite Input Details:", input_details)
    print("TFLite Output Details:", output_details)

    # Generate a random input matching the model's input shape
    input_data = np.random.random(input_details[0]['shape']).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("Test Input Data:", input_data)
    print("Test Output Data:", output_data)


# Define paths to your model files
h5_model_path = "/home/kusabi/DL_Labor/dl-lab-24w-team06/Android_APP/best.complete_model.h5"  # Path to your .h5 model
tflite_model_path = "/home/kusabi/DL_Labor/dl-lab-24w-team06/Android_APP/model.tflite"  # Path to save .tflite model

# Convert the model with optional quantization
convert_h5_to_tflite(h5_model_path, tflite_model_path, enable_quantization=True)
