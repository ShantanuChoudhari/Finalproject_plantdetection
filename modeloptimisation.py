# Convert to TensorFlow Lite for mobile deployment
def convert_to_tflite(model_path, output_path):
    """Convert Keras model to TensorFlow Lite"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to {output_path}")

# Usage
# convert_to_tflite(model, 'plant_disease_model.tflite')