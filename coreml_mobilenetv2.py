import coremltools as ct
import tensorflow as tf # TF 2.2.0

# Load MobileNetV2.
keras_model = tf.keras.applications.MobileNetV2()
input_name = keras_model.input_names[0]

# Convert to Core ML with an MLMultiArray for input.
model = ct.convert(keras_model)

# In Python, provide a NumPy array as input for prediction.
import numpy as np
data = np.random.rand(1, 224, 224, 3)

# Make a prediction using Core ML.
out_dict = model.predict({input_name: data})

# Save to disk.
model.save("MobileNetV2.mlmodel")
