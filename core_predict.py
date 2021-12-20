# Use PIL to load and resize the image to expected size
import coremltools as ct
from PIL import Image
example_image = Image.open("daisy.jpg").resize((224, 224))

# Load the saved model
model = ct.models.MLModel("vgg.mlmodel")

# Make a prediction using Core ML
import numpy as np
import time
data = np.random.rand(1, 3, 224, 224)

begin = time.time()
out_dict = model.predict({"data": data})
end = time.time()

print("time cost: ", end - begin)
# Print out top-1 prediction
#print(out_dict["vgg0_dense2_fwd"])
