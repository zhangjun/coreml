import coremltools
import sys

if len(sys.argv) < 3:
	print(sys.argv[0], " onnx_model save_model")
	exit(0)
onnx_model = sys.argv[1]
saved_model = sys.argv[2]

model = coremltools.converters.onnx._converter.convert(model=onnx_model, minimum_ios_deployment_target='13')
model.save(saved_model + ".mlmodel")
