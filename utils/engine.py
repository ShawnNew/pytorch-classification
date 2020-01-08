import onnx
import sys
import os
import tensorrt as trt
try:
    sys.path.append(os.path.abspath(os.environ['ONNX2TRT']))
    from onnx_tensorrt.tensorrt_engine import Engine as TRTEngine
    import onnx_tensorrt.backend as backend
except ImportError:
    raise ImportError("Please import onnx-tensorrt directory to run this example.")

class Engine:
    def __init__(self, model_path):
        if model_path.rsplit(".", maxsplit=1)[-1] == "onnx":
            model = onnx.load(model_path)
            self.engine = backend.prepare(model,device="CUDA:0", max_workspace_size=self._gb(8), serialize_engine=True)
        elif model_path.rsplit(".", maxsplit=1)[-1] == "trt":
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.VERBOSE))
            with open(model_path, "rb") as f:
                engine = self.runtime.deserialize_cuda_engine(f.read())
            self.engine = TRTEngine(engine)
        else:
            raise RuntimeError("Model path {} is not correct.".format(model_path))
        print("TensorRT engine is built successfully.")

    def run(self, images):
        return self.engine.run([images])

    def serialize(self, path):
        serialized_engine = self.engine.engine.engine.serialize()
        with open(path, "wb") as f:
            f.write(serialized_engine)
        print("Serialized TensorRT engine is saved to {}.".format(path))

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @staticmethod
    def _gb(n):
        assert isinstance(n, int)
        return n << 30