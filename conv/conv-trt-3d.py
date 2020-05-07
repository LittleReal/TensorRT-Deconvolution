import tensorrt as trt
import time
import numpy as np
import common

input_shape=(256,1, 256, 256)
output_shape=(128, 1, 256, 256)

def main():
    model_path="model/trt/conv_3d.trt"

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime=trt.Runtime(TRT_LOGGER)
    f=open(model_path,"rb")
    engine = runtime.deserialize_cuda_engine(f.read())
    context=engine.create_execution_context()
    f.close()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            print("input_size: ", size, "dtype: ", dtype)
        else:
            print("output_size: ", size, "dtype: ", dtype)

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    length=input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]
    data=np.zeros(length,dtype=np.float32)
    data[:]=1.0
    inputs[0].host=data.reshape(input_shape)
    print(inputs[0].host[0][0][0][:10])

    outputs[0].host=np.zeros(output_shape, dtype=np.float32)
    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print(trt_outputs[0][0][0][0][:10])

    print("starting...")
    starttime = time.time()
    for i in range(1000):
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    endtime = time.time()
    print (endtime - starttime)

    print(trt_outputs[0][0][0][0][:10])

if __name__ == "__main__":
    main()
