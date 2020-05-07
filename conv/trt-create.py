import tensorrt as trt
import mxnet as mx
import common
import numpy as np
import time


params_path="model/mxnet/conv-0003.params"
input_shape=(1,256, 256, 256)
output_shape=(1, 128, 256, 256)

def get_params():
    params=mx.nd.load(params_path)
    return params


def build_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 28

    network = builder.create_network()
    data = network.add_input("data", trt.DataType.FLOAT, input_shape)

    params=get_params()
    conv_0_weight=params["arg:conv_0_weight"]
    conv0_w=conv_0_weight.asnumpy()
    conv_0_bias=params["arg:conv_0_bias"]
    conv0_b=conv_0_bias.asnumpy()

    conv0=network.add_convolution_nd(input=data, num_output_maps=128, kernel_shape=(3,3), kernel=conv0_w,bias=conv0_b)
    conv0.padding=(1,1)
    conv0.stride=(1,1)

    network.mark_output(tensor=conv0.get_output(0))

    engine=builder.build_cuda_engine(network)

    return engine

if __name__ == "__main__":
    engine=build_engine()
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

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
    print(length, input_shape)
    data=np.zeros(length,dtype=np.float32)
    data[:]=1.0
    inputs[0].host=data.reshape(input_shape)
    outputs[0].host=np.zeros(output_shape, dtype=np.float32)

    plan=engine.serialize()
    with open("model/trt/conv.trt", "wb") as f:
        f.write(plan)
    f.close()
