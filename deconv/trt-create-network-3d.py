import tensorrt as trt
import mxnet as mx
import common
import numpy as np
import time


params_path="model/mxnet/deconv-0000.params"
input_shape=( 256, 1, 256, 256)
output_shape=(256, 1, 512, 512)


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
    deconv_0_weight=params["arg:dconv_0_weight"]
    deconv0_w=deconv_0_weight.asnumpy()
    deconv0_w_t=deconv0_w[:,:,np.newaxis,:,:]

    deconv0=network.add_deconvolution_nd(input=data, num_output_maps=256, kernel_shape=(1, 4, 4), kernel=deconv0_w_t, bias=None)
    deconv0.name="deconv_0"
    deconv0.stride_nd=(1, 2,2)
    deconv0.padding_nd=(0, 1,1)
    network.mark_output(tensor=deconv0.get_output(0))
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
    data=np.zeros(length,dtype=np.float32)
    data[:]=1.0
    inputs[0].host=data.reshape(input_shape)
    outputs[0].host=np.zeros(output_shape, dtype=np.float32)

    plan=engine.serialize()
    with open("model/trt/deconv_3d.trt", "wb") as f:
        f.write(plan)
    f.close()
