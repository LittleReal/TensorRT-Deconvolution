import time
import mxnet as mx
import numpy as np
from mxnet.io import DataBatch, DataDesc

input_shape=(1,256, 256, 256)
output_shape=(1, 128, 256, 256)

def create_batch():
    length=input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]
    frame=np.zeros(length,dtype=np.float32)
    frame[:]=1.0
    print(frame[:10])
    batch_frame=[mx.nd.array(frame.reshape(input_shape))]
    batch_shape = [DataDesc('data', batch_frame[0].shape)]
    batch = DataBatch(data=batch_frame, provide_data=batch_shape)
    return batch


def main():
    ctx = mx.gpu(0)
    load_symbol, sym_args, sym_auxs = mx.model.load_checkpoint("model/mxnet/conv", 3)
    mod = mx.mod.Module(load_symbol, label_names=None, context=ctx)
    mod.bind(data_shapes=[('data', input_shape)], for_training=False)
    mod.set_params(sym_args, sym_auxs)
    batch=create_batch()
    mod.forward(batch)
    out=mod.get_outputs()[0].asnumpy()
    print(out.shape)
    print(out[0][0][0][:10])
    print("starting...")
    starttime = time.time()
    for i in range(1000):
        mod.forward(batch)
        out=mod.get_outputs()[0].asnumpy()
    endtime = time.time()
    print(out.shape)
    print(out[0][0][0][:10])
    print (endtime - starttime)

if __name__ == "__main__":
    main()