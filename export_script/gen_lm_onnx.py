import numpy as np
import onnx

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor

def main() -> None:
    lm = np.fromfile('slim_lm.bin', dtype=np.float32, count=-1, offset=0)
    ic = 4096
    oc = 130528
    model_input_name = "input"
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           [1, ic, 1, 1])
    model_output_name = "output"
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.INT64,
                                           [1, 1, 1, 1])
    conv1_output_node_name = "conv"
    conv1_in_channels = ic
    conv1_out_channels = oc
    conv1_kernel_shape = (1, 1)
    conv1_pads = (0, 0, 0, 0)
    conv1_W = lm.reshape((conv1_out_channels, conv1_in_channels,
                             *conv1_kernel_shape)).astype(np.float32)
    conv1_W_initializer_tensor_name = "conv_w"
    conv1_W_initializer_tensor = create_initializer_tensor(
        name=conv1_W_initializer_tensor_name,
        tensor_array=conv1_W,
        data_type=onnx.TensorProto.FLOAT)
    conv1_node = onnx.helper.make_node(
        name="Conv",  # Name is optional.
        op_type="Conv",
        inputs=[
            model_input_name, conv1_W_initializer_tensor_name,
        ],
        outputs=[conv1_output_node_name],
        kernel_shape=conv1_kernel_shape,
        pads=conv1_pads,
    )
    relu1_output_node_name = "output"
    relu1_node = onnx.helper.make_node(
        name="ArgMax",  # Name is optional.
        op_type="ArgMax",
        inputs=[conv1_output_node_name],
        outputs=[relu1_output_node_name],
        axis=1,
    )
    graph_def = onnx.helper.make_graph(
        nodes=[conv1_node, relu1_node],
        name="lm",
        inputs=[X],  # Graph input
        outputs=[Y],  # Graph output
        initializer=[
            conv1_W_initializer_tensor, # conv1_B_initializer_tensor
        ],
    )
    model_def = onnx.helper.make_model(graph_def, producer_name="mnn-lm")
    model_def.opset_import[0].version = 13
    model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.save(model_def, "lm.onnx")
if __name__ == "__main__":
    main()
