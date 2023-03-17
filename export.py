import torch
from transformers import AutoTokenizer, AutoModel

def model_export(
    model,
    model_args: tuple,
    output_path: str,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset
):
    from torch.onnx import export
    export(
        model,
        model_args,
        f=output_path,
        input_names=ordered_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )


def test():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float().cpu()
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(8).cuda()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    #response, history = model.chat(tokenizer, "用Python帮我写一段拓扑排序", history=history)
    #print(response)

def to_onnx():
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, resume_download=True).float().cpu()
    model_export(model,
                model_args=(
                    torch.randn(1, 4, 4096),
                    torch.tensor([[[[False, False, False,  True],
                                    [False, False, False,  True],
                                    [False, False, False,  True],
                                    [False, False, False, False]]]]),
                    torch.tensor([[[0, 1, 2, 3], [0, 0, 0, 1]]]),
                    torch.zeros(28, 2, 0, 1, 32, 128)
                ),
                output_path= "model.onnx",
                ordered_input_names=["inputs_embeds", "attention_mask", "position_ids", "past_key_values"],
                output_names=["hidden_states", "presents"],
                dynamic_axes={},
                opset= 14)
    

if __name__ == '__main__':
    # to_onnx()
    test()
