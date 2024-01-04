
import os
import glob
import argparse

import MNN.nn as nn
import MNN.expr as F
import MNN.numpy as np
import numpy

from transformers import AutoTokenizer

class LLM:
    def __init__(self, model_path):
        self.max_length = 2048
        self.load(model_path)

    def load_module(self, path, name, inputs=[], outputs=[]):
        return nn.load_module_from_file(os.path.join(path, name), inputs, outputs,
                                        precision_mode = F.PrecisionMode.Low,
                                        memory_mode = F.MemoryMode.Low,
                                        backend = F.Backend.CPU,
                                        rearrange = True,
                                        shape_mutable = True
                                        )

    def load(self, model_path):
        # load split
        self.block_nums = len(glob.glob(os.path.join(model_path, 'block_*.mnn')))
        self.lm = self.load_module(model_path, 'lm.mnn')
        self.embed = self.load_module(model_path, 'embedding.mnn')
        self.blocks = [None for i in range(self.block_nums)]
        for i in range(self.block_nums):
            self.blocks[i] = self.load_module(model_path, f'block_{i}.mnn',
                                              ["inputs_embeds", "attention_mask", "position_ids", "past_key_values"],
                                              ["hidden_states", "presents"])
    def get_attention_mask(self) -> F.Var:
        raise NotImplementedError

    def get_position_ids(self) -> F.Var:
        raise NotImplementedError
    
    def stop_id(self):
        return self.tokenizer.im_end_id

    def build_prompt(self, query):
        if hasattr(self.tokenizer, 'build_prompt'):
            prompt = self.tokenizer.build_prompt(query)
        else:
            prompt = query
        return prompt

    def str_to_ids(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        input_ids = np.array(input_ids.tolist())
        return input_ids

    def id_to_str(self, token_id):
        word = self.tokenizer._convert_id_to_token(int(token_id))
        word = self.tokenizer.convert_tokens_to_string([word])
        return word

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        hidden_states = self.embed(input_ids)
        presents = []
        for i in range(self.block_nums):
            hidden_states, kv = self.blocks[i]([hidden_states, attention_mask, position_ids, past_key_values[i]])
            presents.append(kv)
        token_id = self.lm(hidden_states)
        token_id = np.asscalar(token_id)
        self.seq_len += 1
        self.token_len += 1
        return token_id, presents

    def response(self, query, stream = False):
        prompt = self.build_prompt(query)
        input_ids = self.str_to_ids(prompt)
        self.seq_len = input_ids.size
        self.context_len = self.seq_len - 2
        self.token_len = 0
        past_key_values = [F.placeholder(self.past_kv_shape, dtype=np.float32) for i in range(self.block_nums)]
        token_id = input_ids
        res = ''
        while self.token_len < self.max_length:
            attention_mask = self.get_attention_mask()
            position_ids = self.get_position_ids()
            token_id, past_key_values = self.forward(token_id, attention_mask, position_ids, past_key_values)
            if token_id == self.stop_id():
                if stream:  print("", end='\n')
                res += '\n'
                break
            word = self.id_to_str(token_id)
            res += word
            if stream: print(word, end="", flush=True)
        return res
    
    def chat(self, tokenizer, query, history = None):
        self.tokenizer = tokenizer
        return self.response(query), None

    def stream_chat(self, tokenizer, query, history = None):
        self.tokenizer = tokenizer
        return self.response(query, True), None
    
    def eval(self):
        pass

class Qwen(LLM):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.context_len = 0
        self.token_len = 0
        self.past_kv_shape = [2, 1, 0, 16, 128]

    def build_prompt(self, query):
        return f'\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'

    def get_attention_mask(self) -> F.Var:
        if self.token_len:
            return np.ones([1, 1, 1, 1])
        return np.array(numpy.tril(numpy.ones([1, 1, self.seq_len, self.seq_len], dtype=numpy.int32)).tolist())

    def get_position_ids(self) -> F.Var:
        if self.token_len:
            return np.array([self.seq_len - 1])
        return np.arange(self.seq_len, dtype=np.int32)

def from_model(model_path):
    return Qwen(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnn-llm', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model_path', type=str, default='./chatglm-6b', required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--token_path', type=str, default='./chatglm-6b', required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    args = parser.parse_args()
    model = from_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.token_path, trust_remote_code=True)
    output = model.chat(tokenizer, '你好')
    print(output)