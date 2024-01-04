import os
import torch
import numpy as np
import argparse
from CMMLU.src.mp_utils import choices, format_example, gen_prompt, softmax, run_eval

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pymnn_llm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

def eval(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot):
    choice_ids = [tokenizer(choice)['input_ids'][0] for choice in choices]
    cors = []
    all_conf = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        prompt = gen_prompt(dev_df=dev_df,
                            subject=subject,
                            prompt_end=prompt_end,
                            num_few_shot=num_few_shot,
                            tokenizer=tokenizer,
                            max_length=max_length,
                            cot=cot)
        label = test_df.iloc[i, test_df.shape[1] - 1]

        with torch.no_grad():
            input_ids = tokenizer([prompt], padding=False)['input_ids']
            input_ids = torch.tensor(input_ids, device=model.device)
            logits = model(input_ids)['logits']
            last_token_logits = logits[:, -1, :]
            if last_token_logits.dtype in {torch.bfloat16, torch.float16}:
                last_token_logits = last_token_logits.to(dtype=torch.float32)
            choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
            conf = softmax(choice_logits[0])[choices.index(label)]
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_logits[0])]

        all_preds += pred
        all_conf.append(conf)
        cors.append(pred == label)

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return acc, all_preds, None



def eval_chat(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot):
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        prompt = gen_prompt(dev_df=dev_df,
                            subject=subject,
                            prompt_end=prompt_end,
                            num_few_shot=num_few_shot,
                            tokenizer=tokenizer,
                            max_length=max_length,
                            cot=cot)
        label = test_df.iloc[i, test_df.shape[1] - 1]

        pred, history = model.chat(tokenizer, prompt, history=None)
        # print(f'{i} of {test_df.shape[0]}: prompt: {prompt}, pred: {pred}, label: {label}')
        if pred and pred[0] in choices:
            cors.append(pred[0] == label)
        all_preds.append(pred.replace("\n", ""))

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print("{} results, {} inappropriate formated answers.".format(len(cors), len(all_preds)-len(cors)))
    return acc, all_preds, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--token_path", type=str, default="")
    parser.add_argument("--lora_weights", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./CMMLU/data")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--load_in_8bit", action='store_true')
    parser.add_argument("--cot", action='store_true')
    args = parser.parse_args()

    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained(args.token_path, trust_remote_code=True)
    
    model = pymnn_llm.from_model(args.model_path)
    
    '''
    model = AutoModelForCausalLM.from_pretrained(args.token_path,
                                      trust_remote_code=True,
                                      device_map="auto"
                                    )
    model.generation_config = GenerationConfig.from_pretrained(args.token_path, trust_remote_code=True)
    '''


    if "chat" in args.token_path.lower():
        run_eval(model, tokenizer, eval_chat, args)
    else:
        run_eval(model, tokenizer, eval, args)