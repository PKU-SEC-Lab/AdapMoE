import argparse
import json
import os
import time

import pandas as pd
import math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
sys.path.append('..')
from modified_mixtral import MixtralForCausalLM


TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions'
        ]

choices = ["A", "B", "C", "D"]

def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))
    return total_acc/total_num


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def load(size):
    n_gpus = torch.cuda.device_count()

    model_id = f"/opt/pretrained_models/Mixtral-8x{size}B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
    )
    model = MixtralForCausalLM.from_pretrained(model_id,quantization_config=quantization_config,device_map="balanced")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    model.eval()

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts):
    batch_size = 24
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers

def reset(model):
    for layer in model.model.layers:
        layer.block_sparse_moe.threshold = 0
        layer.block_sparse_moe.top1_ratio.reset()

def main():
    
    run_results = {}
    import time
    # file name consits of day, hour and minutes
    output_filename = 'run_results_%s.json' % args.threshold
    
    model, tokenizer = load(args.size)
    start_time = time.time()
    if args.size==22:
        weight = [61.74471229314804, 30.827676877379417, 22.530393674969673, 18.11552420258522, 15.121695585548878, 12.262344360351562, 7.301234174519777, 4.385834559798241, 3.1670755706727505, 2.321721287444234, 1.8764339620247483, 1.5008790651336312, 1.2718226062133908, 1.1100198607891798, 0.9757113293744624, 0.8827666169963777, 0.8224676712416112, 0.7467870600521564, 0.6462281453423202, 0.6080560269765556, 0.5719724576920271, 0.501688162330538, 0.48486964078620076, 0.4674123483709991, 0.4506474651861936, 0.4441085911821574, 0.4265871539246291, 0.4034294106531888, 0.3986935189459473, 0.3823837323579937, 0.3598774201236665, 0.34665799466893077, 0.33294674358330667, 0.32827918766997755, 0.31803123420104384, 0.3047495265491307, 0.2975378301925957, 0.28540310449898243, 0.27299695648252964, 0.25855901185423136, 0.24315925838891417, 0.2291009877808392, 0.21316581114660949, 0.19911023264285177, 0.1838320167735219, 0.17445572302676737, 0.1652784994803369, 0.15212985454127192, 0.14296766312327236, 0.12592993152793497, 0.10483352525625378, 0.085503859736491, 0.07232583448057994, 0.06392712384695187, 0.05674008571077138, 0.08505595178576186]
    else:
        weight = [46.69189453125, 17.303466796875, 13.0157470703125, 7.640838623046875, 4.169464111328125, 2.2296905517578125, 1.2559890747070312, 0.8444786071777344, 0.6837844848632812, 0.5602836608886719, 0.5125999450683594, 0.4780292510986328, 0.44536590576171875, 0.4355907440185547, 0.38361549377441406, 0.30994415283203125, 0.23305416107177734, 0.1760721206665039, 0.13840198516845703, 0.1137852668762207, 0.10472536087036133, 0.09542703628540039, 0.08624792098999023, 0.07712841033935547, 0.06937980651855469, 0.06109476089477539, 0.0502467155456543, 0.042557716369628906, 0.03349781036376953, 0.025272369384765625, 0.020682811737060547, 0.02294778823852539]


    if len(args.threshold) == 0:
        args.threshold = [0]
    for threshold in args.threshold:
        reset(model)
        for idx, layer in enumerate(model.model.layers):
            if args.hessian:
                # sqrt(threshold/weight[idx])
                layer.block_sparse_moe.threshold = math.sqrt(threshold/weight[idx])
            else:
                layer.block_sparse_moe.threshold = threshold
        for task in TASKS:
            print('Testing %s ...' % task)
            records = []
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
            for i in range(test_df.shape[0]):
                # get prompt and make sure it fits
                k = args.ntrain
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, task, k)
                prompt = train_prompt + prompt_end
                while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                    prompt_split = prompt.split("\n\n")
                    prompt_split.pop(1)
                    prompt = '\n\n'.join(prompt_split)
                label = test_df.iloc[i, test_df.shape[1]-1]
                records.append({'prompt':prompt, 'answer':label})

            pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records])
            gold_answers = [record['answer'] for record in records]
            run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
        with open(output_filename, 'w') as f:
            json.dump(run_results, f, ensure_ascii=False, indent=2)
        
        acc = compute_metric(output_filename)
        end_time = time.time()
        print("total run time %.2f" % (end_time - start_time))

        top1_ratios = [layer.block_sparse_moe.top1_ratio for layer in model.model.layers]
        avg_top1_ratios = sum([ratio.num for ratio in top1_ratios])/len(top1_ratios)
        result_filename = 'naive_result' if not args.hessian else 'hessian_result'
        with open('mmlu_'+result_filename+f'_{args.size}b', 'a') as f:
            f.write(f'Threshold: {threshold}\n')
            f.write(f'Accuracy: {acc}\nSingle Expert Selection Ratio: {avg_top1_ratios}\nSingle Expert Selection Ratios of each layer{top1_ratios}\n\n')
   


    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='MMLU/data')
    parser.add_argument('--ntrain', type=int, default=5)
    # threshold: list of float
    parser.add_argument('--threshold', type=float, default=[],nargs='+')
    parser.add_argument('--hessian', action='store_true')
    parser.add_argument('--output', type=str, default='run_results.json')
    parser.add_argument('--size', type=int, default=7)
    args = parser.parse_args()
    
    main()

