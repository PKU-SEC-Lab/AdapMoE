import lm_eval 
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig
)
from lm_eval.models.huggingface import HFLM
from modified_mixtral import MixtralForCausalLM

import torch
import argparse
import math

def reset(model):
    for layer in model.model.layers:
        layer.block_sparse_moe.threshold = 0
        layer.block_sparse_moe.top1_ratio.reset()

def main():
    model_id = f"/opt/pretrained_models/Mixtral-8x{args.size}B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
    )
    model = MixtralForCausalLM.from_pretrained(model_id,quantization_config=quantization_config,device_map='auto')
    config = AutoConfig.from_pretrained(
        model_id,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    model.eval()

    # instantiate an LM subclass that takes your initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`
    lm_obj = HFLM(pretrained = model, 
                tokenizer = tokenizer,
                batch_size=args.batch_size)
    
    # layer sensitivity factors
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
                layer.block_sparse_moe.threshold = math.sqrt(threshold/weight[idx])
            else:
                layer.block_sparse_moe.threshold = threshold

        # indexes all tasks from the `lm_eval/tasks` subdirectory.
        # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
        # to include a set of tasks in a separate directory.
        task_manager = lm_eval.tasks.TaskManager()

        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager is the it is set to None here.
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_obj,
            tasks=[args.task],
            num_fewshot=0,
            task_manager=task_manager,
            batch_size=args.batch_size,
            model_args="parallelize=True",
        )
        acc = results['results'][args.task]['acc,none']
        
        
        top1_ratios = [layer.block_sparse_moe.top1_ratio for layer in model.model.layers]
        avg_top1_ratios = sum([ratio.num for ratio in top1_ratios])/len(top1_ratios)
        result_filename = 'naive_result' if not args.hessian else 'hessian_result'
        with open('mmlu_'+result_filename+f'_{args.size}b', 'a') as f:
            f.write(f'Threshold: {threshold}\n')
            f.write(f'Accuracy: {acc}\nSingle Expert Selection Ratio: {avg_top1_ratios}\nSingle Expert Selection Ratios of each layer{top1_ratios}\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hessian', action='store_true')
    parser.add_argument('--threshold', type=float, default=[],nargs='+')
    parser.add_argument('--size', type=int, default=7)
    parser.add_argument('--task', type=str, default="mmlu")
    args = parser.parse_args()
    if args.size==22:
        args.batch_size = 8
    else:
        args.batch_size = 32
    main()
