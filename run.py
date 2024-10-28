import numpy

import sys

import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig

from transformers import AutoConfig, AutoTokenizer

from src.build_model import OffloadConfig, QuantConfig, build_model
from src.dp import get_cache_size

from transformers import TextStreamer
import time
import argparse
import math

def main():
    path = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
    model_name = path
    quantized_model_name = path
    state_path = path

    config = AutoConfig.from_pretrained(quantized_model_name)

    device = torch.device("cuda:0")

    main_size = args.size
    cache_strategy = get_cache_size(main_size,args.adapgate)
    print(cache_strategy)

    num_experts = config.num_local_experts

    offload_config = OffloadConfig(
        main_size=main_size,
        cache_strategy=cache_strategy,
        offload_size=config.num_hidden_layers * 8,
        buffer_size=6,
    )


    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256

    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )
    quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)


    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )
    if args.adapgate:
        weight = [46.69189453125, 17.303466796875, 13.0157470703125, 7.640838623046875, 4.169464111328125, 2.2296905517578125, 1.2559890747070312, 0.8444786071777344, 0.6837844848632812, 0.5602836608886719, 0.5125999450683594, 0.4780292510986328, 0.44536590576171875, 0.4355907440185547, 0.38361549377441406, 0.30994415283203125, 0.23305416107177734, 0.1760721206665039, 0.13840198516845703, 0.1137852668762207, 0.10472536087036133, 0.09542703628540039, 0.08624792098999023, 0.07712841033935547, 0.06937980651855469, 0.06109476089477539, 0.0502467155456543, 0.042557716369628906, 0.03349781036376953, 0.025272369384765625, 0.020682811737060547, 0.02294778823852539]
        for idx, layer in enumerate(model.model.layers):
            layer.block_sparse_moe.threshold = math.sqrt(0.005/weight[idx])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    past_key_values = None
    sequence = None

    seq_len = 0
    total_time = 0
    total_tokens = 0
    while True:
        print("User: ", end="")
        user_input = input()
        print("\n")

        user_entry = dict(role="user", content=user_input)
        input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

        if past_key_values is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
            attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

        print("Mixtral: ", end="")
        start_time = time.time()
        result = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            streamer=streamer,
            do_sample=True,
            top_k=1,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        end_time = time.time()
        print("\n")

        sequence = result["sequences"]
        past_key_values = result["past_key_values"]

        total_time += end_time - start_time
        total_tokens += sequence.size(1)

        # Calculate average time per token
        avg_time_per_token = total_time / 128
        print(f"Average time per token: {avg_time_per_token} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapgate',action='store_true')
    parser.add_argument('--size',type=int,default=64)
    args = parser.parse_args()
    main()


