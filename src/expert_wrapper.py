import typing as tp

import torch
from torch import nn

from .utils import nested_flatten, nested_pack
from .linear_wrapper import MixtralLinearWrapper
from transformers.activations import ACT2FN


class MixtralExpertWrapper(nn.Module):
    def __init__(
        self,
        expert_module: tp.Any,
        device: torch.device,
    ):
        super().__init__()
        self.w1 = MixtralLinearWrapper(expert_module.w1, device)
        self.w2 = MixtralLinearWrapper(expert_module.w2, device)
        self.w3 = MixtralLinearWrapper(expert_module.w3, device)
        self.act_fn = expert_module.act_fn
        self.w1_event = torch.cuda.Event()
        self.w2_event = torch.cuda.Event()
        self.w3_event = torch.cuda.Event()
        self.load = False
        self.free = True

    def forward(self, hidden_states):
        if self.load:
            self.w1_event.wait()
            current_hidden_states = self.act_fn(self.w1(hidden_states))
            self.w3_event.wait()
            current_hidden_states = current_hidden_states * self.w3(hidden_states)
            self.w2_event.wait()
            current_hidden_states = self.w2(current_hidden_states)
        else:
            current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
            current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states