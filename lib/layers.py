import torch
import torch.nn as nn
import torch.nn.functional as F

from globals import CONFIG

class ReLUx(nn.Module):

    def __init__(self, inplace: bool = False):
        super(ReLUx, self).__init__()
        self.relux_operation = CONFIG.experiment_args['relux_operation']
        self.register_buffer('clip_value', torch.tensor([0.0]))

    def __repr__(self):
        if self.relux_operation is None:
            return 'ReLU()'
        elif CONFIG.experiment_args['relux_operation'] == 'relu6':
            return 'ReLU6()'
        elif CONFIG.experiment_args['relux_operation'] == 'max':
            return 'ReLUMax()'
    
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        clip_value_key = prefix + 'clip_value'
        if clip_value_key in state_dict:
            self.clip_value.copy_(state_dict[clip_value_key])
    
    def forward(self, x):
        if self.relux_operation is None: # standard ReLU
            return F.relu(x)
        
        if self.clip_value.device != x.device:
            self.clip_value = self.clip_value.to(x.device)
        
        if self.training:
            with torch.no_grad():
                if self.relux_operation == 'relu6' and self.clip_value.item() != 6.0:
                    self.clip_value.fill_(6.0)

                elif self.relux_operation == 'max': # ReLUMax
                    self.clip_value.fill_(max(self.clip_value.item(), x.max().item()))
        else:
            if CONFIG.experiment_args['relux_operation'] == 'max':
                x[x > self.clip_value.item()] = 0.0
                return F.relu(x)

        x = F.relu(x)
        return torch.clip(x, max=self.clip_value.item())
