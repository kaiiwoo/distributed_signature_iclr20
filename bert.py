from turtle import forward
import torch
import torch.nn as nn

from transformers import BertModel


# final hid states or [cls] token
class BertRepresentation(nn.Module):
    def __init__(self, args) -> None:
        super(BertRepresentation, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.return_seq = args.return_seq
        
    def forward(self, x):
        x = self.model(x)
        
        if self.return_seq:
            return x.last_hidden_states
        else:
            return x.last_hidden_states[:, 0, :]
