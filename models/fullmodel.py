from munch import Munch

import torch
from models.fullmodel import 
from nn import 
from proto import 
from rr import RidgeRegressor


class FullModel(nn.Module):
    def __init__(args: Munch, self) -> None:
        """Attention Generator + Classifier

            args (Munch): arguments
        """
        if args.classifier == 'rr':
            self.top = RidgeRegressor(args)
        elif args.classifier == 'proto':
            self.top = 
        elif args.classifier == 'nn':
            self.top = 
        
        
    def predict(self):
        pass