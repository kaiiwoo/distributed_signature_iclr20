from munch import Munch

from .rr import RidgeRegressor


class FullModel(nn.Module):
    def __init__(args: Munch, vocab) -> None:
        """Attention Generator + Classifier

            args (Munch): arguments
        """
        if args.classifier == 'rr':
            self.top = RidgeRegressor(args)
        elif args.classifier == 'proto':
            # self.top = 
            pass
        elif args.classifier == 'nn':
            # self.top = 
            pass
        
        
    def predict(self):
        pass