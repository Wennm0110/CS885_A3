import torch

class TorchHelper:
    
    def __init__(self):
        if torch.cuda.is_available():
            print("run by cuda")
            self.device = torch.device("cuda")
        else:
            print("run by cpu")
            self.device = torch.device("cpu")
    
    def f(self, x):
        return torch.tensor(x).float().to(self.device)
    
    def i(self, x):
        return torch.tensor(x).int().to(self.device)
    
    def l(self, x):
        return torch.tensor(x).long().to(self.device)