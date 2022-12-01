
import torch.nn as nn

NONLINS = {
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

def get_model(flags):
    models = {
        "lin": Linear,
        "linb": LinearBig,
        "mlp": MLP,
    }
    if flags.model in models:
        return models[flags.model](flags)
    else:
        raise RuntimeError(f"Did not recognize model {flags.model}")


class Linear(nn.Module):
    # super simple, minimal linear model
    def __init__(self, flags):
        super().__init__()
        self.lin = nn.Linear(flags.num_feats, 1)
    
    def forward(self, x):
        return self.lin(x)


class LinearBig(nn.Module):
    # this is still linear but has extra weights
    def __init__(self, flags):
        super().__init__()
        self.lin1 = nn.Linear(flags.num_feats, flags.hdim)

        self.layers = []
        for _ in range(flags.num_layers - 1):
            self.layers.append(nn.Linear(flags.hdim, flags.hdim))
            if flags.dropout > 0:
                self.layers.append(nn.Dropout(p=flags.dropout))
            self.layers.append(NONLINS[flags.nonlin]())
        self.layers = nn.Sequential(*self.layers)

        self.lin2 = nn.Linear(flags.hdim, 1)
        self.dropout = nn.Dropout(p=flags.dropout)
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.dropout(x)
        x = self.layers(x)
        x = self.lin2(x)
        return x


class MLP(nn.Module):
    # Multi-Layer Perceptron: linear-nonlin-linear
    def __init__(self, flags):
        super().__init__()
        self.lin_in = nn.Linear(flags.num_feats, flags.hdim)
        self.nl1 = NONLINS[flags.nonlin]()
        
        self.layers = []
        for _ in range(flags.num_layers - 1):
            self.layers.append(nn.Linear(flags.hdim, flags.hdim))
            if flags.dropout > 0:
                self.layers.append(nn.Dropout(p=flags.dropout))
            self.layers.append(NONLINS[flags.nonlin]())
        self.layers = nn.Sequential(*self.layers)

        self.lin_out = nn.Linear(flags.hdim, 1)
        self.dropout = nn.Dropout(p=flags.dropout)
        

    def forward(self, x):
        x = self.lin_in(x)
        x = self.dropout(x)
        x = self.nl1(x)
        x = self.layers(x)
        x = self.lin_out(x)
        return x
