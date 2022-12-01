
import torch.nn as nn

NONLINS = {
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

def get_model(flags, dictionary=None):
    basic_models = {
        "lin": Linear,
        "linb": LinearBig,
        "mlp": MLP,
    }
    nlp_models = {
        "bow": BagOfWords,
        "rnn": RNN,
    }
    if flags.model in basic_models:
        return basic_models[flags.model](flags)
    elif flags.model in nlp_models:
        return nlp_models[flags.model](flags, dictionary)
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


class BagOfWords(nn.Module):
    # bag of words: average embeddings + linear
    def __init__(self, flags, dict):
        super().__init__()
        self.emb = nn.EmbeddingBag(len(dict), flags.edim)
        flags.num_feats = flags.edim
        self.mlp = MLP(flags)

    def forward(self, x):
        x = self.emb(x.long())
        x = self.mlp(x)
        return x


class RNN(nn.Module):
    # recurrent model: read tokens one at a time
    def __init__(self, flags, dict):
        super().__init__()
        self.emb = nn.Embedding(len(dict), flags.edim)

        if flags.rnn_type == 'gru':
            rnn_type = nn.GRU
        elif flags.rnn_type == 'lstm':
            rnn_type = nn.LSTM
        elif flags.rnn_type == 'rnn':
            rnn_type = nn.RNN
        else:
            raise RuntimeError(f"Did not recognize flags.rnn_type={flags.rnn_type}")
        self.rnn = rnn_type(
            flags.edim, flags.hdim, flags.num_layers,
            dropout=flags.dropout if flags.num_layers > 1 else 0,
            batch_first=True, bidirectional=flags.rnn_bidir)
        
        num_dir = 2 if flags.rnn_bidir and flags.rnn_out == "output" else 1
        self.out = nn.Linear(flags.hdim * num_dir, 1)
        self.dropout = nn.Dropout(flags.dropout)
        self.out_layer = flags.rnn_out
        self.final_dropout = flags.rnn_dropout
    
    def forward(self, x):
        x = self.emb(x.long())
        x = self.dropout(x)
        outputs, hidden = self.rnn(x)
        if self.out_layer == "hidden":
            x = hidden[-1].squeeze(0)
        elif self.out_layer == "output":
            x = outputs[:, -1]
        else:
            raise RuntimeError(f"Did not recognize flags.rnn_out={self.out_layer}")
        if self.final_dropout:
            x = self.dropout(x)
        x = self.out(x)
        return x