import torch.nn as nn


class MLP(nn.Module):
    """
    Parameters :
        in_features (int)   : input size of the MLP
        out_features (int)  : output size of the MLP
        loop (int)      : number of repetition of Linear-Norm-ReLU
    """

    def __init__(self, in_features=512, out_features=512, depth=3):
        super().__init__()
        modules = []
        in_feats = in_features
        for i in range(depth):
            modules.append(nn.Linear(in_feats, out_features))
            in_feats = out_features
            modules.append(nn.LeakyReLU())
            modules.append(nn.modules.normalization.LayerNorm(out_features))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
