from torch import argmax

def nll_accuracy(out, yb):
    predictions = argmax(out, dim=1)
    return (predictions == yb).float().mean()