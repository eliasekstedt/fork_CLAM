
def test0():
    import torch
    lf_counts = torch.tensor([221, 105], dtype=torch.float)
    lf_weights = 1.0 / lf_counts
    lf_weights = lf_weights / lf_weights.sum()  # optional normalization
    print(lf_weights)


test0()