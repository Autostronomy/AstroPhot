import torch


def window_or(other_origin, self_end, other_end):

    new_origin = torch.minimum(-0.5 * torch.ones_like(other_origin), other_origin)
    new_end = torch.maximum(self_end, other_end)

    return new_origin, new_end


def window_and(other_origin, self_end, other_end):
    new_origin = torch.maximum(-0.5 * torch.ones_like(other_origin), other_origin)
    new_end = torch.minimum(self_end, other_end)

    return new_origin, new_end
