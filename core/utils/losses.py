import torch.nn as nn
import torch
from torch import autograd


def boundary_loss(points):
    zeros = - torch.zeros_like(points)
    ones = torch.ones_like(points)

    loss1 = torch.abs(torch.min(points, zeros))

    loss2 = torch.abs(torch.max(zeros, points-ones))

    return torch.mean(loss1 + loss2)


def compute_grad2(d_output, real_input):
    batch_size = real_input.size(0)
    grad_dout = autograd.grad(
        outputs=d_output.sum(), inputs=real_input,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == real_input.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def limb_loss(points):
    """
    compare vector lengths of knee-ankle and shoulder-elbow across
    timepoints
    """
    indices1 = torch.tensor([4, 5, 6, 7])
    indices2 = torch.tensor([8, 9, 10, 11])

    diff = points[:, indices1] - points[:, indices2]

    return torch.norm(diff, dim=-1).mean()



