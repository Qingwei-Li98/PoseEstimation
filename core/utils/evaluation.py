import numpy as np
import torch


def EvalMetrics(preds, gt):
    # Euclidean Distance Between Points
    euclideanDist = EuclideanDistance(preds, gt)  
    batchJointError = torch.sum(euclideanDist,0)
    return batchJointError, euclideanDist

def EuclideanDistance(vector1, vector2):
    # Accepts tensor of size [batchSize, outputSize, 3]
    euclideanDist = torch.square(vector1 - vector2)
    euclideanDist = torch.sum(euclideanDist, 2)
    euclideanDist = torch.sqrt(euclideanDist)
    return euclideanDist

def PCKh(euclideanDist, labels, thresholds, factor):
    thresholds = thresholds.unsqueeze(1) * factor
    maskedDistances = euclideanDist[thresholds[:, 0] > 0]
    thresholds = thresholds[thresholds[:, 0] > 0]

    thresholds = thresholds.expand(-1, maskedDistances.size()[1])
    # Mask threshold onto distances
    ones = torch.ones(maskedDistances.size())
    zeros = torch.zeros(maskedDistances.size())
    # Per frame per joint
    PCKhValues = torch.where(maskedDistances <= thresholds, ones, zeros)
    PCKhValues = torch.where(maskedDistances == 0, zeros, PCKhValues)
    # PCKhValues = torch.prod(PCKhValues, 1)
    return torch.sum(PCKhValues, 0)