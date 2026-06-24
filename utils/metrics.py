import torch

def calc_accuracy(y_pred_matrix: torch.Tensor, actual_y: torch.Tensor) -> int:
    y_pred_class = y_pred_matrix.argmax(axis = 1) 

    return (y_pred_class == actual_y).sum().item()/actual_y.numel()