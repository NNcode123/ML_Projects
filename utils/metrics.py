import torch

def calc_accuracy(y_pred_matrix: torch.Tensor, actual_y: torch.Tensor) -> int:
    y_pred_class = y_pred_matrix.argmax(axis = 1) 

    return (y_pred_class == actual_y).sum().item()/actual_y.numel()




def dice_score(logits:torch.Tensor, ground_truth: torch.Tensor, seg_class: int )->float:

    predicted = logits.argmax(dim = 0)+1 if (logits.ndim == 3) else logits.argmax(dim = 1)+1

    #print(f"pred_shape: {predicted.shape} truth_shape: {ground_truth.shape}")

    pred_seg_class = predicted == seg_class

    ground_seg_class = ground_truth == seg_class


    intersection = pred_seg_class & ground_seg_class

    return (2*intersection.sum().item())/(pred_seg_class.sum().item()+ground_seg_class.sum().item())
    

    

def iou_score(logits:torch.Tensor, ground_truth: torch.Tensor, seg_class: int)->float:


    predicted = logits.argmax(dim = 0)+1 if (logits.ndim == 3) else logits.argmax(dim = 1)+1

    pred_seg_class = predicted == seg_class

    ground_seg_class = ground_truth == seg_class

    intersection = pred_seg_class & ground_seg_class

    union = pred_seg_class | ground_seg_class

    return (intersection.sum().item())/(union.sum().item())



def miou_score(logits:torch.Tensor, ground_truth: torch.Tensor)->float:
    seg_classes = ground_truth.unique()
    return sum(iou_score(logits,ground_truth, i) for i in seg_classes)/seg_classes.numel()
    

def mdice_score(logits:torch.Tensor, ground_truth: torch.Tensor)->float:
    seg_classes = ground_truth.unique()
    return sum(dice_score(logits,ground_truth, i) for i in seg_classes)/seg_classes.numel()
    


