import torch

def calculate_confusion_matrix(y_true: torch.tensor, y_pred: torch.tensor):
  # Assuming y_true and y_pred are binary tensors
  assert y_true.shape == y_pred.shape, "Input tensors must have the same shape"
  assert torch.logical_or((y_true == 0), (y_true == 1)).all(), "Input tensors must be Binary"
  assert torch.logical_or((y_pred == 0), (y_pred == 1)).all(), "Input tensors must be Binary"
  
  # Calculate confusion matrix
  tp = torch.sum((y_true == 1) & (y_pred == 1))
  tn = torch.sum((y_true == 0) & (y_pred == 0))
  fp = torch.sum((y_true == 0) & (y_pred == 1))
  fn = torch.sum((y_true == 1) & (y_pred == 0))
  
  return tp.item(), tn.item(), fp.item(), fn.item()

def accuracy(tp, tn, fp, fn):
  if tp + tn + fp + fn == 0:
    return 0
  return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
  if tp + fp == 0:
    return 0
  return tp / (tp + fp)

def recall(tp, fn):
  if tp + fn == 0:
    return 0
  return tp / (tp + fn)

def f1_score(tp, fp, fn):
  prec = precision(tp, fp)
  rec = recall(tp, fn)
  if prec + rec == 0:
    return 0
  return 2 * (prec * rec) / (prec + rec)