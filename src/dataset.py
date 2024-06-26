import torch
from torch.utils.data import Dataset

import pandas as pd

class CICIDSDataset(Dataset):
  def __init__(self, features_file, target_file, transform=None, target_transform=None):
    """
    Args:
        features_file (string): Path to the csv file with features.
        target_file (string): Path to the csv file with labels.
        transform (callable, optional): Optional transform to be applied on features.
        target_transform (callable, optional): Optional transform to be applied on labels.
    """
    self.features = pd.read_pickle(features_file)
    self.labels = pd.read_pickle(target_file)
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    feature = self.features.iloc[idx, :]
    label = self.labels.iloc[idx]
    if self.transform:
      feature = self.transform(feature.values, dtype=torch.float32)
    if self.target_transform:
      label = self.target_transform(label.values, dtype=torch.int64)
    return feature, label

def get_dataset():
  train_data = CICIDSDataset(
      features_file=f"../data/processed/train/train_features.pkl",
      target_file=f"../data/processed/train/train_labels.pkl",
      transform=torch.tensor,
      target_transform=torch.tensor
  )

  val_data = CICIDSDataset(
      features_file=f"../data/processed/val/val_features.pkl",
      target_file=f"../data/processed/val/val_labels.pkl",
      transform=torch.tensor,
      target_transform=torch.tensor
  )

  test_data = CICIDSDataset(
      features_file=f"../data/processed/test/test_features.pkl",
      target_file=f"../data/processed/test/test_labels.pkl",
      transform=torch.tensor,
      target_transform=torch.tensor
  )

  return train_data, val_data, test_data

def load_data(batch_size: int):
    """Load training, validation and test set."""

    # Get the datasets
    train_data, val_data, test_data = get_dataset()

    # Create the dataloaders - for training, validation and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader


def load_data_split(split: int, batch_size: int):
  train_data = CICIDSDataset(
      features_file=f"data/process_split/split{split}/x{split}.pkl",
      target_file=f"data/process_split/split{split}/y{split}.pkl",
      transform=torch.tensor,
      target_transform=torch.tensor
  )

  test_data = CICIDSDataset(
      features_file=f"data/process_split/split4/x4.pkl",
      target_file=f"data/process_split/split4/y4.pkl",
      transform=torch.tensor,
      target_transform=torch.tensor
  )

    # Create the dataloaders - for training, validation and testing
  train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )

  test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )

  return train_loader, test_loader