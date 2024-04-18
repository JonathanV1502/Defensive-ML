import sys
import setproctitle
import argparse

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from Models import MLP
from dataset import load_data_split
from Trainer import MLPTrainer


if __name__ == "__main__":
  setproctitle.setproctitle("train_mlp")

  parser = argparse.ArgumentParser(description="Add two numbers.")
  parser.add_argument("--data_split", type=int, required=True)
  parser.add_argument("--batch_size", type=int, required=True)
  parser.add_argument("--epochs", type=int, required=True)
  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  train_loader, valid_loader = load_data_split(args.data_split, args.batch_size)

  model = MLP(num_features=49, hidden1_size=64, hidden2_size=64, num_classes=7).to(device)

  learning_rate = 0.02
  optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  criterion = CrossEntropyLoss()

  trainer = MLPTrainer(model=model, criterion=criterion, optimizer=optimizer, device=device)

  trainer.train(train_loader=train_loader, val_loader=valid_loader, num_epochs=args.epochs)

  torch.save(model.state_dict(), f'trained_models/ensemble_model/mlp_model_split{args.data_split}.pth')
