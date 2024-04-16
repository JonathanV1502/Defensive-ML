import copy
from tqdm import tqdm
import torch
from untils import calculate_confusion_matrix, accuracy, precision, recall, f1_score
from sklearn import metrics

class MLPTrainer:
  def __init__(self, model, criterion, optimizer, device='cpu'):
    self.model = model.to(device)
    self.criterion = criterion
    self.optimizer = optimizer
    self.device = device

    self.best_model_state = None
    self.history = {'train_loss': [],
                    'val_loss': [],
                    'train_cr': [],
                    'val_cr': [],}

  def train(self, train_loader, val_loader, num_epochs):
    best_loss = float('inf')
    with tqdm(total=num_epochs, desc='Training Progress') as pbar:
      for epoch in range(num_epochs):
        # Train step
        train_loss, train_cr = self.__train_step(train_loader)
        # Val step
        val_loss, val_cr = self.__train_step(val_loader) # Fix this later

        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_cr'].append(train_cr)
        self.history['val_cr'].append(val_cr)


        # Update best model
        if best_loss > val_loss:
          self.best_model_state = copy.deepcopy(self.model.state_dict())
          best_loss = val_loss

        pbar.set_postfix(Train_loss=f"{train_loss}", Val_Loss=f"{val_loss}")
        pbar.update(1)

      self.model.load_state_dict(self.best_model_state)

  def __train_step(self, train_loader):
    self.model.train()
    epoch_loss = 0.0
    y_preds = []
    y_trues = []

    for inputs, labels in train_loader:
      labels = labels.squeeze(1).to(self.device)
      inputs = inputs.to(self.device)
      self.optimizer.zero_grad()
      outputs = self.model(inputs)
      loss = self.criterion(outputs, labels)
      loss.backward()
      self.optimizer.step()

      # get metrics
      y_preds += outputs.argmax(axis=1).cpu().tolist()
      y_trues += labels.cpu().tolist()
      epoch_loss += loss.item()
    
    epoch_loss /= len(train_loader)
    cr = metrics.classification_report(y_trues, y_preds, output_dict=True, zero_division=0.0)

    return epoch_loss, cr

  def eval(self, val_loader):
    self.model.eval()
    epoch_loss = 0.0
    y_preds = []
    y_trues = []
    
    with torch.no_grad():
      for inputs, labels in val_loader:
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs.squeeze(), labels.to(self.device))
        
        # get metrics
        y_preds += outputs.argmax(axis=1).cpu().tolist()
        y_trues += y.cpu().tolist()
        epoch_loss += loss.item()

    epoch_loss /= len(val_loader)
    cr = metrics.classification_report(y_trues, y_preds, output_dict=True, zero_division=0.0)

    return epoch_loss, cr

  def get_history(self):
    return self.history

  def get_model(self):
    return self.model