import torch
from torch import nn
from torch.utils.data import DataLoader
from time import time
import numpy as np
device = ('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, epochs ,train_loader ,test_loader ,  opt, criterion, train_data, test_data):
  losses = [] # Train Losses in different epochs
  val_losses = []# Val Losses in different epochs
  train_accuracy = []
  val_accuracy = []
  for epoch in range(epochs):
    tic = time() # Start of epoch
    # Train
    model = model.train()
    batch_losses = []
    epoch_corrects = 0 # train corrects
    for batch in train_loader:
      input = batch[0].to(device) # Images
      target = batch[1].to(device) # Labels
      output = model.forward(input)
      opt.zero_grad()
      loss = criterion(output, target)
      loss.backward()
      opt.step()
      batch_losses.append(loss.item())
      batch_corrects = sum(target == output.argmax(dim = 1))
      epoch_corrects += batch_corrects
    epoch_loss = np.mean(batch_losses)
    losses.append(epoch_loss)
    epoch_accuracy = epoch_corrects / len(train_data)
    train_accuracy.append(epoch_accuracy)
    train_string = f"Epoch {epoch}, loss : {epoch_loss: 0.5f}, accuracy : {epoch_accuracy: 0.5f}, "
    
    model = model.eval() ### To disable dropout
    # Validation
    batch_losses = []
    epoch_corrects = 0 # val corrects
    for batch in test_loader:
      input = batch[0].to(device) # Images
      target = batch[1].to(device) # Labels
      output = model.forward(input)
      with torch.no_grad():
        loss = criterion(output, target)
        batch_losses.append(loss.item())
        batch_corrects = sum(target == output.argmax(dim = 1))
        epoch_corrects += batch_corrects
    epoch_loss = np.mean(batch_losses)
    val_losses.append(epoch_loss)
    epoch_accuracy = epoch_corrects / len(test_data)
    val_accuracy.append(epoch_accuracy)
    toc = time() #End of epoch
    val_string = f" val_loss : {epoch_loss: 0.5f}, val_accuracy : {epoch_accuracy: 0.5f}, time : {toc - tic: 0.1f}"
    print(train_string + val_string)

  return model
    
