import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from torch.utils.data import DataLoader
from config import parentConfig as config
from setup_logging import logger
from tqdm import tqdm
from ImageDataset import CustomImageDataset

DEVICE = config.device
model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)


model.to(DEVICE)
model.eval()

criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy for single-class classification
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

transform = config.transform


mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()



train_dir = config.train_dir
val_dir = config.validation_dir

training_data = CustomImageDataset( train_dir , transform)
validation_data = CustomImageDataset( val_dir , transform)


train_dataloader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
validation_dataloader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

num_epochs = config.num_epochs

train_loss = []
validation_loss = []

for epoch in range(num_epochs):

    model.train()
    train_loss_running = 0.0
    validation_loss_running = 0.0


    for inputs, labels in tqdm(train_dataloader, desc = 'Training'):
        
      inputs = torch.tensor(inputs)
      labels = torch.tensor(labels)
      inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs.view(-1), labels.float())
      loss.backward()
      optimizer.step()

      train_loss_running += loss.item()
    
    model.eval()

    for inputs, labels in tqdm(validation_dataloader, desc = 'Validation'):
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.float())
            validation_loss_running += loss.item()
        
    train_loss_running = train_loss_running / len(train_dataloader)
    validation_loss_running = validation_loss_running / len(validation_dataloader)
    torch.save(model, config.model_dir+'epoch' + str(epoch) + 'deepfake_detection.pt')

    logger.info(f'Epoch [{epoch+1}/{num_epochs}],Training Loss: {train_loss_running:.6f}, Validation Loss: {validation_loss_running:.6f}')

    train_loss.append(train_loss_running)
    validation_loss.append(validation_loss_running)



#plot the training and validation loss
logger.info("Plotting the training and validation loss")


import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for servers
import matplotlib.pyplot as plt


# Plotting
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.grid(True)  # Add grid
plt.legend()

# Save the plot to a file (e.g., PNG format)
plt.savefig(os.path.join(config.plots_dir, 'loss_plot.png'))
plt.close()  # Close the plot to free memory




# Compute mean values
train_mean = sum(train_loss) / len(train_loss)
val_mean = sum(validation_loss) / len(validation_loss)
y_max = max(train_mean, val_mean)

# Plot
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.grid(True)
plt.ylim(0, y_max)  # Clamp y-axis
plt.legend()

plt.savefig(os.path.join(config.plots_dir,'loss_plot_clamped_mean.png'))
plt.close()



# Compute mean values
train_mean = sum(train_loss) / len(train_loss)
val_mean = sum(validation_loss) / len(validation_loss)
y_max = y_max/2

# Plot
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.grid(True)
plt.ylim(0, y_max)  # Clamp y-axis
plt.legend()

plt.savefig(os.path.join(config.plots_dir, 'loss_plot_clamped_mean/2.png'))
plt.close()