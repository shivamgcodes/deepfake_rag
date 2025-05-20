import torch
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from config import parentConfig as config
from ImageDataset import CustomImageDataset
from torch.utils.data import DataLoader
import argparse
from facenet_pytorch import MTCNN, InceptionResnetV1
from setup_logging import logger
#plot the graphs and save them
#recall, precision, accuracy, f1_score

DEVICE = config.device
transform = config.transform
val_dir = config.validation_dir
validation_data = CustomImageDataset( val_dir , transform)
validation_dataloader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=False)


parser = argparse.ArgumentParser(description="Evaluate model on validation set")
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pt or .pth)")
args = parser.parse_args()

model = model = InceptionResnetV1(
    pretrained="args.model_path",
    classify=True,
    num_classes=1,
    device=DEVICE
)


model_outputs = []
labels_list = []
for inputs, labels in validation_dataloader:
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs)
        model_outputs.append(outputs.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    
model_outputs = np.concatenate(model_outputs)
labels_list = np.concatenate(labels_list)



# Replace these with your real arrays
y_true = labels_list
y_scores = model_outputs

thresholds = np.linspace(0, 1, 200)
precisions = []
recalls = []
f1s = []

for thresh in thresholds:
    y_pred = (y_scores >= thresh).astype(int)
    precisions.append(precision_score(y_true, y_pred, zero_division=0))
    recalls.append(recall_score(y_true, y_pred, zero_division=0))
    f1s.append(f1_score(y_true, y_pred, zero_division=0))

# Find the optimal threshold (max F1)
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
best_f1 = f1s[best_idx]

# Plot
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.plot(thresholds, f1s, label='F1 Score')
plt.axvline(best_threshold, linestyle='--', color='black', label=f'Best Threshold = {best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 vs Threshold')
plt.legend()
plt.grid(True)

logger.info(f"Optimal threshold: {best_threshold:.3f}")
logger.info(f"Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}, F1: {best_f1:.3f}")

plt.savefig(config.plots_dir + "threshold_analysis.png")
plt.close()
