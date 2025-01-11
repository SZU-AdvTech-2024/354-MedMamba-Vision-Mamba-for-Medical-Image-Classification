import os
import sys
import json
import datetime

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms, models
from tqdm import tqdm

from collections import Counter
from src.MedMamba import VSSM, medmamba_t, medmamba_s, medmamba_b

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score

import medmnist
from medmnist import INFO, Evaluator



info = INFO['pneumoniamnist']
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


dataset_name = 'pneumoniamnist'

# Check if the folder exists, if not, create it
if not os.path.exists(f'{dataset_name}/'):
    os.makedirs(f'{dataset_name}/')
    
batch_size = 64
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

train_dataset = DataClass(split='train', transform=data_transform["train"], download=True)

train_num = len(train_dataset)


train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)


val_dataset = DataClass(split='val', transform=data_transform["val"], download=True)

val_num = len(val_dataset)
validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=nw)

# Initialize the model

medmamba_s = VSSM(depths=[2, 2, 8, 2],dims=[96,192,384,768],num_classes=2)
#medmamba_b = VSSM(depths=[2, 2, 12, 2],dims=[128,256,512,1024],num_classes=2)
medmamba_t = VSSM(depths=[2, 2, 4, 2],dims=[96,192,384,768],num_classes=2)

model_name = 'medmamba_t'
net = medmamba_t
net.to(device)

# Loss and optimizer
# class_weights = torch.tensor([1.0, 399/147], device=device)  # Adjust weights based on class distribution

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.001)
# optimizer = SGD(net.parameters(), lr=0.001)

# Set up the learning rate scheduler
milestones = [50, 75] #range(10,100,10)
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

# Set training parameters
epochs = 150
best_acc = 0.0
train_steps = len(train_loader)



save_path_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
save_path = f'Medmamba_weights/{dataset_name}/{model_name}_{save_path_time}_Net.pth'

train_accuracies = []
train_aucs = []  # Store AUC
tr_sen = []
tr_spe = []
train_losses = []
tr_f1s = []

val_accuracies = []
val_aucs = []  # Store AUC
val_sen = []
val_spe = []
val_losses = []
val_f1s = []

for epoch in range(epochs):
    # train
    net.train()
    train_labels = []
    train_preds = []
    train_probs = []  # To store probabilities for AUC
    train_run_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for idx, (data, target) in enumerate(tqdm(train_bar)):
        img = data.float().to(device)
        label = target.squeeze(dim=1).long().to(device)  # Use .squeeze() to flatten target to shape (batch_size,)
        optimizer.zero_grad()
        y_pred = net(img)
        y_pred_label = torch.argmax(y_pred, dim=1)
        # Collect predicted probabilities for AUC, To save memory from GPU
        train_probs.extend(y_pred[:, 1].detach().cpu().numpy())  # Positive class probabilities
        train_labels.extend(label.cpu().numpy())  # True labels
        train_preds.extend(y_pred_label.cpu().numpy())  # Predicted labels
        loss = loss_function(y_pred, label)
        train_run_loss += loss.item()
        loss.backward()
        optimizer.step()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                    epochs,
                                                                    loss)
    
    train_loss = train_run_loss / (idx + 1)
    train_losses.append(train_loss)

    # Saving metrics of training
    train_accuracy = sum(1 for x,y in zip(train_preds, train_labels) if x==y) / len(train_labels) # OVerall accuracy
    train_accuracies.append(train_accuracy)

    train_auc = roc_auc_score(train_labels, train_probs)
    train_aucs.append(train_auc)

    # Compute metrics
    tr_tn, tr_fp, tr_fn, tr_tp = confusion_matrix(train_labels, train_preds, labels=[0, 1]).ravel()

    tr_sensitivity = tr_tp / (tr_tp + tr_fn) if (tr_tp + tr_fn) > 0 else 0.0
    tr_sen.append(tr_sensitivity)
    tr_specificity = tr_tn / (tr_tn + tr_fp) if (tr_tn + tr_fp) > 0 else 0.0
    tr_spe.append(tr_specificity)
    tr_precision = tr_tp / (tr_tp + tr_fp) if (tr_tp + tr_fp) > 0 else 0.0

    tr_f1 = (2 * tr_precision * tr_sensitivity) / (tr_precision + tr_sensitivity) if (tr_precision + tr_sensitivity) > 0 else 0.0
    tr_f1s.append(tr_f1)


    # validate
    net.eval()
    val_labels = []
    val_preds = []
    val_probs = []  # To store probabilities for AUC
    val_run_loss = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(validate_loader)):
            img = data.float().to(device)
            label = target.squeeze(dim=1).long().to(device)
            y_val_pred = net(img)
            y_val_pred_label = torch.argmax(y_val_pred, dim=1)
            # Collect predicted probabilities for AUC
            val_probs.extend(y_val_pred[:, 1].detach().cpu().numpy())  # Positive class probabilities
            val_labels.extend(label.cpu().numpy())
            val_preds.extend(y_val_pred_label.cpu().numpy())
            loss = loss_function(y_val_pred, label)
            val_run_loss += loss.item()

    val_loss = val_run_loss / (idx + 1)
    val_losses.append(val_loss)
    
    # Saving metrics of validation
    val_accuracy = sum(1 for x,y in zip(val_preds, val_labels) if x==y) / len(val_labels)
    val_accuracies.append(val_accuracy)

    val_auc = roc_auc_score(val_labels, val_probs)
    val_aucs.append(val_auc)

    # Compute metrics
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(val_labels, val_preds, labels=[0, 1]).ravel()

    val_sensitivity = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
    val_sen.append(val_sensitivity)
    val_specificity = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0.0
    val_spe.append(val_specificity)
    val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0

    val_f1 = (2 * val_precision * val_sensitivity) / (val_precision + val_sensitivity) if (val_precision + val_sensitivity) > 0 else 0.0
    val_f1s.append(val_f1)


    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(net.state_dict(), save_path)

print('Finished Training')

train_labels_array = np.array(train_labels) 
train_preds_array = np.array(train_preds)   
val_labels_array = np.array(val_labels)     
val_preds_array = np.array(val_preds)        
print("*" * 30)
print(f"EPOCH {epoch+1}")
print("Training predictions:   ", train_preds_array[:30])
print("Training labels:        ", train_labels_array[:30])
print("Validation predictions: ", val_preds_array[:30])
print("Validation labels:      ", val_labels_array[:30])
print("-" * 30)

## Metric statements

print(f"Accuracy: {train_accuracy:.4f}, Sensitivity: {tr_sensitivity:.4f}, Specificity: {tr_specificity:.4f}, F1-Score: {tr_f1:.4f}")
print(f"AUC: {train_auc:.4f}, Train Loss: {train_loss:.4f}")
print("*" * 30)
print(f"Val. Accuracy: {val_accuracy:.4f}, Val. Sensitivity: {val_sensitivity:.4f}, Val. Specificity: {val_specificity:.4f}, Val. F1-Score: {val_f1:.4f}")
print(f"Val. AUC: {val_auc:.4f}, Valid Loss: {val_loss:.4f}")

epochs_list = list(range(1, epochs + 1))

############################################

# Generate a timestamp for the file name
result_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# Create the filename with the timestamp
filename = f"Medmamba_results/{dataset_name}/training_results_{model_name}_{result_time}.txt"

# Open the file and save the results
with open(filename, "w") as f:
    # Writing the training metrics
    f.write(f"Accuracy: {train_accuracy:.4f}, Sensitivity: {tr_sensitivity:.4f}, Specificity: {tr_specificity:.4f}, F1-Score: {tr_f1:.4f}\n")
    f.write(f"AUC: {train_auc:.4f}, Train Loss: {train_loss:.4f}\n")
    f.write("*" * 30 + "\n")
    
    # Writing the validation metrics
    f.write(f"Val. Accuracy: {val_accuracy:.4f}, Val. Sensitivity: {val_sensitivity:.4f}, Val. Specificity: {val_specificity:.4f}, Val. F1-Score: {val_f1:.4f}\n")
    f.write(f"Val. AUC: {val_auc:.4f}, Valid Loss: {val_loss:.4f}\n")

############################################

# Plot 1
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_list, train_accuracies, label='Training Accuracy')
plt.plot(epochs_list, val_accuracies, label='Validation Accuracy')
plt.xticks(ticks=list(range(0, epochs + 1, 10)))  
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.grid()
plt.tight_layout()
plt.legend()
#
plt.subplot(1, 2, 2)
plt.plot(epochs_list, train_aucs, label='Training AUC')
plt.plot(epochs_list, val_aucs, label='Validation AUC')
plt.xticks(ticks=list(range(0, epochs + 1, 10))) 
plt.title('AUC over epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"Medmamba_results/{dataset_name}/Accuracy & AUC_{model_name}_{save_path_time}.png")  # Save to host server
plt.close()


# Plot 2
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_list, tr_sen, label='Training sensitivity')
plt.plot(epochs_list, val_sen, label='Validation sensitivity')
plt.xticks(ticks=list(range(0, epochs + 1, 10))) 
plt.title('Sensitivity over epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.grid()
plt.tight_layout()
plt.legend()
#
plt.subplot(1, 2, 2)
plt.plot(epochs_list, tr_spe, label='Training specificity')
plt.plot(epochs_list, val_spe, label='Validation specificity')
plt.xticks(ticks=list(range(0, epochs + 1, 10)))  
plt.title('Specificity over epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"Medmamba_results/{dataset_name}/Sensitivity and specificity_{model_name}_{save_path_time}.png")  # Save to host server
plt.close()

# Plot 3
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_list, train_losses, label='Training Loss')
plt.plot(epochs_list, val_losses, label='Validation Loss')
plt.xticks(ticks=list(range(0, epochs + 1, 10))) 
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.grid()
plt.tight_layout()
plt.legend()
#
plt.subplot(1, 2, 2)
plt.plot(epochs_list, tr_f1s, label='Training F1')
plt.plot(epochs_list, val_f1s, label='Validation F1')
plt.xticks(ticks=list(range(0, epochs + 1, 10))) 
plt.title('F1 over epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(f"Medmamba_results/{dataset_name}/Loss & F1_{model_name}_{save_path_time}.png")  # Save to host server
plt.close()
