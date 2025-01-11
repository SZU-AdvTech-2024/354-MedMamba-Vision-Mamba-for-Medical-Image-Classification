import os
import sys
import json
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
#from torch.utils.data import random_split

from tqdm import tqdm

from src.MedMamba import VSSM as medmamba # import model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
    "test": transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}



test_dataset = datasets.ImageFolder(root="CPN X-ray",
                                        transform=data_transform["test"])
test_num = len(test_dataset)

batch_size = 64
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

print("using {} images for test.".format(test_num))

pretrained_path = "Medmamba_weights/pneumoniamnist/Adam=0.0001_medmamba_s_20250111_0036_Net.pth"

# Use map_location to ensure loading on the correct device (cuda:0 or cpu)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(pretrained_path, map_location=device)
#print(checkpoint.keys())  # Check the structure of the checkpoint


net = medmamba(depths=[2, 2, 8, 2],dims=[96,192,384,768], num_classes=2)
net.load_state_dict(checkpoint)
final_net = net.to(device)

#optimizer = optim.Adam(final_net.parameters(), lr=0.1)
#optimizer = optim.SGD(final_net.parameters(), lr=0.001)

test_accuracies = []
test_aucs = []

test_labels = []
test_preds = []
test_probs = []  # To store probabilities for AUC


# test
final_net.eval()
with torch.no_grad():
    for idx, (data, target) in enumerate(tqdm(test_loader)):
        img = data.float().to(device)
        label = target.type(torch.uint8).to(device)
        y_test_pred = final_net(img)
        y_test_pred_label = torch.argmax(y_test_pred, dim=1)
        test_probs.extend(y_test_pred[:, 1].detach().cpu().numpy())  
        test_labels.extend(label.cpu().numpy())
        test_preds.extend(y_test_pred_label.cpu().numpy())


# Saving metrics of test
test_accuracy = sum(1 for x,y in zip(test_preds, test_labels) if x==y) / len(test_labels)
test_accuracies.append(test_accuracy)

test_auc = roc_auc_score(test_labels, test_probs)
test_aucs.append(test_auc)

# Compute metrics
test_tn, test_fp, test_fn, test_tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()

test_sensitivity = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0.0
test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0.0
test_precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0.0
test_f1 = (2 * test_precision * test_sensitivity) / (test_precision + test_sensitivity) if (test_precision + test_sensitivity) > 0 else 0.0



test_labels_array = np.array(test_labels) 
test_preds_array = np.array(test_preds)   
    

print("Test predictions:   ", test_preds_array[:30])
print("Test labels:        ", test_labels_array[:30])
print("-" * 30)

## Metric statements

print(f"Test accuracy: {test_accuracy:.4f}, Test sensitivity: {test_sensitivity:.4f}, Test specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print("*" * 30)



