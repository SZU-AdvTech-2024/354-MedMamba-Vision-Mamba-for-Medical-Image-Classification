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

from src.Medvit import MedViT_large
from torchvision.models import resnet50, inception_v3
#from torchvision.models.resnet import ResNet, Bottleneck

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score



device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device0))

data_transform = {
    "test": transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}



test_dataset = datasets.ImageFolder(root="CPN X-ray",
                                        transform=data_transform["test"])
test_num = len(test_dataset)

batch_size = 128
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

print("using {} images for test.".format(test_num))

pretrained_path = "Medvit_weights/pneumoniamnist/MedViT_large_20250103_2213_Net.pth"

# Use map_location to ensure loading on the correct device (cuda:0 or cpu)
device0 = "cuda:0" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(pretrained_path, map_location=device0)
#print(checkpoint.keys())  # Check the structure of the checkpoint

# Adjust the state dict for DataParallel
checkpoint_state_dict = checkpoint
if "module." in list(checkpoint_state_dict.keys())[0]:  # Check if "module." prefix exists
    new_state_dict = {}
    for k, v in checkpoint_state_dict.items():
        new_state_dict[k[7:]] = v  # Remove "module." prefix from each key
    checkpoint_state_dict = new_state_dict  # Replace with adjusted state dict


net = MedViT_large(num_classes=2)
net.load_state_dict(checkpoint_state_dict)
net.to(device0)
net = nn.DataParallel(net, device_ids=[0, 1])  # Use both GPUs


optimizer = optim.Adam(net.parameters(), lr=0.0001)


test_accuracies = []
test_aucs = []

test_labels = []
test_preds = []
test_probs = []  # To store probabilities for AUC


# test
net.eval()
with torch.no_grad():
    for idx, (data, target) in enumerate(tqdm(test_loader)):
        img = data.float().to(device0)
        label = target.type(torch.uint8).to(device0)
        y_test_pred = net(img)
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
print(f"AUC: {test_auc:.4f}")
print("*" * 30)



