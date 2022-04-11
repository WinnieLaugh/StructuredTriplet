
import os
import sys
sys.path.append('./')
import numpy as np
from models.byol_triplet_supervised import BYOL_Triplet_Supervised_Model, MLP
from models.model import Resnet224Model
from dataset.dataset_linear import LinearEvalSingleDataset

import torch
import torch.nn as nn
from dataset.utils import get_features, create_data_loaders_from_arrays

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

device = torch.device(2)
imagenet = LinearEvalSingleDataset("/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/20x",
                             0, split_name="train")
imagenet_dataloader = torch.utils.data.DataLoader(imagenet, batch_size=512, shuffle=True, num_workers=16)
imagenet_test = LinearEvalSingleDataset("/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/20x",
                             0, split_name="test")
imagenet_dataloader_test = torch.utils.data.DataLoader(imagenet_test, batch_size=512, shuffle=False)

resnet = Resnet224Model(input_dim=4, model_name="resnet18")

learner = BYOL_Triplet_Supervised_Model(net=resnet, hidden_layer='avgpool')

CHECKPOINT_PATH = "checkpoints/three_branch_v3/resnet18/improved-net_last.pt"
learner.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cuda:{}'.format(0)))
resnet = learner.online_encoder
resnet = resnet.to(device)

X_train, y_train, X_test, y_test = get_features(resnet, imagenet_dataloader, imagenet_dataloader_test, device)
train_loader, test_loader = create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, 512)

linear_layer = MLP(512, 7, 256).to(device)
print("linear layer, ", linear_layer)
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(linear_layer.parameters(), lr=1e-4)
max_epoch = 300

i = 0
for epoch_idx in range(1, max_epoch+1):
    linear_layer.train()
    loss_sum = 0
    total = 0
    correct = 0
    for local_batch, local_labels in train_loader:
        local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device)

        classification = linear_layer(local_batch)
        _, predicted = torch.max(classification.data, 1)

        output = cross_entropy_loss(classification, local_labels)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()

        total += local_labels.size(0)
        correct += (predicted == local_labels).sum().item()

    if epoch_idx % 100 == 0:
        print(f"Step [{epoch_idx}/500]\t "
                f"loss:{loss_sum / len(imagenet)}, acc: {correct / total}")

        total = 0
        correct = 0
        with torch.no_grad():
            linear_layer.eval()
            y_pred_this = []
            y_true_this = []
            for local_batch, local_labels in test_loader:
                local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device)
                classification = linear_layer(local_batch)
                _, predicted = torch.max(classification.data, 1)

                total += local_labels.size(0)
                correct += (predicted == local_labels).sum().item()

                y_pred_this += predicted.detach().cpu().numpy().tolist()
                y_true_this += local_labels.cpu().numpy().tolist()

            y_pred_this = np.array(y_pred_this)
            y_true_this = np.array(y_true_this)

            print("y pred this shape: ", y_pred_this.shape)
            print("y true this shape: ", y_true_this.shape)
            accuracy_score_this = np.sum(y_pred_this == y_true_this) / y_pred_this.shape[0]
            precision_score_this = precision_score(y_pred_this, y_true_this, average='weighted')
            recall_score_this = recall_score(y_pred_this, y_true_this, average='weighted')
            f1_score_this = f1_score(y_pred_this, y_true_this, average='weighted')
            print("accuracy: ", accuracy_score_this,
              " precision: ", precision_score_this,
              " recall: ", recall_score_this,
              " f1 score: ", f1_score_this,
              " total num: ", y_pred_this.shape[0])
