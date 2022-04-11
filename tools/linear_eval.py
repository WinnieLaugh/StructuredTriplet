import torch
import os
from models.model import MLP
from models.model import resnet128
from dataset.dataset_linear import LinearEvalSingleDataset
import torch.nn as nn


device = torch.device(2)
dst_folder = "/mnt/group-ai-medical-2/private/wenhuazhang/data/feature_exctracted/PanNuke/triplet_linear_global"
imagenet = LinearEvalSingleDataset("/mnt/group-ai-medical-2/private/wenhuazhang/data/PanNuke/train/global",
                             "/mnt/group-ai-medical-2/private/wenhuazhang/data/PanNuke/train/global_mask", is_train=True)
imagenet_dataloader = torch.utils.data.DataLoader(imagenet, batch_size=512, shuffle=True, num_workers=16)
imagenet_test = LinearEvalSingleDataset("/mnt/group-ai-medical-2/private/wenhuazhang/data/PanNuke/val/global",
                                          "/mnt/group-ai-medical-2/private/wenhuazhang/data/PanNuke/val/global_mask",
                                          is_train=False)
imagenet_dataloader_test = torch.utils.data.DataLoader(imagenet_test, batch_size=512, shuffle=False)

resnet = resnet128(input_dim=4, istrain=False)
resnet.load_state_dict(torch.load("checkpoints/instance_contrastive_byol_global_0.01/improved-net_{}.pt".format(490)))
print("loaded real instance_contrastive_byol path")
resnet = resnet.to(device)

linear_layer = MLP(512, 5, 256).to(device)
print("linear layer, ", linear_layer)
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(linear_layer.parameters(), lr=1e-3)
max_epoch = 500
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

i = 0
for _ in range(max_epoch):
    linear_layer.train()
    loss_sum = 0

    for local_batch, local_labels in imagenet_dataloader:
        local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device)
        with torch.no_grad():
            feature = resnet(local_batch)
            feature = feature.detach()

        classification = linear_layer(feature)
        output = cross_entropy_loss(classification, local_labels)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        loss_sum += output.detach().cpu().numpy()

    print("epoch, ", _, "loss, ", loss_sum/len(imagenet))

    if _ % 10 == 0:
        os.makedirs("checkpoints/triplet_linear_global", exist_ok=True)
        CHECKPOINT_PATH = "checkpoints/triplet_linear_global/linear_{}.pt".format(_)
        torch.save(linear_layer.state_dict(), CHECKPOINT_PATH) 

        total = 0
        correct = 0
        with torch.no_grad():
            linear_layer.eval()
            for local_batch, local_labels in imagenet_dataloader_test:
                local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device)
                feature = resnet(local_batch)
                classification = linear_layer(feature)
                _, predicted = torch.max(classification.data, 1)
                total += local_labels.size(0)
                correct += (predicted == local_labels).sum().item()

            print("accuracy:", correct / total)
