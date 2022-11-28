import torch
from torch import nn
from argparse import ArgumentParser
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import sys
sys.path.append('./')
from dataset.dataset_linear import LinearEvalSingleDataset
from models.model import TripletSupervisedBYOLModel, Resnet224Model
from utils.utils import get_features, create_data_loaders_from_arrays


parser = ArgumentParser()
parser.add_argument("--root_eval", type=str, default="")
parser.add_argument("--name", type=str, default="SelSRL", help="name of experiment")
parser.add_argument("--byol_name",  type=str, default="byol_warmup", help="name of pretrained byol")
parser.add_argument("--port", type=str, default="12353", help="port of distributed")

parser.add_argument("--device", type=int, default=0, help="device id")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--max_epoch", type=int, default=300, help="max epoch number")
parser.add_argument("--linear_max_epoch", type=int, default=500, help="input dimension")
parser.add_argument("--resnet", type=str, default="resnet18", help="name of resnet used")

args = parser.parse_args()

name = args.name
max_epoch = args.max_epoch
batch_size = args.batch_size
lr = args.lr
torch.backends.cudnn.benchmark = True
dataset_names = ["CoNSeP", "panNuke"]


def linear_eval_train(dataset_idx):
    r"""
    The basic function of linear evaluation protocal.
    """
    device = torch.device(0)

    imagenet_train = LinearEvalSingleDataset(args.root_eval, dataset_idx=dataset_idx,
                                            split_name="train")
    imagenet_test = LinearEvalSingleDataset(args.root_eval, dataset_idx=dataset_idx,
                                            split_name="test")

    imagenet_dataloader_train = torch.utils.data.DataLoader(dataset=imagenet_train, batch_size=512,
                                                            num_workers=4, shuffle=False)
    imagenet_dataloader_test = torch.utils.data.DataLoader(dataset=imagenet_test, batch_size=512,
                                                           num_workers=4, shuffle=False)

    resnet = Resnet224Model(input_dim=4, model_name=args.resnet)
    learner = TripletSupervisedBYOLModel(net=resnet, hidden_layer='avgpool')

    checkpoint_path = "checkpoints/improved-net_299.pt"
    learner.load_state_dict(torch.load(checkpoint_path, map_location='cuda:{}'.format(0)))
    model = learner.online_encoder

    model.eval()
    model.to(device)

    print("model loaded")
    x_train, y_train, x_test, y_test =\
        get_features(model, imagenet_dataloader_train, imagenet_dataloader_test, device)
    train_loader, test_loader = create_data_loaders_from_arrays(x_train, y_train, x_test, y_test,
                                                                args.batch_size)

    if dataset_idx == 0:
        linear_layer = nn.Linear(512, 7).to(device)
    elif dataset_idx == 1:
        linear_layer = nn.Linear(512, 5).to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_layer.parameters(), lr=3e-4)

    for epoch_idx in range(args.linear_max_epoch):
        linear_layer.train()
        loss_sum = 0
        total, correct = 0, 0

        for data, gt_labels in train_loader:
            data, gt_labels = data.to(device).float(), gt_labels.to(device)
            output = linear_layer(data)
            loss = cross_entropy_loss(output, gt_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().cpu().numpy()

            _, predicted = torch.max(output.data, 1)
            total += gt_labels.size(0)
            correct += (predicted == gt_labels).sum().item()

        if epoch_idx % 100 == 0:
            print(f"Step [{epoch_idx}/{args.linear_max_epoch}]\t "
                  f"loss:{loss_sum / len(imagenet_train)}, acc: {correct / total}")

    with torch.no_grad():
        linear_layer.eval()
        y_pred_this = []
        y_true_this = []

        for local_batch, local_labels in test_loader:
            local_batch = local_batch.to(device).float()
            output = linear_layer(local_batch)
            _, predicted = torch.max(output.data, 1)

            y_pred_this += predicted.detach().cpu().numpy().tolist()
            y_true_this += local_labels.numpy().tolist()

        y_pred_this = np.array(y_pred_this)
        y_true_this = np.array(y_true_this)

        print("y pred this shape: ", y_pred_this.shape)
        print("y true this shape: ", y_true_this.shape)
        accuracy_score_this = np.sum(y_pred_this == y_true_this) / y_pred_this.shape[0]
        precision_score_this = precision_score(y_pred_this, y_true_this, average='weighted')
        recall_score_this = recall_score(y_pred_this, y_true_this, average='weighted')
        f1_score_this = f1_score(y_pred_this, y_true_this, average='weighted')
        print(dataset_names[dataset_idx],
              " accuracy: ", accuracy_score_this,
              " precision: ", precision_score_this,
              " recall: ", recall_score_this,
              " f1 score: ", f1_score_this,
              " total num: ", y_pred_this.shape[0])

    return accuracy_score_this, precision_score_this, recall_score_this, f1_score_this



if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    for i in range(2):
        linear_eval_train(i)
