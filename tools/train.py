
r"""
Main training of our framework.

"""
import os
import sys
from argparse import ArgumentParser
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

sys.path.append('./')
from dataset.dataset_linear import LinearEvalSingleDataset
from dataset.dataset import TripletSuperSSDatasetAug224
from models.model import TripletSupervisedBYOLModel, Resnet224Model
from utils.utils import get_features, create_data_loaders_from_arrays


parser = ArgumentParser()
parser.add_argument("--root", type=str, help="root directory of the datasets")
parser.add_argument("--root_eval", type=str, help="root directory of the evaluation datasets")
parser.add_argument("--name", type=str, default="SelSRL", help="name of experiment")
parser.add_argument("--byol_name",  type=str, default="byol_warmup", help="name of pretrained byol")
parser.add_argument("--port", type=str, default="12353", help="port of distributed")

parser.add_argument("--device", type=int, default=0, help="device id")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--max_epoch", type=int, default=300, help="max epoch number")
parser.add_argument("--linear_max_epoch", type=int, default=500, help="input dimension")

parser.add_argument("--resnet", type=str, default="resnet18", help="name of resnet used")
parser.add_argument("--alpha", type=float, default="1.0", help="alpha of triplet loss")
parser.add_argument("--beta", type=float, default="1.0", help="beta of supervised loss")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()

name = args.name
max_epoch = args.max_epoch
batch_size = args.batch_size
lr = args.lr
torch.backends.cudnn.benchmark = True
dataset_names = ["CoNSeP", "panNuke"]


def setup(rank, world_size):
    r"""
    Set up the environment for distributed parallel training
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    r"""
    Destroy the parallel training environment and clean up the training
    """
    dist.destroy_process_group()


def run_demo(demo_fn, world_size):
    r"""
    Generate processes for the parallel training with each process running demo_fn
    """
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def demo_basic(rank, world_size):
    r"""
    The basic function to train the feature encoder with our framework.
    The framework is composed of three branches: Structured Triplet, 
                                                 Attribute Learning, 
                                                 and Self-Supervision.
    We add up the loss of the three branches to lead the learning.
    """
    setup(rank, world_size)

    resnet = Resnet224Model(input_dim=4, model_name=args.resnet)
    resnet = resnet.to(rank)

    model = TripletSupervisedBYOLModel(net=resnet, hidden_layer='avgpool',
                                   alpha=args.alpha, beta=args.beta)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    if os.path.exists('checkpoints/{}/{}/latest.pth.tar'.format(name, args.resnet)):
        args.resume = 'checkpoints/{}/{}/latest.pth.tar'.format(name, args.resnet)

    start_epoch = 0

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        loc = 'cuda:{}'.format(rank)
        checkpoint = torch.load(args.resume, map_location=loc)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        checkpointpath = "checkpoints/{}/{}/improved-net_299.pt".format(args.byol_name,
                                                                        args.resnet)
        assert os.path.exists(checkpointpath), "no pretrained byol checkpoint path!"
        model.module.online_encoder.load_state_dict(torch.load(checkpointpath))
        print("Loaded warmed up feature encoder.")

    imagenet = TripletSuperSSDatasetAug224(args.root, image_size=224)

    sampler = DistributedSampler(imagenet)
    imagenet_dataloader = torch.utils.data.DataLoader(dataset=imagenet, batch_size=batch_size,
                                                      num_workers=4, sampler=sampler)
    print(len(imagenet))

    dist.barrier()
    if rank == 0:
        writer = SummaryWriter("logs/{}/{}".format(name, args.resnet))

    for epoch in range(start_epoch, args.linear_max_epoch):
        # train
        model.train()
        loss_sum = 0
        loss_byol_sum = 0
        loss_triplet_sum = 0
        loss_sup_sum = 0
        sampler.set_epoch(epoch)

        for local_batch in imagenet_dataloader:
            anc_img_one = local_batch['anc_img_one'].to(rank).float()
            anc_img_two = local_batch['anc_img_two'].to(rank).float()
            pos_img = local_batch['pos_img'].to(rank).float()
            neg_img = local_batch['neg_img'].to(rank).float()
            disease_gt = local_batch["anc_class"]
            method_gt = local_batch["anc_method"]

            loss, loss_byol, loss_sup, loss_triplet = model(anc_img_one, anc_img_two, pos_img,
                                                                neg_img, disease_gt, method_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.module.update_moving_average()

            if rank == 0:
                loss_sum += loss.detach().cpu().numpy()
                loss_byol_sum += loss_byol.detach().cpu().numpy()
                loss_sup_sum += loss_sup.detach().cpu().numpy()
                loss_triplet_sum += loss_triplet.detach().cpu().numpy()

        if rank == 0:
            print("epoch ", epoch, "loss", loss_sum / len(imagenet),
                  "byol", loss_byol_sum / len(imagenet),
                  "sup", loss_sup_sum / len(imagenet),
                  "triplet", loss_triplet_sum / len(imagenet))

            writer.add_scalar('Loss/loss', loss_sum / len(imagenet), epoch)
            writer.add_scalar('Loss/loss_byol', loss_byol_sum / len(imagenet), epoch)
            writer.add_scalar('Loss/loss_sup', loss_sup_sum / len(imagenet), epoch)
            writer.add_scalar('Loss/loss_triplet', loss_triplet_sum / len(imagenet), epoch)

        # check point save
        if rank == 0:
            os.makedirs("checkpoints/{}/{}".format(name, args.resnet), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoints/{}/{}/latest.pth.tar'.format(name, args.resnet))

            if epoch % 50 == 0 or epoch == (max_epoch - 1):
                checkpoint_path = "checkpoints/{}/{}/improved-net_{}.pt".format(name,
                                                                    args.resnet, epoch)
                torch.save(model.module.state_dict(), checkpoint_path)

        dist.barrier()

    if rank == 0:
        writer.close()

    cleanup()

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

    checkpoint_path = "checkpoints/{}/{}/improved-net_{}.pt".format(name, 
                                                                args.resnet, 
                                                                args.max_epoch-1)
    learner.load_state_dict(torch.load(checkpoint_path, map_location='cuda:{}'.format(0)))
    model = learner.online_encoder

    model.eval()
    model.to(device)

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

    if not os.path.exists("checkpoints/{}/{}/improved-net_{}.pt".format(name,
                                                            args.resnet,
                                                            args.max_epoch-1)):
        run_demo(demo_basic, n_gpus)

    for i in range(2):
        linear_eval_train(i)
    