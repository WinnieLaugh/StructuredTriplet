r"""
The warm up training of our framework.

We first warmup the encoders by a conventioal self-training process
"""

import os
import sys
from argparse import ArgumentParser
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
sys.path.append('./')

from dataset.dataset import BYOLDatasetAug224
from models.model import Resnet224Model, BYOL


parser = ArgumentParser()
parser.add_argument("--root", type=str, help="root directory of the datasets")
parser.add_argument("--name", type=str, default="byol_warmup", help="name of experiment")
parser.add_argument("--resnet", type=str, default="resnet18", help="name of resnet used")

parser.add_argument("--device", type=int, default=0, help="device id")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--max_epoch", type=int, default=300, help="max epoch number")
parser.add_argument("--port", type=str, default="12353", help="port of distributed")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()

name = args.name
max_epoch = args.max_epoch
batch_size = args.batch_size
lr = args.lr
torch.backends.cudnn.benchmark = True


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
    The basic function to train a model
    """
    setup(rank, world_size)

    if rank == 0:
        writer = SummaryWriter("logs/{}/{}".format(name, args.resnet))

    resnet = Resnet224Model(input_dim=4, model_name=args.resnet)
    resnet = resnet.to(rank)

    model = BYOL(net=resnet, hidden_layer='avgpool')

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.Adam(ddp_model.module.parameters(), lr=lr, weight_decay=1e-4)

    start_epoch = 0
    if os.path.exists('checkpoints/{}/{}/latest.pth.tar'.format(name, args.resnet)):
        args.resume = 'checkpoints/{}/{}/latest.pth.tar'.format(name, args.resnet)

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        loc = 'cuda:{}'.format(rank)
        checkpoint = torch.load(args.resume, map_location=loc)
        start_epoch = checkpoint['epoch']
        ddp_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']))

    imagenet = BYOLDatasetAug224(args.root, image_size=224)
    sampler = DistributedSampler(imagenet)
    imagenet_dataloader = torch.utils.data.DataLoader(dataset=imagenet, batch_size=batch_size,
                                                      num_workers=4, sampler=sampler)

    dist.barrier()

    for epoch_idx in range(start_epoch, max_epoch):
        # train
        ddp_model.train()
        loss_sum = 0
        sampler.set_epoch(epoch_idx)

        for image1, image2 in imagenet_dataloader:
            image1 = image1.to(rank).float()
            image2 = image2.to(rank).float()
            loss = ddp_model(image1, image2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ddp_model.module.update_moving_average()

            if rank == 0:
                loss_sum += loss.detach().cpu().numpy()

        if rank == 0:
            print("epoch ", epoch_idx, "loss", loss_sum / len(imagenet))
            writer.add_scalar('Loss/loss', loss_sum / len(imagenet), epoch_idx)

        # check point save
        if rank == 0:
            os.makedirs("checkpoints/{}/{}".format(name, args.resnet), exist_ok=True)
            torch.save({
                'epoch': epoch_idx + 1,
                'state_dict': ddp_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoints/{}/{}/latest.pth.tar'.format(name, args.resnet))

            if epoch_idx % 50 == 0 or epoch_idx == (max_epoch - 1):
                checkpointpath = "checkpoints/{}/{}/improved-net_{}.pt".format(name,
                                                                        args.resnet,
                                                                        epoch_idx)
                torch.save(ddp_model.module.online_encoder.state_dict(), checkpointpath)

        dist.barrier()

    if rank == 0:
        writer.close()

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    CHECKPOINT_PATH = "checkpoints/{}/{}/improved-net_{}.pt".format(name,
                                                            args.resnet,
                                                            max_epoch - 1)
    if not os.path.exists(CHECKPOINT_PATH):
        run_demo(demo_basic, n_gpus)
