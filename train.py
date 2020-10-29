import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random

from utils import *
from losses import *
import Dataset
# from models.unet import UNet
# from models.pix2pix_networks import PixelDiscriminator
# from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
# from models.flownet2.models import FlowNet2SD
from models.model import convAE
from evaluate import val

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--iters', default=40000, type=int, help='The total iteration number.')
parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=1000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')
parser.add_argument('--show_flow', default=False, action='store_true',
                    help='If True, the first batch of ground truth optic flow could be visualized and saved.')
parser.add_argument('--flownet', default='lite', type=str, help='lite: LiteFlownet, 2sd: FlowNet2SD.')

args = parser.parse_args()
#train_cfg = update_config(args, mode='train')
#train_cfg.print_cfg()

assert args.flownet in ('lite', '2sd'), 'Flow net only supports LiteFlownet or FlowNet2SD currently.'
model = convAE(args.flownet)
generator = model.generator
discriminator = model.discriminator
flow_net = model.flow_net

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr)

if args.resume:
    generator.load_state_dict(torch.load(args.resume)['net_g'])
    discriminator.load_state_dict(torch.load(args.resume)['net_d'])
    optimizer_G.load_state_dict(torch.load(args.resume)['optimizer_g'])
    optimizer_D.load_state_dict(torch.load(args.resume)['optimizer_d'])
    print(f'Pre-trained generator and discriminator have been loaded.\n')
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    print('Generator and discriminator are going to be trained from scratch.\n')

if args.flownet == '2sd':
    flow_net.load_state_dict(torch.load('models/flownet2/FlowNet2-SD.pth')['state_dict'])
else:
    flow_net.load_state_dict(torch.load('models/liteFlownet/network-default.pytorch'))

flow_net.cuda().eval()  # Use flow_net to generate optic flows, so set to eval mode.

train_dataset = Dataset.train_dataset(args.train_data, args.img_size)

# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

writer = SummaryWriter(f'tensorboard_log/{args.dataset}_bs{args.batch_size}')
start_iter = int(args.resume.split('_')[-1].split('.')[0]) if args.resume else 0

trainer = Trainer(
    model=model,
    optimizer_G=optimizer_G,
    optimizer_D=optimizer_D,
    data_root=args.data_root,
    train_dataloader=train_dataloader,
    start_iter=start_iter,
    flow_backbone=args.batch_size,
    iters=args.iters,
    val_interval=args.val_interval,
    dataset=args.dataset,
    save_interval=args.save_interval,
    train_data=args.train_data,
    batch_size=args.batch_size,
    show_flow=args.show_flow,
)
trainer.train()
