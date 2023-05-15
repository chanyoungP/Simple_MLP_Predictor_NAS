### Using MLP network predictor predict final test acc of architecture in nasbench101
### and use predicted top 1 architecture
###      decode the architecture to real architecture
###      training on CIFAR10 (finetuning)
###      evaluate on CIFAR10  get final test acc
###      let's compare with predicted acc and real acc

#TODO :: MLP model 불러오기 --> 데이터셋 모두 통과시켜서 베스트 데이터 셋 저장하기 --> 아키텍처 변환하기 --> CIFAR10 학습하기 --> 평가하기
import logging
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Nb101Dataset
from model import MLP
from utils import  get_logger, reset_seed
from nasbench_pytorch.model import Network as NBNetwork  #NASbench architecture --> 실제 모델로
from nasbench_pytorch.model import ModelSpec
from cifar10 import prepare_dataset
from trainer import train,test
import os


device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu') #mac m1 mps gpu

def save_checkpoint(net, postfix='cifar10'):
    print('--- Saving Checkpoint ---')

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    torch.save(net.state_dict(), './checkpoint/ckpt_' + postfix + '.pt')

def reload_checkpoint(path, device=None):
    print('--- Reloading Checkpoint ---')

    assert os.path.isdir('checkpoint'), '[Error] No checkpoint directory found!'
    return torch.load(path, map_location=device)


def search_top():
    # predict arch final acc in nasbench by split option
    # split = 172 이면 172개만 예측해서 봄
    valid_splits = ["172", "334", "860", "91-172", "91-334", "91-860", "denoise-91", "denoise-80", "all"]
    parser = ArgumentParser()
    parser.add_argument("--eval_split", choices=valid_splits, default="all")  # "all"
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    args = parser.parse_args()

    reset_seed(args.seed)

    dataset_test = Nb101Dataset(split=args.eval_split)
    test_data_loader = DataLoader(dataset_test, batch_size=args.eval_batch_size)

    net = MLP()
    net.load_state_dict( #학습한 predictor MLP weight 불러오기
        torch.load("/Users/bagchan-yeong/Desktop/stat_project/neuralpredictor.pytorch/runs/MLP_predictor.pth"))
    net.to(device)

    top1_arch = None
    top_predicted = 0
    predicted_acc = []
    valid_acc_list = []

    net.eval()
    with torch.no_grad():
        for step, arch in enumerate(test_data_loader):
            num_vertices = arch['num_vertices'].to(torch.float32).to(device)
            adjacency = arch['adjacency'].to(torch.float32).to(device)
            operations = arch['operations'].to(torch.float32).to(device)
            mask = arch['mask'].to(torch.float32).to(device)

            num_vertices = torch.reshape(num_vertices, (-1, 1))
            adjacency = torch.flatten(adjacency, start_dim=1)
            operations = torch.flatten(operations, start_dim=1)


            test_acc = arch['val_acc'].to(torch.float32).to(device)
            test_acc = torch.reshape(test_acc, (-1, 1))
            valid_acc_list.append(arch['val_acc'].cpu().numpy())

            predict = net(num_vertices, adjacency, operations, mask)
            predicted_acc.append(predict.cpu().numpy())
            print("Pred value : {}".format(predict.cpu().numpy()))

            # top1 architecture save
            if top_predicted < predict.cpu().numpy():
                top_predicted = predict.cpu().numpy()
                top1_arch = arch

            print("STEP[%d/%d] PREDICTING FINAL ACC" % (step, len(test_data_loader)))
    print("prediction finish")
    plt.figure(figsize=(12, 6))
    plt.plot(np.reshape(predicted_acc, (-1, 1)), 'o')
    plt.xlabel("arch index")
    plt.ylabel("predicted acc")
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylim([0, 1])  # Y축의 범위: [ymin, ymax]
    plt.savefig("assets/predvalue.png", bbox_inches="tight")

    plt.figure(figsize=(12, 6))
    plt.hist(np.reshape(valid_acc_list, (-1, 1)),bins=np.arange(0,1,0.1),rwidth=1.5)
    plt.xlim([0, 1])
    plt.savefig("assets/dist_val_acc.png")
    print(np.mean(np.reshape(valid_acc_list, (-1, 1))))
    print(np.std(np.reshape(valid_acc_list, (-1, 1))))


    print("top1 architecture", top1_arch)
    print("top_pred value", top_predicted)
    print("top1_arch final test acc", top1_arch["val_acc"])

    return top1_arch,top_predicted,top1_arch["val_acc"]


def decode_ops(ops):
    # decode one-hot encoded operation to label
    # this is 필수 because transform network to torch network
    ops = ops.numpy().reshape(7,5)
    ops = np.argmax(ops,1)
    ops = ops-2
    LABEL2ID = {
        "input": -1,
        "output": -2,
        "conv3x3-bn-relu": 0,
        "conv1x1-bn-relu": 1,
        "maxpool3x3": 2
        }
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    decoded_ops = [ID2LABEL[id] for id in ops]

    return decoded_ops


def main():
    parser = ArgumentParser()
    parser.add_argument('--random_state', default=1, type=int, help='Random seed.')
    parser.add_argument('--data_root', default='./data/', type=str, help='Path where cifar will be downloaded.')
    parser.add_argument('--in_channels', default=3, type=int, help='Number of input channels.')
    parser.add_argument('--stem_out_channels', default=128, type=int, help='output channels of stem convolution')
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--test_batch_size', default=64, type=int, help='test set batch size')
    parser.add_argument('--epochs', default=108, type=int, help='#epochs of training')
    parser.add_argument('--validation_size', default=10000, type=int, help="Size of the validation set to split off.")
    parser.add_argument('--num_workers', default=0, type=int, help="Number of parallel workers for the train dataset.")
    parser.add_argument('--learning_rate', default=0.02, type=float, help='base learning rate')
    parser.add_argument('--lr_decay_method', default='COSINE_BY_STEP', type=str, help='learning decay method')
    parser.add_argument('--optimizer', default='rmsprop', type=str, help='Optimizer (sgd, rmsprop or rmsprop_tf)')
    parser.add_argument('--rmsprop_eps', default=1.0, type=float, help='RMSProp eps parameter.')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization weight')
    parser.add_argument('--grad_clip', default=5, type=float, help='gradient clipping')
    parser.add_argument('--grad_clip_off', default=False, type=bool, help='If True, turn off gradient clipping.')
    parser.add_argument('--batch_norm_momentum', default=0.997, type=float, help='Batch normalization momentum')
    parser.add_argument('--batch_norm_eps', default=1e-5, type=float, help='Batch normalization epsilon')
    parser.add_argument('--load_checkpoint', default='', type=str, help='Reload model from checkpoint')
    parser.add_argument('--num_labels', default=10, type=int, help='#classes')
    parser.add_argument('--device', default='cuda', type=str, help='Device for network training.')
    parser.add_argument('--print_freq', default=100, type=int, help='Batch print frequency.')
    parser.add_argument('--tf_like', default=False, type=bool,
                        help='If true, use same weight initialization as in the tensorflow version.')
    args = parser.parse_args()
    #CIFAR10 DATASET
    dataset = prepare_dataset(args.batch_size, test_batch_size=args.test_batch_size, root=args.data_root,
                              validation_size=args.validation_size, random_state=args.random_state,
                              set_global_seed=True, num_workers=args.num_workers)

    train_loader, test_loader, test_size = dataset['train'], dataset['test'], dataset['test_size']
    valid_loader = dataset['validation'] if args.validation_size > 0 else None

    top1_arch, top_predicted,top1_val_acc = search_top()

    ops = top1_arch['operations']
    adjacency = top1_arch['adjacency']
    adjacency = adjacency.numpy()
    decoded_ops = decode_ops(ops)
    adjacency = adjacency.reshape(7,7)

    spec = ModelSpec(adjacency, decoded_ops) #TOP1 arch --> torch model
    net = NBNetwork(spec, num_labels=args.num_labels, in_channels=args.in_channels,
                  stem_out_channels=args.stem_out_channels, num_stacks=args.num_stacks,
                  num_modules_per_stack=args.num_modules_per_stack,
                  momentum=args.batch_norm_momentum, eps=args.batch_norm_eps, tf_like=args.tf_like)

    net.to(device)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD
        optimizer_kwargs = {}
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop
        optimizer_kwargs = {'eps': args.rmsprop_eps}
    else:
        raise ValueError(f"Invalid optimizer {args.optimizer}, possible: SGD, RMSProp")

    optimizer = optimizer(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay, **optimizer_kwargs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    result = train(net, train_loader, loss=criterion, optimizer=optimizer, scheduler=scheduler,
                   grad_clip=args.grad_clip if not args.grad_clip_off else None,
                   num_epochs=args.epochs, num_validation=args.validation_size, validation_loader=valid_loader,
                   device=args.device, print_frequency=args.print_freq)

    last_epoch = {k: v[-1] for k, v in result.items() if len(v) > 0}
    print(f"Final train metrics: {last_epoch}")

    result = test(net, test_loader, loss=criterion, num_tests=test_size, device=args.device)
    print(f"\nFinal test metrics: {result}")

    save_checkpoint(net) # trainer 중간에 넣는게 좋음.










if __name__ == "__main__":
    main()

