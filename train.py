'''Training MLP model to predict final acc of neural architecture in NASBENCH101 dataset '''
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
from utils import AverageMeter, AverageMeterGroup, get_logger, reset_seed, to_cuda
import h5py
from scipy.stats import kendalltau
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')


def accuracy_mse(predict, target, scale=100.):
    predict = Nb101Dataset.denormalize(predict.detach()) * scale
    target = Nb101Dataset.denormalize(target) * scale
    return F.mse_loss(predict, target)


def visualize_scatterplot(predict, target, scale=100.):
    def _scatter(x, y, subplot, threshold=None):
        plt.subplot(subplot)
        plt.grid(linestyle="--")
        plt.xlabel("Validation Accuracy")
        plt.ylabel("Prediction")
        plt.scatter(target, predict, s=1)
        if threshold:
            ax = plt.gca()
            ax.set_xlim(threshold, 95)
            ax.set_ylim(threshold, 95)
    predict = Nb101Dataset.denormalize(predict) * scale
    target = Nb101Dataset.denormalize(target) * scale
    plt.figure(figsize=(12, 6))
    _scatter(predict, target, 121)
    _scatter(predict, target, 122, threshold=90)
    plt.savefig("assets/scatterplot.png", bbox_inches="tight")
    plt.close()


def main():
    valid_splits = ["172", "334", "860", "91-172", "91-334", "91-860", "denoise-91", "denoise-80", "all"]
    parser = ArgumentParser()
    parser.add_argument("--train_split", choices=valid_splits, default="172") #172개 sampling 후 training 나머지 validation
    parser.add_argument("--eval_split", choices=valid_splits, default="all")
    parser.add_argument("--gcn_hidden", type=int, default=144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_batch_size", default=10, type=int)
    parser.add_argument("--eval_batch_size", default=1000, type=int)
    parser.add_argument("--epochs", default=300, type=int) #300
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--wd", "--weight_decay", default=1e-3, type=float)
    parser.add_argument("--train_print_freq", default=None, type=int)
    parser.add_argument("--eval_print_freq", default=10, type=int)
    parser.add_argument("--visualize", default=True, action="store_true")
    args = parser.parse_args()

    reset_seed(args.seed)

    dataset = Nb101Dataset(split=args.train_split)
    dataset_test = Nb101Dataset(split=args.eval_split)
    data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(dataset_test, batch_size=args.eval_batch_size)
    # net = NeuralPredictor(gcn_hidden=args.gcn_hidden)
    net = MLP()
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    logger = get_logger()

    net.train()
    predict_, target_ = [], []  # scatter 용도
    best_val = 999 #any point

    for epoch in range(args.epochs):
        meters = AverageMeterGroup()
        lr = optimizer.param_groups[0]["lr"]

        #epoch loss, epoch mse
        epoch_loss = 0
        epoch_mse = 0

        for step, batch in enumerate(data_loader):
            num_vertices = batch['num_vertices'].to(torch.float32).to(device)
            adjacency = batch['adjacency'].to(torch.float32).to(device)
            operations = batch['operations'].to(torch.float32).to(device)
            mask = batch['mask'].to(torch.float32).to(device)

            num_vertices = torch.reshape(num_vertices,(-1,1))
            adjacency = torch.flatten(adjacency,start_dim=1)
            operations = torch.flatten(operations,start_dim=1)


            val_acc = batch['val_acc'].to(torch.float32).to(device)
            test_acc = batch['test_acc']


            target = torch.reshape(val_acc,(-1,1))
            predict = net(num_vertices,adjacency,operations,mask)
            optimizer.zero_grad()
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            mse = accuracy_mse(predict, target)
            epoch_loss+=loss.item()
            epoch_mse+=mse.item()

            print("TRAINING PHASE EPOCH[%d/%d] Step[%d/%d] lr = %.3e LOSS : %d mse : %d "%(epoch+1,
                  args.epochs,step+1,len(data_loader),
                  lr,loss.item(),mse.item()))

            meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))
            # if (args.train_print_freq and step % args.train_print_freq == 0) or \
            #         step + 1 == len(data_loader):
            #     logger.info("Epoch [%d/%d] Step [%d/%d] lr = %.3e  %s",
            #                 epoch + 1, args.epochs, step + 1, len(data_loader), lr, meters)
                # writer.add_scalar("train_loss",meters.__getitem__("loss"))
                # writer.add_scalar("train_acc_mse",meters.__getitem__("mse"))
        lr_scheduler.step()
        writer.add_scalar("train_loss",epoch_loss/len(data_loader),epoch+1)
        writer.add_scalar("train_mse",epoch_mse/len(data_loader),epoch+1)
        print("EPOCH[%d/%d] LOSS : %d MSE : %d"%(epoch+1,args.epochs,epoch_loss/len(data_loader),
                                                 epoch_mse/len(data_loader)))

        net.eval()
        meters = AverageMeterGroup()
        epoch_val_loss = 0
        epoch_val_mse = 0
        with torch.no_grad():
            for step, batch in enumerate(test_data_loader):
                num_vertices = batch['num_vertices'].to(torch.float32).to(device)
                adjacency = batch['adjacency'].to(torch.float32).to(device)
                operations = batch['operations'].to(torch.float32).to(device)
                mask = batch['mask'].to(torch.float32).to(device)

                num_vertices = torch.reshape(num_vertices, (-1, 1))
                adjacency = torch.flatten(adjacency, start_dim=1)
                operations = torch.flatten(operations, start_dim=1)


                val_acc = batch['val_acc'].to(torch.float32).to(device)
                val_acc = torch.reshape(val_acc,(-1,1))
                test_acc = batch['test_acc']

                predict = net(num_vertices,adjacency,operations,mask)
                loss = criterion(predict,val_acc)
                mse = accuracy_mse(predict,val_acc)

                predict_.append(predict.cpu().numpy()) # scatter 용도
                target_.append(val_acc.cpu().numpy())  # scatter 용도

                epoch_val_loss+=loss.item()
                epoch_val_mse+=mse.item()
                meters.update({"loss": loss.item(),
                               "mse": mse.item()}, n=val_acc.size(0))
                #TODO:: save best model with small loss

                # print("VALIDATION PHASE EPOCH[%d/%d] Step[%d/%d] lr = %.3e LOSS : %d mse : %d "%(epoch + 1, args.epochs,
                #       step + 1,len(data_loader),lr,loss.item(),mse.item()))


                # if (args.eval_print_freq and step % args.eval_print_freq == 0) or \
                #         step % 10 == 0 or step + 1 == len(test_data_loader):
                #     logger.info("Evaluation Step [%d/%d]  %s", step + 1, len(test_data_loader), meters)
                    # writer.add_scalar("val_loss", meters.__getitem__("loss"))
                    # writer.add_scalar("val_acc_mse", meters.__getitem__("mse"))
            print("VAL EPOCH[%d/%d] LOSS : %d MSE : %d"%(epoch + 1, args.epochs, epoch_val_loss/len(test_data_loader),
                  epoch_val_mse/len(test_data_loader)))
        writer.add_scalar("val_loss",epoch_val_loss/len(test_data_loader),epoch+1)
        writer.add_scalar("val_mse",epoch_val_mse/len(test_data_loader),epoch+1)
        if epoch_val_loss/len(test_data_loader) < best_val:
            best_val = epoch_val_loss/len(test_data_loader)
            torch.save(net.state_dict(),'./runs/MLP_predictor.pth')

    predict_ = np.concatenate(predict_)
    target_ = np.concatenate(target_)
    logger.info("Kendalltau: %.6f", kendalltau(predict_, target_)[0])
    if args.visualize:
        visualize_scatterplot(predict_, target_)
    writer.close()


if __name__ == "__main__":
    main()
