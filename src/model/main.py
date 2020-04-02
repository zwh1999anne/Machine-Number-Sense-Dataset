# coding: utf-8 -*-


import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange

import criteria
import networks
from utils import dataset

torch.backends.cudnn.benchmark = True


def check_paths(args):
    try:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        new_log_dir = os.path.join(args.log_dir, time.ctime().replace(" ", "-"))
        args.log_dir = new_log_dir
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
    except OSError as e:
        print e
        sys.exit(1)


def train(args, device):
    def train_epoch(epoch, steps):
        model.train()
        loss_avg = 0.0
        acc_avg = 0.0
        counter = 0
        train_loader_iter = iter(train_loader)
        for _ in trange(len(train_loader_iter)):
            steps += 1
            counter += 1
            images, targets = next(train_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            loss = criterion(model_output, targets)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, targets)
            acc_avg += acc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Train Loss", loss.item(), steps)
            writer.add_scalar("Train Acc", acc.item(), steps)
        print "Epoch {}, Total Iter: {}, Train Avg Loss: {:.6f}".format(epoch, counter, loss_avg / float(counter))

        return steps

    def validate_epoch(epoch, steps):
        model.eval()
        loss_avg = 0.0
        acc_avg = 0.0
        counter = 0
        val_loader_iter = iter(val_loader)
        for _ in trange(len(val_loader_iter)):
            counter += 1
            images, targets = next(val_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            loss = criterion(model_output, targets)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, targets)
            acc_avg += acc.item()
        writer.add_scalar("Valid Avg Loss", loss_avg / float(counter), steps)
        writer.add_scalar("Valid Avg Acc", acc_avg / float(counter), steps)
        print "Epoch {}, Valid Avg Loss: {:.6f}, Valid Avg Acc: {:.4f}".format(epoch, loss_avg / float(counter), acc_avg / float(counter))
    
    def test_epoch(epoch, steps):
        model.eval()
        loss_avg = 0.0
        acc_avg = 0.0
        counter = 0
        test_loader_iter = iter(test_loader)
        for _ in trange(len(test_loader_iter)):
            counter += 1
            images, targets = next(test_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            loss = criterion(model_output, targets)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, targets)
            acc_avg += acc.item()
        writer.add_scalar("Test Avg Loss", loss_avg / float(counter), steps)
        writer.add_scalar("Test Avg Acc", acc_avg / float(counter), steps)
        print "Epoch {}, Test  Avg Loss: {:.6f}, Test  Avg Acc: {:.4f}".format(epoch, loss_avg / float(counter), acc_avg / float(counter))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = getattr(networks, args.model_name)()
    if args.cuda and args.multigpu and torch.cuda.device_count() > 1:
        print "Running the model on {} GPUs".format(torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], args.lr, weight_decay=args.weight_decay)

    if args.norm:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = None

    train_set = dataset(os.path.join(args.dataset, "train_set"), args.img_size, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_set = dataset(os.path.join(args.dataset, "val_set"), args.img_size, transform=transform)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)
    test_set = dataset(os.path.join(args.dataset, "test_set"), args.img_size, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    
    criterion = getattr(criteria, "cross_entropy")

    writer = SummaryWriter(args.log_dir)

    total_steps = 0

    for epoch in xrange(args.epochs):
        total_steps = train_epoch(epoch, total_steps)
        with torch.no_grad():
            validate_epoch(epoch, total_steps)
            test_epoch(epoch, total_steps)
        
        # save checkpoint    
        model.eval().cpu()
        ckpt_model_name = "epoch_{}_batch_{}_seed_{}_model_{}_norm_{}_lr_{}_l2_{}.pth".format(
            epoch, 
            args.batch_size, 
            args.seed,
            args.model_name, 
            args.norm, 
            args.lr, 
            args.weight_decay)
        ckpt_file_path = os.path.join(args.checkpoint_dir, ckpt_model_name)
        torch.save(model.state_dict(), ckpt_file_path)
        model.to(device)
    
    # save final model
    model.eval().cpu()
    save_model_name = "Final_epoch_{}_batch_{}_seed_{}_model_{}_norm_{}_lr_{}_l2_{}.pth".format(
        epoch, 
        args.batch_size, 
        args.seed,
        args.model_name, 
        args.norm, 
        args.lr, 
        args.weight_decay)
    save_file_path = os.path.join(args.save_dir, save_model_name)
    torch.save(model.state_dict(), save_file_path)

    print "Done. Model saved."


def test(args, device):
    def test_epoch():
        model.eval()
        correct_avg = 0.0
        test_loader_iter = iter(test_loader)
        for _ in range(len(test_loader_iter)):
            images, targets = next(test_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            correct_num = criteria.calculate_correct(model_output, targets)
            correct_avg += correct_num.item()
        acc = correct_avg / float(test_set_size)
        print "Test Avg Acc: {:.4f}".format(correct_avg / float(test_set_size))
        return acc
    
    if args.norm:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = None

    model = getattr(networks, args.model_name)()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    # name_combinations = [["combination", "holistic"], 
    #                      ["combination", "analytical"],
    #                      ["permutation", "holistic"],
    #                      ["permutation", "analytical"],
    #                      ["segmentation", "holistic"],
    #                      ["segmentation", "analytical"]]
    # for name_combination in name_combinations:    
    #     print "Evaluating {}".format("_".join(name_combination))
    #     test_set = dataset(os.path.join(args.dataset, "test_set"), args.img_size, transform=transform, names=name_combination)
    #     test_set_size = len(test_set)
    #     print test_set_size
    #     test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    #     with torch.no_grad():
    #         test_epoch()
    
    two_int = [["combination", "circle_overlap"], 
               ["segmentation", "circle_2"]]
    three_int = [["combination", "circle_tangent"]]
    four_int = [["permutation", "circle_line"],
                ["segmentation", "circle_4"]]
    six_int = [["permutation", "circle_triangle"],
               ["segmentation", "circle_6"]]
    eight_int = [["combination", "circle_include"],
                 ["permutation", "circle_cross"],
                 ["permutation", "circle_square"],
                 ["permutation", "circle_circle"],
                 ["segmentation", "circle_8"]]
    number_combinations = [two_int, three_int, four_int, six_int, eight_int]
    for number_combination in number_combinations:
        total_acc = 0.0
        total_num = 0
        for name_combination in number_combination:
            acc = 0.0
            print "Evaluating {}".format("_".join(name_combination))
            test_set = dataset(os.path.join(args.dataset, "test_set"), args.img_size, transform=transform, names=name_combination)
            test_set_size = len(test_set)
            print test_set_size
            test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
            with torch.no_grad():
                acc = test_epoch()
            total_acc += acc * test_set_size
            total_num += test_set_size
        print total_acc / total_num


def main(): 
    main_arg_parser = argparse.ArgumentParser(description="Machine Number Sense")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    
    train_arg_parser = subparsers.add_parser("train", help="parser for training")
    train_arg_parser.add_argument("--epochs", type=int, default=200,
                                  help="the number of training epochs")
    train_arg_parser.add_argument("--batch-size", type=int, default=32,
                                  help="size of batch")
    train_arg_parser.add_argument("--seed", type=int, default=1234,
                                  help="random number seed")
    train_arg_parser.add_argument("--device", type=int, default=0,
                                  help="device index for GPU; if GPU unavailable, leave it as default")
    train_arg_parser.add_argument("--num-workers", type=int, default=16,
                                  help="number of workers for data loader")
    train_arg_parser.add_argument("--dataset", type=str, default="../Datasets/IQMath/",
                                  help="dataset path")
    train_arg_parser.add_argument("--checkpoint-dir", type=str, default="./experiments/ckpt/",
                                  help="checkpoint save path")
    train_arg_parser.add_argument("--save-dir", type=str, default="./experiments/save/",
                                  help="final model save path")
    train_arg_parser.add_argument("--log-dir", type=str, default="./experiments/log/",
                                  help="log save path")
    train_arg_parser.add_argument("--img-size", type=int, default=80,
                                  help="image size for training")
    train_arg_parser.add_argument("--lr", type=float, default=0.95e-4,
                                  help="learning rate")
    train_arg_parser.add_argument("--weight-decay", type=float, default=0.0,
                                  help="weight decay of optimizer, same as l2 reg")
    train_arg_parser.add_argument("--model-name", type=str, required=True,
                                  help="name of the model")
    train_arg_parser.add_argument("--multigpu", type=int, default=0,
                                  help="whether to use multi gpu")
    train_arg_parser.add_argument("--norm", type=int, default=0,
                                  help="whether to normalize input to [0, 1]")
    
    test_arg_parser = subparsers.add_parser("test", help="parser for testing")
    test_arg_parser.add_argument("--batch-size", type=int, default=32,
                                 help="size of batch")
    test_arg_parser.add_argument("--device", type=int, default=0,
                                 help="device index for GPU; if GPU unavailable, leave it as default")
    test_arg_parser.add_argument("--num-workers", type=int, default=16,
                                 help="number of workers for data loader")
    test_arg_parser.add_argument("--dataset", type=str, default="/home/chizhang/Datasets/IQMath",
                                 help="dataset path")
    test_arg_parser.add_argument("--model-name", type=str, required=True,
                                 help="name of the model")
    test_arg_parser.add_argument("--model-path", type=str, required=True,
                                 help="path to a trained model")
    test_arg_parser.add_argument("--img-size", type=int, default=80,
                                 help="image size for training")
    test_arg_parser.add_argument("--norm", type=int, default=0,
                                 help="whether to normalize input to [0, 1]")

    args = main_arg_parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    if args.subcommand is None:
        print "ERROR: Specify train or test"
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args, device)
    elif args.subcommand == "test":
        test(args, device)
    else:
        print "ERROR: Unknown subcommand"
        sys.exit(1)


if __name__ == "__main__":
    main()
