#!/usr/bin/env python

import numpy as np
import time
from tqdm import tqdm, trange
import pickle

import torch
from torch import nn

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from differentiable_robot_model.robot_model import DifferentiableRobotModel, DifferentiableFrankaPanda

from Manipulability import *
from learn_manipulability import MLP

def plot_losses(train_losses, valid_losses):
    train = train_losses
    valid = valid_losses

    # train = np.log(train_losses)
    # valid = np.log(valid_losses)
    
    # Plot log curve
    fig = plt.figure()
    plt.plot(train, label='train')
    plt.plot(valid, label='valid')

    plt.title("Losses")
    plt.grid()
    plt.legend()
    plt.ylim(0, 0.002)

def plot_manipulability_surface(robot, ee_link, joint_config, filename=None, N=20):
    # Retrieve joint limits
    robot_limits = robot.get_joint_limits()
    upper_limit = torch.tensor([x['upper'] for x in robot_limits])
    lower_limit = torch.tensor([x['lower'] for x in robot_limits])

    # Convert joint_space config into regular grid to plot
    #   Set the config as a list of strs
    #   [ z (zero), l (linspace), r (random) ]
    joint_space = []
    space_indices = []
    for i, config in enumerate(joint_config):
        if config == 'z':
            joint_space.append(torch.zeros(1))
        elif config == 'l':
            joint_space.append(torch.linspace(lower_limit[i], upper_limit[i], N))
            space_indices.append(i)
        elif config == 'r':
            joint_space.append(torch.rand(1) * (upper_limit[i] - lower_limit[i]) + lower_limit[i])

    # Form X of shape (N,7)
    X = torch.cartesian_prod(*joint_space)

    # Compute groundtruth MN
    y_true = compute_manipulability_neighborhood(robot, ee_link, X)

    # Evaluate model
    with torch.no_grad():
        model.eval()
        y_pred = model(X)
        model.train()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Get x/y axes from joint_config
    xdata = X[:,space_indices[0]]
    ydata = X[:,space_indices[1]]

    # Surface plot for prediction
    ax.plot_trisurf(xdata, ydata, y_pred.cpu(),
        cmap='viridis',
        edgecolor=None,
        alpha=0.9,
    )
    # Scatter for true values
    ax.scatter(xdata, ydata, y_true.cpu(),
        cmap='viridis'
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("MN")
    
    # Save figures
    if filename:
        plt.savefig(filename, dpi=300)

if __name__ == '__main__':
    # Get arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate manipulability neighborhood dataset')
    parser.add_argument("-m", "--model", type=str, help="Model to load", default='model')
    parser.add_argument("--loss", action='store_true')
    args = parser.parse_args()

    # Define robot model
    robot = DifferentiableFrankaPanda()
    ee_link = "panda_virtual_ee_link"

    # Load model
    device = 'cpu'
    depth, width = pickle.load(open(f"{args.model}.config.p", "rb"))
    model = MLP(depth, width).to(device)
    model.load_state_dict(torch.load(f"{args.model}.pt"))

    # Plot loss curves
    if args.loss:
        train_losses, valid_losses = pickle.load(open(f"{args.model}.losses.p", "rb"))
        plot_losses(train_losses, valid_losses)

    # Plot MN over joints 2 and 4
    joint_config = ['z', 'l', 'r', 'l', 'z', 'z', 'z']
    plot_manipulability_surface(robot, ee_link, joint_config, 'surface1.png')

    # Plot MN over joints 3 and 6
    joint_config = ['z', 'z', 'l', 'r', 'z', 'l', 'z']
    plot_manipulability_surface(robot, ee_link, joint_config, 'surface2.png')

    # Show now
    plt.show()