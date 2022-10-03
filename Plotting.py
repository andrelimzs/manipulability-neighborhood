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

if __name__ == '__main__':
    # Get arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate manipulability neighborhood dataset')
    parser.add_argument("-m", "--model", type=str, help="Model to load", default='model.pt')
    args = parser.parse_args()

    # Define robot model
    robot = DifferentiableFrankaPanda()
    ee_link = "panda_virtual_ee_link"

    # Retrieve joint limits
    robot_limits = robot.get_joint_limits()
    upper_limit = torch.tensor([x['upper'] for x in robot_limits])
    lower_limit = torch.tensor([x['lower'] for x in robot_limits])

    # Load model
    device = 'cpu'
    depth = 8
    width = 50
    model = MLP(depth, width).to(device)
    model.load_state_dict(torch.load(args.model))

    # Plot
    # Form X of shape (N,7)
    X = torch.cartesian_prod(
        torch.zeros(1),
        torch.linspace(lower_limit[1], upper_limit[1], 20),
        torch.rand(1) * (upper_limit[2] - lower_limit[2]) + lower_limit[2],
        torch.linspace(lower_limit[3], upper_limit[3], 20),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
    )

    # Compute groundtruth MN
    y_true = compute_manipulability_neighborhood(robot, ee_link, X)

    # Evaluate model
    with torch.no_grad():
        model.eval()
        y_pred = model(X)
        model.train()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xdata = X[:,1]
    ydata = X[:,3]
    ax.plot_trisurf(xdata, ydata, y_pred.cpu(),
        cmap='viridis',
        edgecolor=None
    )
    ax.scatter(xdata, ydata, y_true.cpu(),
        cmap='viridis'
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("MN")

    plt.show()

    # Form X of shape (N,7)
    X = torch.cartesian_prod(
        torch.zeros(1),
        torch.zeros(1),
        torch.linspace(lower_limit[2], upper_limit[2], 20),
        torch.rand(1) * (upper_limit[3] - lower_limit[3]) + lower_limit[2],
        torch.zeros(1),
        torch.linspace(lower_limit[5], upper_limit[5], 20),
        torch.zeros(1),
    )

    # Compute groundtruth MN
    y_true = compute_manipulability_neighborhood(robot, ee_link, X)

    # Evaluate model
    with torch.no_grad():
        model.eval()
        y_pred = model(X)
        model.train()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xdata = X[:,2]
    ydata = X[:,5]
    ax.plot_trisurf(xdata, ydata, y_pred.cpu(),
        cmap='viridis',
        edgecolor=None
    )
    ax.scatter(xdata, ydata, y_true.cpu(),
        cmap='viridis'
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("MN")

    plt.show()