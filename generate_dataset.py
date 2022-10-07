#!/usr/bin/env python

import numpy as np
import torch
import time
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from differentiable_robot_model.robot_model import DifferentiableRobotModel, DifferentiableFrankaPanda

from Manipulability import *

def generate_N_samples(robot, ee_link, upper_limit, lower_limit, N, num_neighbors):
    # soboleng = torch.quasirandom.SobolEngine(dimension=7)
    # q = soboleng.draw(N) * (upper_limit - lower_limit) + lower_limit

    # Generate N samples of the manipulability neighborhood
    q = torch.rand(N,7) * (upper_limit - lower_limit) + lower_limit

    # Return X(q) and y(MN)
    return q, compute_manipulability_neighborhood(robot, ee_link, q, num_neighbors)

if __name__ == '__main__':
    # Get arguments
    import argparse

    parser = argparse.ArgumentParser(description='Generate manipulability neighborhood dataset')
    parser.add_argument("--trainN", type=int, help="Number of samples to generate", default=1000000)
    parser.add_argument("--validN", type=int, help="Number of samples to generate", default=2000)
    parser.add_argument("-N", "--numNeighbors", type=int, help="Number of neighbors per axis", default=10)
    parser.add_argument("-o", '--output', type=str, help="File to save to", default='dataset.npz')
    args = parser.parse_args()

    print(f"Generating {args.trainN} training samples and {args.validN} validation samples")
    print(f"With {args.numNeighbors} neighbors")

    # Define robot model
    robot = DifferentiableFrankaPanda()
    ee_link = "panda_virtual_ee_link"

    # Retrieve joint limits
    robot_limits = robot.get_joint_limits()
    upper_limit = torch.tensor([x['upper'] for x in robot_limits])
    lower_limit = torch.tensor([x['lower'] for x in robot_limits])

    tic = time.time()
    # Generate dataset
    X_train, y_train = generate_N_samples(robot, ee_link, upper_limit, lower_limit, args.trainN, args.numNeighbors)
    X_valid, y_valid = generate_N_samples(robot, ee_link, upper_limit, lower_limit, args.validN, args.numNeighbors)
    toc = time.time()
    print(f"Completed in {toc-tic:.1f} s")

    # Save to npz
    np.savez(f'Data/{args.output}',
         X_train=X_train.numpy(),
         y_train=y_train.numpy(),
         X_valid=X_valid.numpy(),
         y_valid=y_valid.numpy()
    )