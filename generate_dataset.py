#!/usr/bin/env python

import numpy as np
import torch
import time
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from differentiable_robot_model.robot_model import DifferentiableRobotModel, DifferentiableFrankaPanda

from Manipulability import *

if __name__ == '__main__':
    # Get arguments
    import argparse

    parser = argparse.ArgumentParser(description='Generate manipulability neighborhood dataset')
    parser.add_argument("-trainN", type=int, help="Number of samples to generate", default=1000000)
    parser.add_argument("-validN", type=int, help="Number of samples to generate", default=20000)
    args = parser.parse_args()

    print(f"Generating {args.trainN} training samples and {args.validN} validation samples")

    # Define robot model
    robot = DifferentiableFrankaPanda()
    ee_link = "panda_virtual_ee_link"

    # Retrieve joint limits
    robot_limits = robot.get_joint_limits()
    upper_limit = torch.tensor([x['upper'] for x in robot_limits])
    lower_limit = torch.tensor([x['lower'] for x in robot_limits])

    tic = time.time()
    # Generate dataset
    X_train, y_train = generate_N_samples(robot, ee_link, upper_limit, lower_limit, args.trainN)
    X_valid, y_valid = generate_N_samples(robot, ee_link, upper_limit, lower_limit, args.validN)
    toc = time.time()
    print(f"Completed in {toc-tic:.1f} s")

    # Save to npz
    np.savez('Data/dataset.npz',
         X_train=X_train.numpy(),
         y_train=y_train.numpy(),
         X_valid=X_valid.numpy(),
         y_valid=y_valid.numpy()
    )