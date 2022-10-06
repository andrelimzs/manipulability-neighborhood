#!/usr/bin/env python

import numpy as np
import time
from tqdm import tqdm, trange
import pickle

import torch
from torch import nn

import matplotlib.pyplot as plt

from differentiable_robot_model.robot_model import DifferentiableRobotModel, DifferentiableFrankaPanda

from Manipulability import *

def read_dataset(dataset, batch_size):
    # Read dataset
    dataset = np.load(dataset)
    X_train = torch.from_numpy(dataset['X_train'])
    y_train = torch.from_numpy(dataset['y_train'])
    X_valid = torch.from_numpy(dataset['X_valid'])
    y_valid = torch.from_numpy(dataset['y_valid'])
    print(f"{len(X_train)} training | {len(X_valid)} validation")

    # Split into batches
    X_train = torch.split(X_train, batch_size)
    y_train = torch.split(y_train, batch_size)

    return X_train, y_train, X_valid, y_valid

# Define model
class MLP(nn.Module):
    def __init__(self, depth, width):
        super(MLP, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(21, width),
            # nn.BatchNorm1d(width),
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width,width),
                # nn.BatchNorm1d(width),
                nn.ReLU(),
            ) for _ in range(depth)
        ])
        self.output =  nn.Linear(width,1)
    
    def forward(self, x):
        x = torch.concat([x, torch.sin(x), torch.cos(x)], dim=1)
        x = self.input(x)
        for i in range(0, len(self.layers), 2):
            x2 = self.layers[i](x)
            x2 = self.layers[i+1](x2)
            x = x + x2
        x = self.output(x)
        return x.view(-1)

def training_loop(model, num_epochs, patience,
                  X_train, y_train, X_valid, y_valid,
                  train_losses=[], valid_losses=[]):
    tic = time.time()
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Early stopping
    best_loss = np.inf
    num_of_worse_losses = 0

    print("Starting training")
    # Train the neural network
    for epoch in trange(num_epochs):
        
        # for i in range(dataset_size // batch_size + 1):
        for (X,y) in zip(X_train, y_train):
            model.train()
            # Reset gradients
            optimizer.zero_grad()
            # Forward propagation through the network
            out = model(X)
            # Calculate the loss
            loss = criterion(out, y)
            # Backpropagation
            loss.backward()
            # Update the parameters
            optimizer.step()
        
        # Track training loss
        train_losses.append(loss.item())
        
        # Track validation loss
        with torch.no_grad():
            model.eval()
            out = model(X_valid)
            loss = criterion(out, y_valid)

        validation_loss = loss.item()
        valid_losses.append(validation_loss)
            
        # Early stopping
        if patience > 0:
            if validation_loss > best_loss:
                num_of_worse_losses += 1
            else:
                num_of_worse_losses = 0
                best_loss = validation_loss
            
            if num_of_worse_losses > patience:
                print(f"Early stopping at {epoch} ({num_of_worse_losses})")
                break

    toc = time.time() 
    print(f"Trained for {toc-tic:.0f} s")

    return model, train_losses, valid_losses

def plot_loss_curves(train_losses, valid_losses, save_plot='loss_curve.png'):
    fig = plt.figure()
    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.title("Losses")
    plt.grid()
    if save_plot:
        plt.savefig(save_plot, dpi=300)

if __name__ == '__main__':
    # Get arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate manipulability neighborhood dataset')
    parser.add_argument("--data_dir", type=str, default="Data/")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset to train with", default='dataset.npz')
    parser.add_argument("-b", "--batch_size", type=int, help="Training batch size", default=32)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train for", default=100)
    parser.add_argument("-m", "--model", type=str, help="Model to load", default=None)
    parser.add_argument("-p", "--patience", type=int, help="Early stopping patience", default=3)
    parser.add_argument("--plot_loss", type=str, help="Model to load", default=False)
    args = parser.parse_args()

    # Read dataset
    X_train, y_train, X_valid, y_valid = read_dataset(args.data_dir + args.dataset, args.batch_size)

    # Instantiate model
    device = 'cpu'
    depth = 16
    width = 50
    model = MLP(depth, width).to(device)
    print(f"depth = {depth}, width = {width}")

    # # Print model summary
    # print(model)

    # Load model if specified
    if args.model:
        print(f"Loaded model from {args.model}")
        model.load_state_dict(torch.load(f"{args.model}.pt"))
        (train_losses, valid_losses) = pickle.load(open(f"{args.model}.losses.p", "rb"))
        print(f"Starting training from epoch {len(train_losses)}")
    else:
        train_losses = []
        valid_losses = []

    # Train for N epochs
    model, train_losses, valid_losses = training_loop(
        model, args.epochs, args.patience,
        X_train, y_train, X_valid, y_valid,
        train_losses, valid_losses
    )

    print(f"Validation loss: {valid_losses[-1]:0.1e}")

    # Save model
    torch.save(model.state_dict(), f"{args.model}.pt")

    # Save train/valid losses as well
    pickle.dump((train_losses, valid_losses), open(f"{args.model}.losses.p", "wb"))
    pickle.dump((depth, width), open(f"{args.model}.config.p", "wb"))

    # Save loss curve figure
    plot_loss_curves(train_losses, valid_losses)

    if args.plot_loss:
        plt.show()