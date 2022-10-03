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
    print(f"Dataset\n: Training {len(X_train)} samples | Validation {len(X_valid)} samples")

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
            nn.BatchNorm1d(width),
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width,width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
            ) for _ in range(depth)
        ])
        self.output =  nn.Linear(width,1)
    
    def forward(self, x):
        x = torch.concat([x, torch.sin(x), torch.cos(x)], dim=1)
        x = self.input(x)
        for i, l in enumerate(self.layers):
            x = l(x)
        x = self.output(x)
        return x.view(-1)

def training_loop(model, num_epochs, X_train, y_train, X_valid, y_valid):
    tic = time.time()
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Early stopping
    patience = 20
    best_loss = np.inf
    num_of_worse_losses = 0

    # Track batch loss
    train_losses = []
    valid_losses = []

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
    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.title("Losses")
    plt.grid()
    if save_plot:
        plt.savefig(save_plot)

if __name__ == '__main__':
    # Get arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate manipulability neighborhood dataset')
    parser.add_argument("--dataset", type=str, help="Dataset to train with", default='Data/dataset.npz')
    parser.add_argument("-b", "--batch_size", type=int, help="Training batch size", default=32)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train for", default=100)
    parser.add_argument("-m", "--model", type=str, help="Model to load", default=None)
    args = parser.parse_args()

    # Read dataset
    X_train, y_train, X_valid, y_valid = read_dataset(args.dataset, args.batch_size)

    # Instantiate model
    device = 'cpu'
    depth = 4
    width = 50
    model = MLP(depth, width).to(device)

    # Load model if specified
    if args.model:
        print(f"Loaded model from {args.model}")
        model.load_state_dict(torch.load(args.model))
        (train_losses, valid_losses) = pickle.load(open("model.losses.p", "rb"))

    # Train for N epochs
    model, train_losses, valid_losses = training_loop(model, args.epochs, X_train, y_train, X_valid, y_valid)

    print(f"Validation loss: {valid_losses[-1]:0.1e}")

    # Save model
    torch.save(model.state_dict(), "model.pt")

    # Save train/valid losses as well
    pickle.dump((train_losses, valid_losses), open("model.losses.p", "wb"))

    # Save loss curve figure
    plot_loss_curves(train_losses, valid_losses)