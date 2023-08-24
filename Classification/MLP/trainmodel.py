"""
DESCRIPTION: training of deep learning model.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from typing import Tuple

from torch import no_grad
from torch.cuda import is_available
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from Classification.MLP.focaloss import FocalLoss

# CUDA FLAG
cuda_flag = is_available()


# MODEL TRAINING
def train_model(*, model: Module, optimizer: Optimizer, loss_function: FocalLoss, loader_train: DataLoader,
                loader_val: DataLoader, maximum_epochs: int) -> Tuple[Module, list, list]:
    # Memory allocation
    loss_train_epochs = []
    loss_val_epochs = []

    # Cuda allocation model
    if cuda_flag:
        model.cuda()

    # Iteration across epochs
    for epoch in range(maximum_epochs):
        # Memory allocation
        loss_train_batches = []
        loss_val_batches = []

        # Model setting in training mode
        model.train()

        # Iteration across batches (training)
        for batch_train in loader_train:
            # Data extraction
            features_train = batch_train['features']
            labels_train = batch_train['label']

            # Cuda allocation
            if cuda_flag:
                features_train = features_train.cuda()
                labels_train = labels_train.cuda()

            # Forward propagation
            probs_train = model.forward(features=features_train)

            # Loss calculation
            loss_train_batch = loss_function.calculate_loss(labels_train, probs_train)

            # Parameter updating
            optimizer.zero_grad()
            loss_train_batch.backward()
            optimizer.step()

            # Loss epoch updating
            loss_train_batches.append(loss_train_batch.item())

        # Model setting in evaluation mode
        model.eval()

        # Iteration across batches (validation)
        with no_grad():
            for batch_val in loader_val:
                # Data extraction
                features_val = batch_val['features']
                labels_val = batch_val['label']

                # Cuda allocation
                if cuda_flag:
                    features_val = features_val.cuda()
                    labels_val = labels_val.cuda()

                # Forward propagation
                probs_val = model.forward(features=features_val)

                # Loss calculation
                loss_val_batch = loss_function.calculate_loss(labels_val, probs_val)

                # Loss epoch updating
                loss_val_batches.append(loss_val_batch.item())

        # Aggregation
        loss_train_epoch = sum(loss_train_batches) / len(loss_train_batches)
        loss_val_epoch = sum(loss_val_batches) / len(loss_val_batches)

        # Arrangement
        loss_train_epochs.append(loss_train_epoch)
        loss_val_epochs.append(loss_val_epoch)

    # Output
    return model, loss_train_epochs, loss_val_epochs
