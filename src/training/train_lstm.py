import logging
import torch

from src.training.evaluate import evaluate


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def train(
    optimizer: torch.optim.Adam,
    criterion: torch.nn.CrossEntropyLoss,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    """
    Description:
        - Train the model using the given optimizer and criterion.

    Args:
        - optimizer (torch.optim.Adam): optimizer for training
        - criterion (torch.nn.CrossEntropyLoss): loss function
        - model (torch.nn.Module): model to train
        - train_loader (torch.utils.data.DataLoader): training data loader
        - test_loader (torch.utils.data.DataLoader): test data loader
        - device (torch.device): device to use for training

    Returns:
        - accuracies (List[float]): list of accuracy values for each epoch
        - precisions (List[float]): list of precision values for each epoch
        - recalls (List[float]): list of recall values for each epoch
        - f1_scores (List[float]): list of f1 score values for each epoch
        - losses (List[float]): list of loss values for each epoch
        - incorrect_indexes (List[List[int]]): list of incorrect indexes for each epoch
    """
    # store metrics for display
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    losses = []
    incorrect_indexes = []

    # train
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        # average loss for this epoch
        epoch_loss = total_loss / len(train_loader)
        losses.append(epoch_loss)

        # store metrics
        accuracy, precision, recall, f1, epoch_incorrect_indexes, _ = evaluate(
            model, test_loader
        )
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        incorrect_indexes.append(epoch_incorrect_indexes)

    return accuracies, precisions, recalls, f1_scores, losses, incorrect_indexes
