import logging
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot
from IPython.core.display import HTML
from itertools import cycle
from functools import partial


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def evaluate(
    trained_model: nn.Module,
    test_loader: DataLoader,
    attention: bool = False,
) -> tuple:
    """
    Description:
        - Evaluate the model on the test set and return various performance metrics.

    Args:
        - trained_model (nn.Module): The PyTorch model to be evaluated.
        - test_loader (DataLoader): The DataLoader containing the test dataset.
        - attention (bool): Flag to indicate if attention weights should be returned.

    Returns:
        - Tuple containing accuracy, precision, recall, F1 score, incorrect indexes, and attention maps.
    """
    # evaluate
    trained_model.eval()

    true_labels = []
    predicted_labels = []
    incorrect_indexes = []
    attention_maps = []
    index = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            device = next(trained_model.parameters()).device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = trained_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels.append(labels)
            predicted_labels.append(predicted)

            if attention:
                attention_maps.append(trained_model.attention_weights.cpu().numpy())

            # collect incorrect predictions
            mismatches = predicted != labels
            incorrect_batch_indexes = np.where(mismatches.cpu().numpy())[0]
            incorrect_batch_indexes += (
                index  # adjust index to match global position in dataset
            )
            incorrect_indexes.extend(incorrect_batch_indexes.tolist())

            index += len(labels)  # update index

    # combine lists of results
    true_labels = torch.cat(true_labels).cpu().numpy()
    predicted_labels = torch.cat(predicted_labels).cpu().numpy()

    # metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average="weighted")
    recall = recall_score(true_labels, predicted_labels, average="weighted")
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    logging.info(f"Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1, incorrect_indexes, attention_maps


def plot_metrics(
    accuracies: list,
    precisions: list,
    recalls: list,
    f1_scores: list,
    losses: list,
    filename: str,
) -> None:
    """
    Description:
        - Plot the accuracy, precision, recall, F1 score, and loss metrics over epochs.

    Args:
        - accuracies (list): List of accuracy scores per epoch.
        - precisions (list): List of precision scores per epoch.
        - recalls (list): List of recall scores per epoch.
        - f1_scores (list): List of F1 scores per epoch.
        - losses (list): List of loss values per epoch.
        - filename (str): filename to use when saving the plot
    """
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(accuracies, label="Accuracy")
    plt.plot(precisions, label="Precision")
    plt.plot(recalls, label="Recall")
    plt.plot(f1_scores, label="F1 Score")
    plt.title("Metrics over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(losses, label="Loss", color="red")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.show()


def plot_incorrect_predictions(
    incorrect_indexes: list, test_loader: DataLoader, filename: str
) -> None:
    """
    Description:
        - Plot the frequency of label misclassification per epoch.

    Args:
        - incorrect_indexes (list): A list of lists, where each sublist contains indices of incorrect predictions for each epoch.
        - test_loader (DataLoader): The DataLoader containing the test dataset.
        - filename (str): filename to use when saving the plot
    """
    label_frequencies_per_epoch = []

    all_labels = []
    for _, labels in test_loader:
        all_labels.extend(labels.tolist())

    for epoch_indexes in incorrect_indexes:
        incorrect_labels = [all_labels[idx] for idx in epoch_indexes]
        labels, counts = np.unique(incorrect_labels, return_counts=True)
        label_frequencies_per_epoch.append(dict(zip(labels, counts)))

    all_possible_labels = set(all_labels)

    plot_data = {label: [] for label in all_possible_labels}
    for epoch_data in label_frequencies_per_epoch:
        for label in all_possible_labels:
            plot_data[label].append(epoch_data.get(label, 0))

    # plot
    plt.figure(figsize=(12, 8))
    for label, counts in plot_data.items():
        plt.plot(counts, label=f"Label {label}")

    plt.title("Frequency of Label Misclassification per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Frequency of Label Misclassification")
    plt.legend()
    plt.xticks(range(len(label_frequencies_per_epoch)))
    plt.grid(True)

    plt.savefig(f"{filename}.png")
    plt.show()


def visualize_model_graph(
    trained_model: nn.Module, input_tensor: torch.Tensor, file_name: str = "model_graph"
) -> None:
    """
    Description:
        - Visualize and save the computation graph of the model.

    Args:
        - trained_model (nn.Module): The PyTorch model to visualize.
        - input_tensor (torch.Tensor): A tensor representing a sample input to the model.
        - file_name (str): The name of the output file where the graph visualization will be saved.
    """
    device = next(trained_model.parameters()).device
    input_tensor = input_tensor.to(device)

    model_output = trained_model(input_tensor)
    graph = make_dot(model_output, params=dict(trained_model.named_parameters()))
    graph.render(file_name, format="png", cleanup=True)


def sample(
    trained_model: nn.Module, test_loader: DataLoader, num_samples: int = 5
) -> tuple:
    """
    Description:
        - Randomly sample a specified number of correctly and incorrectly classified instances
          from the test data, using the predictions made by the trained model.

    Args:
        - trained_model (nn.Module): The trained model used to classify instances.
        - test_loader (DataLoader): The DataLoader for the test set.
        - num_samples (int): The number of instances to sample from each of the correct and incorrect classifications.

    Returns:
        - tuple: Two lists containing the indices of the sampled correct and incorrect instances, respectively.
    """
    trained_model.eval()
    correct_indices = []
    incorrect_indices = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            device = next(trained_model.parameters()).device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = trained_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct_batch_indices = (
                (predicted == labels).nonzero(as_tuple=False).view(-1).tolist()
            )
            incorrect_batch_indices = (
                (predicted != labels).nonzero(as_tuple=False).view(-1).tolist()
            )

            # add batch offset to indices
            correct_indices.extend(
                [i * test_loader.batch_size + idx for idx in correct_batch_indices]
            )
            incorrect_indices.extend(
                [i * test_loader.batch_size + idx for idx in incorrect_batch_indices]
            )

    # randomly sample from the correct and incorrect instances
    sampled_correct = np.random.choice(
        correct_indices, min(num_samples, len(correct_indices)), replace=False
    )
    sampled_incorrect = np.random.choice(
        incorrect_indices, min(num_samples, len(incorrect_indices)), replace=False
    )

    return sampled_correct, sampled_incorrect


def batch_predict(trained_model: nn.Module, data: np.ndarray) -> np.ndarray:
    """
    Description:
        - Make a batch prediction using the trained model.

    Args:
        - trained_model (nn.Module): The PyTorch model to use for predictions.
        - data (np.ndarray): The input data for making predictions.

    Returns:
        - np.ndarray: The predicted probabilities for each class.
    """
    trained_model.eval()
    data_tensor = torch.tensor(data).float().unsqueeze(2)

    device = next(trained_model.parameters()).device
    data_tensor = data_tensor.to(device)

    with torch.no_grad():
        outputs = trained_model(data_tensor)

    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    return probabilities


def explain_instance(
    trained_model: nn.Module,
    explainer: lime.lime_tabular.LimeTabularExplainer,
    test_instance: np.ndarray,
    true_label: int,
    test_idx: int,
    y_train: np.ndarray,
    num_features: int = 10,
) -> str:
    """
    Description:
        - Generate a LIME explanation for a single instance and return the HTML representation.

    Args:
        - trained_model (nn.Module): The PyTorch model to explain.
        - explainer (LimeTabularExplainer): The LIME explainer instance.
        - test_instance (np.ndarray): The input data instance to explain.
        - true_label (int): The true label of the instance.
        - test_idx (int): The index of the test instance.
        - num_features (int): The number of features to include in the explanation.

    Returns:
        - str: The HTML representation of the explanation.
    """
    test_instance_tensor = (
        torch.tensor(test_instance).float().unsqueeze(0).unsqueeze(2)
    )  # Add batch and channel dimensions
    device = next(trained_model.parameters()).device
    test_instance_tensor = test_instance_tensor.to(device)

    trained_model.eval()
    with torch.no_grad():
        model_output = trained_model(test_instance_tensor)
        _, predicted_label = torch.max(model_output.data, 1)

    logging.info(
        f"Instance {test_idx}: Predicted Label: {predicted_label.item()}, True Label: {true_label}"
    )

    unique_classes = [i for i in range(len(np.unique(y_train)))]
    explanation = explainer.explain_instance(
        test_instance,
        partial(batch_predict, trained_model),
        labels=unique_classes,
        num_features=num_features,
    )

    return explanation.as_html()


def roc_plot(
    trained_model: nn.Module,
    x_data: np.ndarray,
    y_data: np.ndarray,
    num_classes: int,
    filename: str,
) -> None:
    """
    Description:
        - Generate and save a Receiver Operating Characteristic (ROC) plot for the trained model
          on multi-class classification.

    Args:
        - trained_model (nn.Module): The trained model used for prediction.
        - x_data (np.ndarray): Input features for generating model predictions.
        - y_data (np.ndarray): True labels for comparison with model predictions.
        - num_classes (int): The number of unique classes in the classification task.
        - filename (str): The file path (without extension) where the ROC plot will be saved.
    """
    # probabilities for each class
    test_probabilities = np.vstack(
        [batch_predict(trained_model, x_data[i : i + 1]) for i in range(len(x_data))]
    )

    # ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_data, test_probabilities[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # plot
    plt.figure(figsize=(8, 6))

    colors = cycle(["blue", "red", "green", "orange", "purple"])
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic for multi-class")
    plt.legend(loc="lower right")

    plt.savefig(f"{filename}.png")
    plt.show()


def generate_saliency_map(
    trained_model: nn.Module, input_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Description:
        - Generate a saliency map for the input tensor using the trained model.

    Args:
        - trained_model (nn.Module): The PyTorch model used for generating the saliency map.
        - input_tensor (torch.Tensor): The input data tensor for which the saliency map is generated.

    Returns:
        - torch.Tensor: The generated saliency map.
    """
    device = next(trained_model.parameters()).device
    input_tensor = input_tensor.to(device)

    trained_model.eval()

    # forward pass
    output = trained_model(input_tensor)

    output_idx = output.max(1)[1].item()

    # temporarily enable training mode for backward pass
    trained_model.train()
    trained_model.zero_grad()
    output[0, output_idx].backward()
    saliency = input_tensor.grad.data.abs().squeeze()

    trained_model.eval()

    return saliency


def saliency_maps(trained_model: nn.Module, indexes: list, filename: str) -> None:
    """
    Description:
        - Generate and save saliency maps for the specified instances.

    Args:
        - trained_model (nn.Module): The PyTorch model used for generating the saliency maps.
        - indexes (list): The list of instance indices for which to generate saliency maps.
        - filename (str): name for the saved file
    """
    for idx in indexes:
        input_tensor = X_test_tensor[idx : idx + 1]  # Selecting the instance
        saliency = generate_saliency_map(trained_model, input_tensor)

        # Plotting the saliency map
        plt.figure()
        plt.plot(saliency.numpy())
        plt.title(f"Saliency Map for Instance {idx}")
        plt.xlabel("Feature")
        plt.ylabel("Importance")

        plt.savefig(f"{filename}_{idx}.png")
        plt.show()


def overlay_saliency_maps(
    trained_model: nn.Module, data_tensor: torch.Tensor, indexes: list, filename: str
) -> None:
    """
    Description:
        - Overlay the saliency map on the original data instances and save the resulting plot.
          This function iterates over given indexes to plot and save each instance's saliency map.

    Args:
        - trained_model (nn.Module): The trained PyTorch model to generate saliency maps.
        - data_tensor (torch.Tensor): The tensor containing the dataset instances.
        - indexes (list): A list of indices to generate saliency maps for.
        - filename (str): Base name for the saved plot files.
    """
    for idx in indexes:
        # Get the original instance data and saliency map
        original_instance = data_tensor[idx].numpy().squeeze()
        input_tensor = data_tensor[idx : idx + 1]

        saliency = generate_saliency_map(trained_model, input_tensor).numpy()

        # Normalize the saliency map for overlay
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        # Plotting the original instance
        plt.figure(figsize=(10, 4))
        plt.plot(original_instance, label="Original Instance")

        # Overlaying the saliency map
        plt.plot(
            saliency, label="Saliency Overlay", color="red", alpha=0.5
        )  # Use 'color' instead of 'cmap'

        plt.title(f"Saliency Overlay for Instance {idx}")
        plt.xlabel("Time Point")
        plt.ylabel("Amplitude / Importance")
        plt.legend()

        # Save the overlay plot
        plt.savefig(f"saliency_overlay_{filename}_{idx}.png")
        plt.show()
