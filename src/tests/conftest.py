import pytest
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.models.attention_os_cnn import (
    AttentionOSCNN,
)


@pytest.fixture
def num_classes():
    return 5


@pytest.fixture
def hidden_size():
    return 128


@pytest.fixture
def batch_size():
    """
    the number of instances used in a training step
    """
    return 32


@pytest.fixture
def sequence_length():
    """
    the number of timesteps per instance
    """
    return 187


@pytest.fixture
def input_size():
    """
    the number of features per timestep
    """
    return 1


@pytest.fixture
def num_layers():
    return 2


@pytest.fixture
def dummy_dataset(sequence_length, input_size, batch_size, num_classes):
    """
    Creates a dummy dataset with specified dimensions.
    """
    # Total number of samples - make sure it's a multiple of batch_size for simplicity
    total_samples = batch_size * 10

    # Create random data and labels
    X = torch.randn(total_samples, sequence_length, input_size)
    y = torch.randint(0, num_classes, (total_samples,))

    return TensorDataset(X, y)


@pytest.fixture
def model_fixture():
    """
    example:
        return AttentionOSCNN(num_classes=num_classes)
    """
    raise ValueError("override model_fixture with a model")


@pytest.fixture
def sample_input(batch_size, sequence_length, input_size):
    """
    [batch_size, sequence_length, input_size]
    """
    return torch.randn(batch_size, sequence_length, input_size)
