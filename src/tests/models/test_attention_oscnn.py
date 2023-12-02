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
def model_fixture(num_classes, hidden_size):
    return AttentionOSCNN(num_classes=num_classes, attention_size=hidden_size)


@pytest.fixture
def sample_input(batch_size, sequence_length, input_size):
    """
    [batch_size, sequence_length, input_size]
    """
    return torch.randn(batch_size, sequence_length, input_size)


def test_output_shape(model_fixture, sample_input, batch_size, num_classes):
    output = model_fixture(sample_input)
    assert output.shape == (batch_size, num_classes), "Output shape is not as expected"


@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])  # Different batch sizes
def test_model_with_various_batch_sizes(
    num_classes, hidden_size, sequence_length, input_size, batch_size
):
    model = AttentionOSCNN(num_classes=num_classes, attention_size=hidden_size)
    sample_input = torch.randn(batch_size, sequence_length, input_size)

    output = model(sample_input)
    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Output shape for batch size {batch_size} is not as expected"


def test_model_training_loop(
    num_classes, hidden_size, sequence_length, input_size, batch_size, dummy_dataset
):
    model = AttentionOSCNN(num_classes=num_classes, attention_size=hidden_size)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    data_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        assert outputs.shape == (
            inputs.size(0),
            num_classes,
        ), "Output shape is not as expected"

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
