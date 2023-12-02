import pytest
import torch
from src.models.attention_lstm_fcn import (
    AttentionLSTMFCN,
)


@pytest.fixture
def num_classes():
    return 5


@pytest.fixture
def hidden_size():
    return 128


@pytest.fixture
def num_layers():
    return 2


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
def model_fixture(input_size, num_classes, hidden_size, num_layers):
    return AttentionLSTMFCN(
        input_size, num_classes, hidden_size=hidden_size, num_layers=num_layers
    )


@pytest.fixture
def sample_input(batch_size, sequence_length, input_size):
    """
    [batch_size, sequence_length, input_size]
    """
    return torch.randn(batch_size, sequence_length, input_size)


def test_output_shape(model_fixture, sample_input):
    output = model_fixture(sample_input)
    assert output.shape == (32, 5), "Output shape is not as expected"
