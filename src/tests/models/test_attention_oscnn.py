import pytest
import torch
from src.models.attention_os_cnn import (
    AttentionOSCNN,
)  # Replace with the actual import for your model class


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
def attention_os_cnn_model(num_classes, hidden_size, sequence_length):
    return AttentionOSCNN(
        num_classes=num_classes,
        attention_size=sequence_length,
    )


@pytest.fixture
def sample_input(batch_size, sequence_length, input_size):
    """
    [batch_size, sequence_length, input_size]
    """
    return torch.randn(batch_size, sequence_length, input_size)


def test_attention_os_cnn_output_shape(attention_os_cnn_model, sample_input):
    output = attention_os_cnn_model(sample_input)
    assert output.shape == (32, 5), "Output shape is not as expected"


# def test_attention_os_cnn_with_without_intermediate(attention_os_cnn_model, sample_input):
#    original_output = attention_os_cnn_model(sample_input)
#
#    # bypass the intermediate layer
#    attention_os_cnn_model.intermediate = torch.nn.Identity()
#
#    # forward pass
#    modified_output = attention_os_cnn_model(sample_input)
#
#    assert torch.allclose(original_output, modified_output, atol=1e-6), "Outputs differ significantly with and without intermediate layer"
#
