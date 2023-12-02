import pytest
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.models.attention_lstm_fcn import (
    AttentionLSTMFCN,
)


@pytest.fixture
def model_fixture(input_size, num_classes, hidden_size, num_layers):
    return AttentionLSTMFCN(
        input_size, num_classes, hidden_size=hidden_size, num_layers=num_layers
    )


def test_output_shape(model_fixture, sample_input, batch_size, num_classes):
    output = model_fixture(sample_input)
    assert output.shape == (batch_size, num_classes), "Output shape is not as expected"


@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])  # Different batch sizes
def test_model_with_various_batch_sizes(
    model_fixture, num_classes, hidden_size, sequence_length, input_size, batch_size
):
    sample_input = torch.randn(batch_size, sequence_length, input_size)

    output = model_fixture(sample_input)
    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Output shape for batch size {batch_size} is not as expected"


def test_model_training_loop(
    model_fixture,
    num_classes,
    hidden_size,
    sequence_length,
    input_size,
    batch_size,
    dummy_dataset,
):
    criterion = CrossEntropyLoss()
    optimizer = Adam(model_fixture.parameters(), lr=0.001)

    data_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

    model_fixture.train()
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model_fixture(inputs)

        assert outputs.shape == (
            inputs.size(0),
            num_classes,
        ), "Output shape is not as expected"

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
