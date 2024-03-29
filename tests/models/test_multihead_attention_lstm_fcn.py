import pytest
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.models.attention_lstm_fcn import (
    MultiheadAttentionLSTMFCN,
)


@pytest.fixture
def model_fixture(input_size, num_classes, hidden_size, num_heads, device):
    return MultiheadAttentionLSTMFCN(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_heads=num_heads,
    ).to(device)


def test_output_shape(model_fixture, sample_input, batch_size, num_classes, device):
    sample_input = sample_input.to(device)
    output = model_fixture(sample_input)

    assert output.shape == (batch_size, num_classes), "Output shape is not as expected"


@pytest.mark.parametrize(
    "batch_size",
    [1, 16, 32, 64],
    ids=["batch_size_1", "batch_size_16", "batch_size_32", "batch_size_64"],
)
def test_model_with_various_batch_sizes(
    model_fixture,
    num_classes,
    sequence_length,
    input_size,
    batch_size,
    device,
):
    sample_input = torch.randn(batch_size, sequence_length, input_size).to(device)
    output = model_fixture(sample_input)

    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Output shape for batch size {batch_size} is not as expected"


@pytest.mark.parametrize(
    "num_heads",
    [4, 8, 16, 2],
    ids=["4 heads, 32 dim", "8 heads, 16 dim", "16 heads, 8 dim", "2 heads, 64 dim"],
)
def test_multi_head_attention_dimensions(
    model_fixture,
    num_classes,
    sequence_length,
    input_size,
    batch_size,
    device,
):
    sample_input = torch.randn(batch_size, sequence_length, input_size).to(device)
    output = model_fixture(sample_input)

    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Output shape is not as expected"


def test_model_training_loop_produces_expected_output_shape_given_expected_input(
    model_fixture,
    num_classes,
    batch_size,
    dummy_dataset,
    device,
):
    criterion = CrossEntropyLoss()
    optimizer = Adam(model_fixture.parameters(), lr=0.001)

    data_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

    model_fixture.train()
    for inputs, labels in data_loader:
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_fixture(inputs)

        assert outputs.shape == (
            inputs.size(0),
            num_classes,
        ), "Output shape is not as expected"

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
