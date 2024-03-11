import pytest
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.models.attention_lstm import (
    FlashAttentionLSTM,
)


pytestmark = pytest.mark.skip()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def num_heads():
    return 4


@pytest.fixture
def head_dim():
    return 32


@pytest.fixture
def dummy_dataset(sequence_length, input_size, batch_size, num_classes):
    """
    Creates a dummy dataset with specified dimensions.
    """
    # total number of samples - make sure it's a multiple of batch_size for simplicity
    total_samples = batch_size * 10

    # create random data and labels
    X = torch.randn(total_samples, sequence_length, input_size).to(torch.float16)
    y = torch.randint(0, num_classes, (total_samples,))

    return TensorDataset(X, y)


@pytest.fixture
def model_fixture(
    input_size, num_classes, hidden_size, num_heads, head_dim, num_layers, device
):
    return FlashAttentionLSTM(
        input_size, hidden_size, num_layers, num_heads, head_dim, num_classes
    ).to(device)


def test_output_shape(model_fixture, sample_input, batch_size, num_classes, device):
    sample_input = sample_input.to(torch.float16).to(device)
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
    hidden_size,
    sequence_length,
    input_size,
    batch_size,
    device,
):
    sample_input = (
        torch.randn(batch_size, sequence_length, input_size)
        .to(torch.float16)
        .to(device)
    )
    output = model_fixture(sample_input)

    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Output shape for batch size {batch_size} is not as expected"


def test_model_training_loop_produces_expected_output_shape_given_expected_input(
    model_fixture,
    num_classes,
    hidden_size,
    sequence_length,
    input_size,
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
