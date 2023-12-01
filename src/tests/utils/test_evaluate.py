import pytest
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.utils import evaluate


# Mock a simple model for testing
class MockModel(nn.Module):
    def __init__(self, attention=False):
        super(MockModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # assuming binary classification
        self.attention = attention
        self.attention_weights = None

    def forward(self, x):
        if self.attention:
            self.attention_weights = torch.rand(
                x.size(0), x.size(1)
            )  # mock attention weights
        return self.fc(x)


# Fixture for the model
@pytest.fixture
def mock_model():
    return MockModel()


# Fixture for the test data loader
@pytest.fixture
def test_loader():
    X = torch.rand(100, 10)  # 100 samples, 10 features each
    y = torch.randint(0, 2, (100,))  # binary target
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10)


# Test evaluate function without attention
def test_evaluate_without_attention(mock_model, test_loader):
    accuracy, precision, recall, f1, incorrect_indexes, attention_maps = evaluate(
        mock_model, test_loader, attention=False
    )
    assert isinstance(accuracy, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    assert isinstance(incorrect_indexes, list)
    assert attention_maps == []


# Test evaluate function with attention
def test_evaluate_with_attention(mock_model, test_loader):
    mock_model_with_attention = MockModel(attention=True)
    accuracy, precision, recall, f1, incorrect_indexes, attention_maps = evaluate(
        mock_model_with_attention, test_loader, attention=True
    )
    assert isinstance(accuracy, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    assert isinstance(incorrect_indexes, list)
    assert len(attention_maps) > 0  # Attention maps should be populated
