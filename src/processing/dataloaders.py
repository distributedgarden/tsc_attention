import logging

from torch.utils.data import DataLoader
from src.processing.datasets import MITBIHDataset, PTBXLDataset


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def mitbih_data_loaders(
    train_csv_file: str, test_csv_file: str, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """
    Description:
        - Create data loaders for the training and test datasets.

    Usage:
        >>> train_loader, test_loader = create_data_loaders(
                "mitbih_train.csv",
                "mitbih_test.csv",
                batch_size
            )
        >>> X_train, y_train = next(iter(train_loader))

    Args:
        - batch_size (int): batch size for training

    Returns:
        - train_loader (DataLoader): training data loader
        - test_loader (DataLoader): test data loader
    """
    train_dataset = MITBIHDataset(train_csv_file)
    test_dataset = MITBIHDataset(test_csv_file)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def ptbxl_data_loaders(
    path: str, batch_size: int, test_fold: int = 10, sampling_rate: int = 100
):
    """
    Description:
        - Create data loaders for the training and test datasets.

    Usage:
    >>> train_loader, test_loader = ptbxl_data_loaders(path, batch_size)
    >>> X_train, y_train = next(iter(train_loader))

    Args:
        - path (str): path to the PTB-XL dataset
        - batch_size (int): batch size for training and testing
        - test_fold (int): fold number to use for testing (default: 10)
        - sampling_rate (int): sampling rate of the ECG signals (default: 100)

    Returns:
        - train_loader (DataLoader): training data loader
        - test_loader (DataLoader): test data loader
    """
    train_dataset = PTBXLDataset(path, sampling_rate, test_fold, train=True)
    test_dataset = PTBXLDataset(path, sampling_rate, test_fold, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
