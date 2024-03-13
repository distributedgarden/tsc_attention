import logging
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class MITBIHDataset(Dataset):
    """
    Description:
        - Dataset class for the MIT-BIH Arrhythmia Database.
        - This dataset contains ECG signals for 5 classes of arrhythmia.
        - The dataset is stored in a CSV file with each row containing a signal and its corresponding class label.

    Attributes
        - data (pd.DataFrame): DataFrame containing the dataset
        - X (np.ndarray): input data
        - y (np.ndarray): output data
    """

    def __init__(self, csv_file: str, verbose: bool = False):
        self.verbose = verbose

        self.data = pd.read_csv(csv_file, header=None)
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values

        if self.verbose:
            logging.info(f"Shape of input data: {self.X.shape}")
            logging.info(f"Shape of output data: {self.y.shape}")
            logging.info(f"Unique classes: {len(set(self.y))}")
            logging.info(f"Sample dataset: {self.data.head()}")

        self._preprocess_data()

    def _preprocess_data(self):
        """
        Description:
            - Preprocess the ECG signals.

        Args:
            - None

        Returns:
            - None
        """
        if self.verbose:
            logging.info("Preprocessing data...")

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Description:
            - Get the item at the given index.

        Args:
            - idx (int): index of the item to retrieve

        Returns:
            - tuple[torch.Tensor, torch.Tensor]: tuple of input and output tensors
        """
        return torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(
            1
        ), torch.tensor(self.y[idx], dtype=torch.long)


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
