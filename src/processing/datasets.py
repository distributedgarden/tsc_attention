import logging
import pandas as pd
import numpy as np
import wfdb
import ast
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder
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


class PTBXLDataset(Dataset):
    """
    Description:
        - Dataset class for the PTB-XL ECG dataset.

        - The PTB-XL dataset contains electrocardiogram (ECG) records from patients, along with
          annotations of diagnostic statements and other metadata. This class provides a way to
          load and preprocess the dataset for machine learning tasks.

        - Key features of the dataset:
            - 21,837 clinical 12-lead ECG records from 18,885 patients
            - 10-second recordings sampled at 500 Hz or 100 Hz
            - Annotations of diagnostic statements, ECG statements, and signal quality
            - Metadata including patient demographics, recording information, and diagnostic classes

        - The dataset is split into training and testing sets based on a stratified 10-fold
          cross-validation scheme. The `strat_fold` column in the metadata indicates the fold number
          for each record.

        - This class performs the following preprocessing steps:
            1. Loads the metadata from the `ptbxl_database.csv` file and converts the `scp_codes`
               column from string to dictionary format.
            2. Loads the mapping of SCP-ECG statements to diagnostic classes from the
               `scp_statements.csv` file.
            3. Aggregates the diagnostic statements for each record into a list of unique diagnostic
               classes using the `_aggregate_diagnostic` method.
            4. Splits the dataset into training and testing sets based on the specified `test_fold`.
            5. Loads the raw ECG signal data for each record using the `_load_raw_data` method.
            6. Assigns the diagnostic classes to the `y` attribute and the ECG signals to the `X`
               attribute.

    Attributes:
        - path (str): Path to the directory containing the dataset files.
        - sampling_rate (int): Sampling rate of the ECG signals (500 Hz or 100 Hz).
        - test_fold (int): Index of the fold to be used as the test set (0-9).
        - train (bool): Flag indicating whether to load the training set (True) or test set (False).
        - verbose (bool): Flag indicating whether to print verbose output during dataset loading.
        - data (pd.DataFrame): DataFrame containing the metadata for the selected records.
        - X (np.ndarray): Array containing the ECG signals for the selected records.
        - y (np.ndarray): Array containing the diagnostic classes for the selected records.

    Methods:
        __len__(): Returns the number of records in the dataset.
        __getitem__(index): Returns the ECG signal and diagnostic class for the record at the
            specified index.

    Example usage:
        # Load the training set
        train_dataset = PTBXLDataset(path='./ptb_xl_dataset/', train=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Load the test set
        test_dataset = PTBXLDataset(path='./ptb_xl_dataset/', train=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Iterate over the dataset
        for signals, labels in train_loader:
            # Perform operations on the signals and labels
            ...
    """

    def __init__(
        self,
        path: str,
        sampling_rate: int = 100,
        test_fold: int = 10,
        train: bool = True,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.path = path
        self.sampling_rate = sampling_rate
        self.test_fold = test_fold
        self.train = train

        self.data = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
        self.data.scp_codes = self.data.scp_codes.apply(lambda x: ast.literal_eval(x))

        self.agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

        self.data["diagnostic_superclass"] = self.data.scp_codes.apply(
            self._aggregate_diagnostic
        )

        if self.train:
            self.data = self.data[self.data.strat_fold != self.test_fold]
        else:
            self.data = self.data[self.data.strat_fold == self.test_fold]

        self.X = self._load_raw_data(self.data, self.sampling_rate, self.path)
        self.y = self.data.diagnostic_superclass.apply(
            lambda x: x[0] if x else -1
        ).values

        if self.verbose:
            logging.info(f"Shape of input data: {self.X.shape}")
            logging.info(f"Shape of output data: {self.y.shape}")
            logging.info(f"Unique classes: {len(set(self.y))}")
            logging.info(f"Sample dataset: {self.data.head()}")

    def _aggregate_diagnostic(self, y_dic):
        """
        Description:
            - Aggregate the diagnostic statements into a list of unique diagnostic classes.
            - This method takes a dictionary of SCP-ECG statements (y_dic) as input and
              returns a list of unique diagnostic classes associated with those statements.
            - The SCP-ECG statements are mapped to diagnostic classes using the agg_df
              DataFrame, which contains a mapping between SCP-ECG statements and their
              corresponding diagnostic classes.

        Args:
            - y_dic (dict): A dictionary of SCP-ECG statements, where the keys are the
                statement codes and the values are the corresponding probabilities or
                presence indicators.

        Returns:
            - list: A list of unique diagnostic classes associated with the input SCP-ECG
                statements. If a statement code is not found in the agg_df DataFrame,
                it is not included in the output list.

        Example:
            - Suppose the input y_dic is {"SCP_0001": 1, "SCP_0002": 0, "SCP_0003": 1},
              and the agg_df DataFrame contains the following mapping:
                SCP_0001 -> Diagnostic_Class_A
                SCP_0003 -> Diagnostic_Class_B
            - Then, the output of this method would be ["Diagnostic_Class_A", "Diagnostic_Class_B"].
        """
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def _load_raw_data(self, df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.long
        )
