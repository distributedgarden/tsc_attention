import logging
import pandas as pd
import numpy as np
import wfdb
import ast
import torch
import os

from typing import List, Tuple, Dict, Union

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
        - https://physionet.org/content/ptb-xl/1.0.1/

        - The PTB-XL dataset contains electrocardiogram (ECG) records from patients, along with
          annotations of diagnostic statements and other metadata. This class provides a way to
          load and preprocess the dataset for machine learning tasks.

        - Key features of the dataset:
            - 21,837 clinical 12-lead ECG records from 18,885 patients
            - 10-second recordings sampled at 500 Hz or 100 Hz
            - Annotations of diagnostic statements, ECG statements, and signal quality
            - Metadata including patient demographics, recording information, and diagnostic classes

        - The dataset contains 5 unique diagnostic classes:
            - NORM: Normal ECG
            - MI: Myocardial Infarction
            - STTC: ST/T Change
            - CD: Conduction Disturbance
            - HYP: Hypertrophy

        - The dataset is split into training and testing sets based on a stratified 10-fold
          cross-validation scheme. The `strat_fold` column in the metadata indicates the fold number
          for each record.

    Args:
        - path (str): Path to the directory containing the dataset files.
        - sampling_rate (int): Sampling rate of the ECG signals (500 Hz or 100 Hz).
        - test_fold (int): Index of the fold to be used as the test set (0-9).
        - train (bool): Flag indicating whether to load the training set (True) or test set (False).
        - impute_strategy (str): Strategy for imputing missing values ('mean', 'median', or 'zero').
        - normalize (bool): Flag indicating whether to normalize the ECG signals.
        - categorical_encoding (str): Encoding strategy for categorical variables ('ohe' or 'label').
        - verbose (bool): Flag indicating whether to print verbose output during dataset loading.

    Attributes:
        - data (pd.DataFrame): DataFrame containing the metadata for the selected records.
        - X (np.ndarray): Array containing the ECG signals for the selected records.
        - y (np.ndarray): Array containing the diagnostic classes for the selected records.
        - unique_classes (np.ndarray): Array of unique diagnostic class tuples.
        - class_to_idx (Dict[Tuple[str], int]): Mapping of diagnostic class tuples to integer labels.
    """

    def __init__(
        self,
        path: str,
        sampling_rate: int = 100,
        test_fold: int = 10,
        train: bool = True,
        impute_strategy: str = "mean",
        normalize: bool = True,
        categorical_encoding: str = "ohe",
        verbose: bool = True,
    ):
        self.path = path
        self.sampling_rate = sampling_rate
        self.test_fold = test_fold
        self.train = train
        self.impute_strategy = impute_strategy
        self.normalize = normalize
        self.categorical_encoding = categorical_encoding
        self.verbose = verbose

        self.data = self._load_data()
        self.X, self.y, self.unique_classes, self.class_to_idx = self._preprocess_data()

        if self.verbose:
            logging.info(f"Shape of input data: {self.X.shape}")
            logging.info(f"Shape of output data: {self.y.shape}")
            logging.info(f"Number of unique classes: {len(self.unique_classes)}")
            logging.info(f"Unique classes: {self.unique_classes}")
            logging.info(f"Class-to-index mapping: {self.class_to_idx}")
            logging.info(f"Sample dataset: {self.data.head()}")

    def _load_data(self) -> pd.DataFrame:
        """
        Description:
            - Load the PTB-XL dataset metadata from the 'ptbxl_database.csv' file.
            - 'ptbxl_database.csv' contains metadata information for each record in the PTB-XL dataset.
                - 'ecg_id': The unique identifier for each record.
                - 'scp_codes': Standard Communications Protocol (SCP) codes for each record.
                  String representation of a dictionary, which is converted to an actual dictionary.

        Returns:
            pd.DataFrame: A DataFrame containing the metadata for each record in the dataset,
                with the 'ecg_id' column as the index and the 'scp_codes' column converted to
                a dictionary.
        """
        data = pd.read_csv(
            os.path.join(self.path, "ptbxl_database.csv"), index_col="ecg_id"
        )
        data.scp_codes = data.scp_codes.apply(lambda x: ast.literal_eval(x))

        return data

    def _load_raw_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Description:
            - Load the raw ECG signals from WFDB (WaveForm DataBase) files.
            - WFDB files contain the actual ECG recordings for each sample in the dataset.
            - The ECG signals are converted to a NumPy array and returned.
              (num_samples, num_leads, num_timesteps)
                - num_samples: The number of ECG recordings in the dataset.
                - num_leads: The number of ECG leads (e.g., 12) in each ECG recording.
                - num_timesteps: The number of time points in each ECG recording.

            - Filepaths:
                - filename_lr: The 'filename_lr' column contains the filepaths of the low-resolution
                  (100 Hz) WFDB files.
                - filename_hr: The 'filename_hr' column contains the filepaths of the high-resolution
                  (500 Hz) WFDB files.

        Args:
            - data (pd.DataFrame): DataFrame containing the filepaths of the WFDB files.

        Returns:
            - np.ndarray: NumPy array containing the loaded ECG signals.

        Note:
            The 'wfdb' library is used to read the WFDB files. Make sure you have the 'wfdb'
            library installed before using this method.
        """
        if self.sampling_rate == 100:
            filepaths = data.filename_lr
        else:
            filepaths = data.filename_hr

        signals = []
        for filepath in filepaths:
            signal, _ = wfdb.rdsamp(os.path.join(self.path, filepath))
            signals.append(signal)

        return np.array(signals)

    def _preprocess_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[str], int]]:
        """
        Description:
            - Preprocess the dataset by handling missing values, normalizing signals, and encoding categorical variables.

        Returns:
            - Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[str], int]]:
                - X (np.ndarray): Array containing the ECG signals.
                - y (np.ndarray): Array containing the diagnostic classes.
                - unique_classes (np.ndarray): Array of unique diagnostic class tuples.
                - class_to_idx (Dict[Tuple[str], int]): Mapping of diagnostic class tuples to integer labels.
        """
        agg_df = pd.read_csv(os.path.join(self.path, "scp_statements.csv"), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        data = self.data.copy()

        # Split the data into train and test sets
        if self.train:
            data = data[data.strat_fold != self.test_fold]
        else:
            data = data[data.strat_fold == self.test_fold]

        # Load raw ECG signals
        X = self._load_raw_data(data)

        # Handle missing values
        X = self._impute_nans(X)

        # Normalize the ECG signals
        if self.normalize:
            X = self._normalize_data(X)

        # Encode categorical variables
        data = self._encode_categoricals(data)

        # Aggregate diagnostic statements and assign labels
        data["diagnostic_superclass"] = data.scp_codes.apply(
            lambda x: self._aggregate_diagnostic(x, agg_df)
        )
        unique_classes = (
            data["diagnostic_superclass"].apply(lambda x: tuple(x)).unique()
        )
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        y = data.diagnostic_superclass.apply(lambda x: class_to_idx[tuple(x)]).values

        # Update the data attribute to match the preprocessed data
        self.data = data

        return X, y, unique_classes, class_to_idx

    def _impute_nans(self, X: np.ndarray) -> np.ndarray:
        """
        Description:
            - Impute missing values in the ECG signals.

        Args:
            - X (np.ndarray): Array containing the ECG signals.

        Returns:
            - np.ndarray: Array containing the ECG signals with imputed missing values.
        """
        for i in range(X.shape[1]):
            if np.isnan(X[:, i]).any():
                if self.impute_strategy == "mean":
                    X[:, i][np.isnan(X[:, i])] = np.nanmean(X[:, i])
                elif self.impute_strategy == "median":
                    X[:, i][np.isnan(X[:, i])] = np.nanmedian(X[:, i])
                elif self.impute_strategy == "zero":
                    X[:, i][np.isnan(X[:, i])] = 0
                else:
                    raise ValueError(
                        f"Invalid imputation strategy: {self.impute_strategy}"
                    )
        return X

    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """
        Description:
            - Normalize the ECG signals using clipping based on the interquartile range (IQR).

            - This method first calculates the IQR of the input data, then clips the extreme values
              based on the IQR, and finally applies standard scaling to the clipped data.

        Args:
            - X (np.ndarray): Input ECG signals of shape (num_samples, num_timesteps, num_leads).

        Returns:
            - np.ndarray: Normalized ECG signals of shape (num_samples, num_timesteps, num_leads).
        """
        # Reshape X to (num_samples * num_timesteps, num_leads)
        X_reshaped = X.reshape(-1, X.shape[-1])

        # Calculate the IQR for each lead
        Q1 = np.percentile(X_reshaped, 25, axis=0)
        Q3 = np.percentile(X_reshaped, 75, axis=0)
        IQR = Q3 - Q1

        # Define the clipping range based on the IQR
        clip_min = Q1 - 1.5 * IQR
        clip_max = Q3 + 1.5 * IQR

        # Clip the extreme values based on the IQR
        X_clipped = np.clip(X_reshaped, clip_min, clip_max)

        # Apply standard scaling to the clipped data
        mean = np.mean(X_clipped, axis=0)
        std = np.std(X_clipped, axis=0)
        X_scaled = (X_clipped - mean) / std

        # Reshape the scaled data back to (num_samples, num_timesteps, num_leads)
        X_scaled = X_scaled.reshape(X.shape)

        return X_scaled

    def _encode_categoricals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Description:
            - Encode categorical variables.

        Args:
            - data (pd.DataFrame): DataFrame containing the dataset metadata.

        Returns:
            - pd.DataFrame: DataFrame containing the encoded dataset metadata.
        """
        categorical_cols = ["sex", "norm_rhy", "sinus_rhy"]
        present_cols = [col for col in categorical_cols if col in data.columns]

        if self.categorical_encoding == "ohe":
            data = pd.get_dummies(data, columns=present_cols)
        elif self.categorical_encoding == "label":
            if "sex" in present_cols:
                data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
            if "norm_rhy" in present_cols:
                data["norm_rhy"] = data["norm_rhy"].map({False: 0, True: 1})
            if "sinus_rhy" in present_cols:
                data["sinus_rhy"] = data["sinus_rhy"].map({False: 0, True: 1})
        else:
            raise ValueError(
                f"Invalid categorical encoding strategy: {self.categorical_encoding}"
            )

        return data

    def _aggregate_diagnostic(
        self, y_dic: Dict[str, float], agg_df: pd.DataFrame
    ) -> List[str]:
        """
        Description:
            - Aggregate the diagnostic statements into a list of unique diagnostic classes.

        Args:
            - y_dic (Dict[str, float]): Dictionary containing the diagnostic statements.
            - agg_df (pd.DataFrame): DataFrame containing the aggregated diagnostic statements.

        Returns:
            - List[str]: List of unique diagnostic classes for the given record.
        """
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)

        return list(set(tmp))

    def __len__(self) -> int:
        """
        Description:
            - Return the number of records in the dataset.

        Returns:
            - int: Number of records in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Description:
            - Return the ECG signal tensor and the integer label index for the record at the specified index.

        Args:
            - idx (int): Index of the record to retrieve.

        Returns:
            - Tuple[torch.Tensor, int]: Tuple containing the ECG signal tensor and the integer label index.
        """
        label_idx = self.y[idx]

        return torch.tensor(self.X[idx], dtype=torch.float32), label_idx
