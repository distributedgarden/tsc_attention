# Description
Notebook-based experiments and figure generation. `SKF` notebooks are only scoped to generate training related plots.

# Experiments
- training accuracy, precision, recall, f1 score, ROC curve, label misclassification per epoch, LIME explainability on randomly selected set of correctly and incorrectly labeled instances from the test dataset, saliency instance overlay, attention weight instance overlay.

# Notebooks
- `TSC_LSTM.ipynb`: LSTM and LSTM with self-attention models.
- `TSC_LSTMFCN.ipynb`: LSTMFCN and LSTMFCN with self-attention models.
- `TSC_OSCNN.ipynb`: OS-CNN and OS-CNN with self-attention models.
- `TSC_LSTM_SKF.ipynb`: LSTM and LSTM with self-attention models using stratified 10-fold cross validation.
- `TSC_LSTMFCN_SKF.ipynb`: LSTMFCN and LSTMFCN with self-attention models using stratified 10-fold cross validation.
- `TSC_OSCNN_SKF.ipynb`: OS-CNN and OS-CNN with self-attention models using stratified 10-fold cross validation.
- `TSC_data_exploration.ipynb`: MIT-BIH Heart Arrhythmia dataset exploration.
