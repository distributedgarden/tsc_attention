# tsc_attention

An exploration on the effect of self-attention on an ECG time series classification task using LSTM and CNN based models

# Setup
Note: GPU required now that flash attention is integrated.

```
> python3 -m venv env
> source env/bin/activate
> python3 -m pip install -r requirements.txt
```

### tests
```
> python3 -m pytest --cov=src --cov-report xml tests/
```

### pre-commit hooks
```
> pre-commit install
> pre-commit autoupdate
> pre-commit run --all-files
```


# References
1. Karim, F., Majumdar, S., Darabi, H., & Chen, S. (2018). LSTM Fully Convolutional networks for Time series classification. IEEE Access, 6, 1662–1669. https://doi.org/10.1109/access.2017.2779939
1. Karim, F., Majumdar, S., & Darabi, H. (2019). Insights into LSTM Fully Convolutional Networks for Time Series Classification. IEEE Access, 7, 67718–67725. https://doi.org/10.1109/access.2019.2916828
1. Tang, W., Long, G., Liu, L., Zhou, T., Jiang, J., & Blumenstein, M. (2020). Rethinking 1D-CNN for time series classification: a stronger baseline. arXiv (Cornell University). https://arxiv.org/pdf/2002.10061.pdf
1. Tóth, C. D., Bonnier, P., & Oberhauser, H. (2021). Seq2Tens: an efficient representation of sequences by Low-Rank tensor projections. arXiv (Cornell University). https://arxiv.org/pdf/2006.07027
1. Liu, F., Zhou, X., Wang, T., Cao, J., Wang, Z., Wang, H., & Zhang, Y. (2019). An Attention-based Hybrid LSTM-CNN Model for Arrhythmias Classification. International Joint Conference on Neural Networks. https://doi.org/10.1109/ijcnn.2019.8852037 
1. Papers with Code - ECG Benchmark (Time Series Classification). (n.d.). https://paperswithcode.com/sota/time-series-classification-on-ecg
1. Electrocardiogram. (2021, August 8). Johns Hopkins Medicine. https://www.hopkinsmedicine.org/health/treatment-tests-and-therapies/electrocardiogram
1. PhysioNet databases. (n.d.). https://www.physionet.org/about/database/
1. Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018). ECG Heartbeat Classification: A Deep Transferable Representation. https://doi.org/10.1109/ichi.2018.00092
1. ECG Heartbeat Categorization Dataset. (2018, May 31). Kaggle. https://www.kaggle.com/datasets/shayanfazeli/heartbeat
1. Staudemeyer, R. C. (2019, September 12). Understanding LSTM -- a tutorial into Long Short-Term Memory Recurrent Neural Networks. arXiv.org. https://arxiv.org/abs/1909.09586
1. Rußwurm, M., & Körner, M. (2020). Self-attention for raw optical Satellite Time Series Classification. Isprs Journal of Photogrammetry and Remote Sensing, 169, 421–435. https://doi.org/10.1016/j.isprsjprs.2020.06.006
1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?”: Explaining the Predictions of Any Classifier. KDD. https://doi.org/10.18653/v1/n16-3020
1. Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: visualising image classification models and saliency maps. International Conference on Learning Representations. https://www.robots.ox.ac.uk/~vgg/publications/2014/Simonyan14a/simonyan14a.pdf
1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All you Need. arXiv (Cornell University), 30, 5998–6008. https://arxiv.org/pdf/1706.03762v5
1. Flach, P. A., & Kull, M. (2015). Precision-Recall-Gain curves: PR analysis done right. Advances in Neural Information Processing Systems 28 (NIPS 2015), 28, 838–846.

