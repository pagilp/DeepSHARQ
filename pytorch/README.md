# Pytorch DeepSHARQ

This repository contains the required code and datasets to train the DeepHEC and DeepSHARQ models.

## Model Training

* DeepHEC models may be trained with the `deephec_superconvergence.ipynb` notebook.

* DeepSHARQ models without output regularization may be trained with the `deepsharq_superconvergance.ipynb` notebook.

* DeepSHARQ models with output regularization may be trained with the `deepsharq_[loss]_delta.ipynb` notebook, where `[loss]=superconvergance|plateau` represents the loss update function.

## Model Conversion

The `weight_transfer_onnx.ipynb` notebook can convert any model trained with PyTorch into a TFLite model.
