# ML_Projects

## Purpose

The purpose of this repository is to summarize my ML progression, from training DNNs on the CIFAR-10 data to training segmentation-based CNN models on the Oxford-IIIT dataset.

## Training Pipelines

Use the following commands to run the training and evaluation pipelines.

- Run basic model evaluation:
  - `python -m training.basic_evaluate`
- Run CNN model evaluation:
  - `python -m training.cnn_evaluate`
- Train the basic DNN model:
  - `python -m training.basic_train`
- Train the CNN model:
  - `python -m training.cnn_train`
- Train the U-Net segmentation model:
  - `python -m training.unet_train`

## Visualization

Use these commands to run visualization scripts.

- Visualize generic outputs:
  - `python -m visualization.visualization`
- Visualize U-Net segmentation results:
  - `python -m visualization.unet_seg_visualization`

