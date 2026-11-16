# CIFAR-10 Image Classification with CNNs and Optuna

A PyTorch project for classifying CIFAR-10 images. It builds a baseline convolutional neural network, compares learning-rate schedulers, uses Optuna to search for a stronger CNN configuration, and trains a final model with the selected settings.

> **Results status:** The notebooks contain useful recorded results, but the final metrics should be treated as preliminary until the checkpoint placement is corrected and the model is rerun. The Optuna search also used CIFAR-10's official test split as its evaluation set, so that split is no longer an untouched final holdout.

## Project notebooks

1. **`base_model.ipynb`** — establishes a small CNN baseline and compares `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau`.
2. **`optuna_optimization.ipynb`** — searches CNN architecture and training hyperparameters with Optuna.
3. **`cifar10_final_training.ipynb`** — applies the best recorded Optuna configuration, trains with cosine annealing, plots learning curves, and evaluates the model.

## Dataset and preprocessing

The notebooks load CIFAR-10 from Hugging Face (`uoft-cs/cifar10`). The dataset has 50,000 training images and 10,000 test images. Images are 32×32 RGB; preprocessing converts them to tensors and normalizes channels using CIFAR-10 statistics.

The final-training notebook creates a reproducible, stratified 90/10 split from the training portion using `seed=42`, giving 45,000 training examples and 5,000 validation examples. The official test split is evaluated separately in that notebook. However, the earlier Optuna notebook evaluated each trial on the official test split, so the test results have been used during model selection.

## Model and training approach

The baseline CNN uses two convolution and pooling blocks followed by fully connected layers and dropout. The Optuna model, `FlexibleCNN`, varies the number of convolutional blocks and their filter counts and kernel sizes; it builds its classifier dynamically to fit the resulting feature-map shape.

The best configuration recorded in the saved study is:

| Setting | Selected value |
| --- | ---: |
| Convolutional blocks | 3 |
| Filters per block | 95, 111, 44 |
| Kernel sizes | 5×5, 5×5, 5×5 |
| Dropout | 0.1047 |
| Hidden classifier width | 113 |
| Adam learning rate | 0.0010123 |

The final-training notebook uses batch size 128, cross-entropy loss, Adam, and `CosineAnnealingLR` for 50 epochs (`T_max=50`, `eta_min=0.0002`). A dummy forward pass initializes the dynamic classifier before the optimizer is created, ensuring its parameters are included.

## Recorded results

The notebook outputs report:

- Best validation accuracy: **78.02% at epoch 47**.
- Test accuracy: **77.69%**.
- Test loss: **0.8998**.
- Macro precision: **0.7757**.
- Macro recall: **0.7769**.
- Macro F1: **0.7759**.

These values are reproduced from the saved notebook outputs and are **preliminary** for two reasons:

1. In `cifar10_final_training.ipynb`, the code that reloads the best checkpoint is currently inside the epoch loop. This restores the best weights after every epoch, while the optimizer and scheduler continue forward. Move the restore block after the loop and rerun before treating the result as the final training score.
2. `optuna_optimization.ipynb` uses the official test split as the trial evaluation split. This means the reported test accuracy is not an unbiased, untouched estimate after hyperparameter selection.

The saved Optuna table shows 12 completed trials and one failed trial, while the run cell requests 20 and the reported project count has been 11. Confirm the intended study record before publishing the trial count.

## Environment and dependencies

Use Python with Jupyter Notebook or JupyterLab. Install the project’s main packages with:

```bash
pip install torch torchvision datasets torchmetrics optuna joblib matplotlib huggingface_hub msgpack jupyter
```

The notebooks import `data_preprocess.py` and `help_utils.py`. Those helper files are not included in this notebook-only folder, so place them beside the notebooks or upload them in the Colab setup cell. The Optuna analysis/final-training workflow also loads `optuna_study.pkl`; provide the saved study file at the path expected by the notebook. Hugging Face authentication is optional for public dataset access, though authentication may improve download limits.

## Suggested run order

1. Make the helper modules and dataset available, then run `base_model.ipynb` to review the baseline and scheduler comparison.
2. Run or load the Optuna study in `optuna_optimization.ipynb` and confirm which study file and completed-trial count you are using.
3. In `cifar10_final_training.ipynb`, keep the seeded training/validation split, move best-checkpoint restoration outside the epoch loop, and rerun training.
4. Use validation metrics to choose the checkpoint. Keep a fresh holdout untouched during further tuning and report its result only after model selection is complete.
