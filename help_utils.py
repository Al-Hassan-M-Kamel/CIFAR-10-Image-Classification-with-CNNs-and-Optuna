import torch
from tqdm import tqdm
import math

from typing import Optional

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


# = DLAI plots style =
def apply_dlai_style():
    # Global plot style
    PLOT_STYLE = {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "font.family": "sans",  # "sans-serif",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 3,
        "lines.markersize": 6,
    }

    # Custom colors (reusable)
    color_map = {
        "pink": "#F65B66",
        "blue": "#1C74EB",
        "yellow": "#FAB901",
        "red": "#DD3C66",
        "purple": "#A12F9D",
        "cyan": "#237B94",
    }
    return color_map, PLOT_STYLE



color_map, PLOT_STYLE = apply_dlai_style()
mpl.rcParams.update(PLOT_STYLE)



# Custom colors (reusable)
BLUE_COLOR_TRAIN = color_map["blue"]
PINK_COLOR_TEST = color_map["pink"]


def set_seed(seed=42):
    """Sets the seed for random number generators for reproducibility.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    # Set the seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set the seed for all available GPUs
    torch.cuda.manual_seed_all(seed)

    

class NestedProgressBar:
    """A handler for nested tqdm progress bars for training and evaluation loops.

    This class creates and manages an outer progress bar for epochs and an
    inner progress bar for batches. It supports automatic detection for
    terminal and notebook environments and includes a granularity feature to
    control the number of visual updates for very long processes.
    """
    def __init__(
        self,
        total_epochs,
        total_batches,
        g_epochs=None,
        g_batches=None,
        use_notebook=True,
        epoch_message_freq=None,
        batch_message_freq=None,
        mode="train",
    ):
        """Initializes the nested progress bars.

        Args:
            total_epochs (int): The absolute total number of epochs.
            total_batches (int): The absolute total number of batches per epoch.
            g_epochs (int, optional): The visual granularity for the epoch bar.
                                      Defaults to total_epochs.
            g_batches (int, optional): The visual granularity for the batch bar.
                                       Defaults to total_batches.
            use_notebook (bool, optional): A flag to control notebook mode.
                                           `tqdm.auto` is used, so this is
                                           often not needed. Defaults to True.
            epoch_message_freq (int, optional): Frequency to log epoch
                                                messages. Defaults to None.
            batch_message_freq (int, optional): Frequency to log batch
                                                messages. Defaults to None.
            mode (str, optional): The operational mode, either 'train' or 'eval'.
                                  Defaults to "train".
        """
        # Store initial configuration
        self.mode = mode
        self.use_notebook = use_notebook
        self.epoch_message_freq = epoch_message_freq
        self.batch_message_freq = batch_message_freq

        # Import the appropriate tqdm implementation
        from tqdm.auto import tqdm as tqdm_impl
        self.tqdm_impl = tqdm_impl

        # Store the absolute total counts for epochs and batches
        self.total_epochs_raw = total_epochs
        self.total_batches_raw = total_batches
        # Determine the visual granularity, ensuring it doesn't exceed the total count
        self.g_epochs = min(g_epochs or self.total_epochs_raw, self.total_epochs_raw)
        self.g_batches = min(
            g_batches or self.total_batches_raw, self.total_batches_raw
        )
        # Set the progress bar totals to the calculated granularity
        self.total_epochs = self.g_epochs
        self.total_batches = self.g_batches

        # Initialize the progress bar widgets
        self._init_bars()

    def _init_bars(self):
        """Initializes or re-initializes the progress bar widgets."""
        # Close existing epoch bar if it exists, ignoring any errors
        if hasattr(self, "epoch_bar") and self.epoch_bar is not None:
            try:
                self.epoch_bar.close()
            except Exception:
                pass
        # Close existing batch bar if it exists, ignoring any errors
        if hasattr(self, "batch_bar") and self.batch_bar is not None:
            try:
                self.batch_bar.close()
            except Exception:
                pass

        # Initialize bars for training mode with nested layout
        if self.mode == "train":
            self.epoch_bar = self.tqdm_impl(
                total=self.total_epochs, desc="Current Epoch", position=0, leave=True
            )
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Current Batch", position=1, leave=False
            )
        # Initialize a single bar for evaluation mode
        elif self.mode == "eval":
            self.epoch_bar = None
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Evaluating", position=0, leave=False
            )

        # Initialize trackers for the last visualized update step
        self.last_epoch_step = -1
        self.last_batch_step = -1

    def update_epoch(self, epoch, postfix_dict=None, message=None):
        """Updates the epoch-level progress bar.

        Args:
            epoch (int): The current epoch number.
            postfix_dict (dict, optional): A dictionary of metrics to display.
                                           Defaults to None.
            message (str, optional): A message to potentially log. Defaults to None.
        """
        # Skip epoch updates if not in training mode
        if self.mode != "train":
            return
        # Map the raw epoch count to its corresponding visual step based on granularity
        epoch_step = math.floor((epoch - 1) * self.g_epochs / self.total_epochs_raw)

        # Update the progress bar only when the visual step changes
        if epoch_step > self.last_epoch_step:
            update_amount = epoch_step - self.last_epoch_step
            self.epoch_bar.update(update_amount)
            self.last_epoch_step = epoch_step
        # Ensure the progress bar completes on the final epoch
        elif epoch == self.total_epochs_raw and self.epoch_bar.n < self.g_epochs:
            self.epoch_bar.update(self.g_epochs - self.epoch_bar.n)
            self.last_epoch_step = epoch_step

        # Set the dynamic description for the progress bar
        self.epoch_bar.set_description(f"Training - Current Epoch: {epoch}")
        # Update the postfix with any provided metrics or information
        if postfix_dict:
            self.epoch_bar.set_postfix(postfix_dict)

        # Reset the inner batch bar at the start of each new epoch
        self.batch_bar.reset()
        self.last_batch_step = -1

    def update_batch(self, batch, postfix_dict=None, message=None):
        """Updates the batch-level progress bar.

        Args:
            batch (int): The current batch number.
            postfix_dict (dict, optional): A dictionary of metrics to display.
                                           Defaults to None.
            message (str, optional): A message to potentially log. Defaults to None.
        """
        # Map the raw batch count to its corresponding visual step
        batch_step = math.floor((batch - 1) * self.g_batches / self.total_batches_raw)

        # Update the progress bar only when the visual step changes
        if batch_step > self.last_batch_step:
            update_amount = batch_step - self.last_batch_step
            self.batch_bar.update(update_amount)
            self.last_batch_step = batch_step
        # Ensure the progress bar completes on the final batch
        elif batch == self.total_batches_raw and self.batch_bar.n < self.g_batches:
            self.batch_bar.update(self.g_batches - self.batch_bar.n)
            self.last_batch_step = batch_step

        # Set the dynamic description for the progress bar based on the mode
        if self.mode == "train":
            self.batch_bar.set_description(f"Training - Current Batch: {batch}")
        elif self.mode == "eval":
            self.batch_bar.set_description(f"Evaluation - Current Batch: {batch}")

        # Update the postfix with any provided metrics
        if postfix_dict:
            self.batch_bar.set_postfix(postfix_dict)

    def maybe_log_epoch(self, epoch, message):
        """Logs a message at a specified epoch frequency.

        Args:
            epoch (int): The current epoch number.
            message (str): The message to log.
        """
        if self.epoch_message_freq and epoch % self.epoch_message_freq == 0:
            print(message)

    def maybe_log_batch(self, batch, message):
        """Logs a message at a specified batch frequency.

        Args:
            batch (int): The current batch number.
            message (str): The message to log.
        """
        if self.batch_message_freq and batch % self.batch_message_freq == 0:
            print(message)

    def close(self, last_message=None):
        """Closes all active progress bars and optionally prints a final message.

        Args:
            last_message (str, optional): A final message to print after closing.
                                          Defaults to None.
        """
        # Print a concluding message if one is provided
        if last_message:
            print(last_message)
        # Ensure the batch bar is closed in evaluation mode
        if self.mode == "eval":
            self.batch_bar.close()

    def reset(self):
        """Resets the progress of the bars without re-creating the widgets."""
        # Reset the epoch bar if it exists
        if self.mode == "train" and self.epoch_bar:
            self.epoch_bar.reset()
        # Reset the batch bar if it exists
        if self.batch_bar:
            self.batch_bar.reset()
        # Reset the step trackers
        self.last_epoch_step = -1
        self.last_batch_step = -1


def plot_learning_rates(history, color, label="", ax=None):
    """Plots the learning rate schedule over epochs.

    This function generates a plot showing how the learning rate changed
    during training. It can either create a new figure or plot on an
    existing Matplotlib axes object.

    Args:
        history (dict): A dictionary containing training metrics. It must have
                        the key 'lr'.
        color (str): The color to use for the plot line.
        label (str, optional): A label for the plot line in the legend.
                               Defaults to an empty string.
        ax (matplotlib.axes.Axes, optional): A Matplotlib axes object to plot on.
                                              If None, a new figure and axes are
                                              created. Defaults to None.
    """
    # If no axes are provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the learning rate against the epochs
    ax.plot(history["lr"], label=f"{label}", color=color)
    # Set the title of the plot
    ax.set_title("Learning Rate")
    # Set the label for the x-axis
    ax.set_xlabel("Epochs")
    # Set the label for the y-axis
    ax.set_ylabel("Learning Rate")
    # Display the legend on the plot
    ax.legend()

    

def plot_learning_rates_curves(training_curves_new, colors, labels):
    """Plots and compares multiple learning rate schedules on a single figure.

    This function creates a figure and overlays the learning rate curves from
    several training runs, each with a different color and label.

    Args:
        training_curves_new (list): A list of training history dictionaries.
        colors (list): A list of color strings, one for each history.
        labels (list): A list of labels for the legend, corresponding to each history.
    """
    # Create a new figure and a single axes object for the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    # Iterate through the histories, colors, and labels, skipping the first element
    for history, color, label in zip(training_curves_new[1:], colors[1:], labels[1:]):
        # Call a helper function to plot the learning rate curve for the current history
        plot_learning_rates(history, color, label=label, ax=ax)
    # Adjust the plot layout to prevent labels from overlapping
    plt.tight_layout()


def plot_learning(history, color, axes=None):
    """Plots the training loss and validation accuracy from a training history.

    This function generates two subplots: one for the training loss per epoch
    and one for the validation accuracy per epoch. It can either create a new
    figure or plot on existing axes.

    Args:
        history (dict): A dictionary containing training metrics. It must have
                        the keys 'train_loss' and 'val_acc'.
        color (str): The color to use for the plot lines.
        axes (matplotlib.axes.Axes, optional): A tuple of two Matplotlib axes
                                               objects to plot on. If None, new
                                               subplots are created. Defaults to None.
    """
    # If no axes are provided, create a new figure with two subplots
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the training loss on the first subplot
    axes[0].plot(history["train_loss"], linestyle="-", color=color)
    # Set the title for the loss subplot
    axes[0].set_title("Training Loss")
    # Set the x-axis label for the loss subplot
    axes[0].set_xlabel("Epochs")
    # Set the y-axis label for the loss subplot
    axes[0].set_ylabel("Loss")

    # Plot the validation accuracy on the second subplot
    axes[1].plot(history["val_acc"], linestyle="-", color=color)
    # Set the title for the accuracy subplot
    axes[1].set_title("Validation Accuracy")
    # Set the x-axis label for the accuracy subplot
    axes[1].set_xlabel("Epochs")
    # Set the y-axis label for the accuracy subplot
    axes[1].set_ylabel("Accuracy")

    # Adjust layout if a new figure was created to prevent overlapping titles
    if axes is None:
        plt.tight_layout()

        

def plot_learning_curves(colors, labels, histories):
    """Plots and compares the learning curves for multiple training histories.

    This function creates a figure with two subplots (training loss and
    validation accuracy) and overlays the learning curves from several
    training runs, each with a different color and label.

    Args:
        colors (list): A list of color strings, one for each history.
        labels (list): A list of labels for the legend, corresponding to each history.
        histories (list): A list of history dictionaries. Each dictionary should
                          contain keys like 'train_loss' and 'val_acc'.
    """
    # Create a new figure and a set of subplots for the learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Iterate through each training history and its corresponding color
    for history, color in zip(histories, colors):
        # Call a helper function to plot the curves for the current history
        plot_learning(history, color=color, axes=axes)

    # Create custom legend handles for each training run
    color_handles = [
        mlines.Line2D([], [], color=color, label=label)
        for color, label in zip(colors, labels)
    ]

    # Add the combined legend to each of the subplots
    for ax in axes:
        ax.legend(handles=color_handles)

    # Adjust the plot layout to prevent labels from overlapping
    plt.tight_layout()