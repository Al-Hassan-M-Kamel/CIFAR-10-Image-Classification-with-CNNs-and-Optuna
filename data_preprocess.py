from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


# Convert each PIL image to a tensor and normalize its RGB channels.
# ToTensor() changes the image to a float tensor with shape [3, 32, 32]
# and values in [0, 1]. Normalize() applies the CIFAR-10 channel statistics.
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
])


def get_data():
    """download CIFAR-10 from huggingface.


    Returns:
        A Hugging Face DatasetDict containing the dataset splits.
    """
    print("Downloading the dataset...")
    ds = load_dataset("uoft-cs/cifar10")

    return ds


class CIFAR_10(Dataset):
    """Wrap a Hugging Face CIFAR-10 split for use with a PyTorch DataLoader."""

    def __init__(self, data_source, transform=None):
        """Store a dataset split and an optional image transformation.

        Args:
            data_source: A Hugging Face dataset split with `img` and `label`
                columns.
            transform: Optional callable applied to each image when retrieved.
        """
        self.data = data_source
        self.transform = transform

    def __len__(self):
        """Return the number of examples in this dataset split."""
        return len(self.data)

    def __getitem__(self, index:int):
        """Return one transformed image and its integer class label.

        Args:
            index: Position of the example within the dataset split.

        Returns:
            A pair `(image, label)`, where `image` is transformed if a
            transform was provided, and `label` is a torch.long tensor.
        """
        # The Hugging Face dataset stores the image and label in named columns.
        
        x = self.data[index]["img"]
        y = torch.tensor(self.data[index]["label"], dtype=torch.long)

        # Convert and normalize the image if a transform was supplied.
        if self.transform:
            x = self.transform(x)

        return x, y
