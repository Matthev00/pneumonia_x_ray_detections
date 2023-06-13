from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os


def create_dataloaders(train_dir: str,
                       test_dir: str,
                       train_transforms: transforms,
                       test_transforms: transforms,
                       num_workers: int,
                       batch_size: int = 32):
    """
    Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders

    Args:
        train_dir (str): Path to training directory.
        test_dir (str): Path to testing directory.
        train_transforms (transforms): torchvision train transforms to perform on training data # noqa 5501
        test_transforms (transforms): torchvision test transforms to perform on testing data # noqa 5501
        num_workers (int): An integer for number of workers per DataLoader.
        batch_size (int, optional): Number of samples per batch in each of the DataLoaders. Defaults to 32. # noqa 5501

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    """

    # # Create datasets using ImageFolder
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transforms)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=test_transforms)

    # class names
    class_names = train_data.classes

    # # Turn datasets into dataloaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

    return train_dataloader, test_dataloader, class_names


def main():
    # # Example of usage

    data_dir = Path("data/10_percent_data")
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    NUM_WORKERS = os.cpu_count()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224))])

    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transforms=transform,
        test_transforms=transform,
        num_workers=NUM_WORKERS,
    )

    print(next(iter(test_dataloader))[0].shape)


if __name__ == "__main__":
    main()
