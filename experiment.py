import torch
from torchvision import transforms

import data_setup
import utils
import model_builder
import engine

import os
from pathlib import Path


def main():
    NUM_WORKERS = os.cpu_count()
    BATCH_SIZE = 32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # # Create models
    effnetb2, effnetb2_transforms = model_builder.create_effnetb2(
        num_classes=2,
        device=device
    )
    effnetb2_train_transforms = transforms.Compose([
        transforms.TrivialAugmentWide(),
        effnetb2_transforms
    ])

    densenet, densenet_transforms = model_builder.create_densenet(
        num_classes=2,
        device=device
    )
    densenet_train_transforms = transforms.Compose([
        transforms.TrivialAugmentWide(),
        densenet_transforms
    ])

    googlenet, googlenet_transforms = model_builder.create_googlenet(
        num_classes=2,
        device=device
    )
    googlenet_train_transforms = transforms.Compose([
        transforms.TrivialAugmentWide(),
        googlenet_transforms
    ])

    # # Create Dataloaders for ech model
    # 10_percent
    data_dir_10 = Path("data/10_percent_data")
    test_dir_10 = data_dir_10 / "test"
    train_dir_10 = data_dir_10 / "train"
    # 20 percent
    data_dir_20 = Path("data/20_percent_data")
    test_dir_20 = data_dir_20 / "test"
    train_dir_20 = data_dir_20 / "train"

    effnetb2_train_dataloader_10, effnetb2_test_dataloader_10, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir_10,
        test_dir=test_dir_10,
        train_transforms=effnetb2_train_transforms,
        test_transforms=effnetb2_transforms,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    )

    effnetb2_train_dataloader_20, effnetb2_test_dataloader_20, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir_20,
        test_dir=test_dir_20,
        train_transforms=effnetb2_train_transforms,
        test_transforms=effnetb2_transforms,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    )

    densenet_train_dataloader_10, densenet_test_dataloader_10, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir_10,
        test_dir=test_dir_10,
        train_transforms=densenet_train_transforms,
        test_transforms=densenet_transforms,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    )

    densenet_train_dataloader_20, densenet_test_dataloader_20, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir_20,
        test_dir=test_dir_20,
        train_transforms=densenet_train_transforms,
        test_transforms=densenet_transforms,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    )

    googlenet_train_dataloader_10, googlenet_test_dataloader_10, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir_10,
        test_dir=test_dir_10,
        train_transforms=googlenet_train_transforms,
        test_transforms=googlenet_transforms,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    )

    googlenet_train_dataloader_20, googlenet_test_dataloader_20, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir_20,
        test_dir=test_dir_20,
        train_transforms=googlenet_train_transforms,
        test_transforms=googlenet_transforms,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    )

    epochs = [5, 10]
    models = {
        "effnetb2": [effnetb2,
                     effnetb2_train_dataloader_10,
                     effnetb2_test_dataloader_10,
                     effnetb2_train_dataloader_20,
                     effnetb2_test_dataloader_20],
        "densenet": [densenet,
                     densenet_train_dataloader_10,
                     densenet_test_dataloader_10,
                     densenet_train_dataloader_20,
                     densenet_test_dataloader_20],
        "googlenet": [googlenet,
                      googlenet_train_dataloader_10,
                      googlenet_test_dataloader_10,
                      googlenet_train_dataloader_20,
                      googlenet_test_dataloader_20]
    }
    dataloaders = [10, 20]
    experiment_number = 0

    for model_name in models:
        for dataloader in dataloaders:
            if dataloader == 10:
                train_dataloader = models[model_name][1]
                test_dataloader = models[model_name][2]
            else:
                train_dataloader = models[model_name][3]
                test_dataloader = models[model_name][4]
            for epoch in epochs:
                experiment_number += 1
                print(f"[INFO] Experiment number: {experiment_number}")
                print(f"[INFO] Model: {model_name}")
                print(f"[INFO] Data size: {dataloader}%")
                print(f"[INFO] Number of epochs: {epoch}")

                model = models[model_name][0]
                # Setup loss fn and optimizer
                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

                # Train
                engine.train(
                    model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    writer=utils.create_writer(
                        experiment_name=f"{dataloader}%_data",
                        model_name=model_name,
                        extra=f"{epoch}_epochs"
                    ),
                    epochs=epoch,
                    device=device
                )

                # Save model
                save_name = f"Pretrained_{model_name}_{dataloader}%_data_{epoch}_epochs.pth" # noqa 5501
                utils.save_model(model=model,
                                 target_dir="models",
                                 model_name=save_name)

                print(50*"-")


if __name__ == "__main__":
    main()
