import torch
from torchvision import transforms

import data_setup
import utils
import model_builder

import os
from pathlib import Path
import random
from PIL import Image


def main():
    args = utils.parse_arguments()
    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs # noqa 5501
    NUM_WORKERS = os.cpu_count()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_dir = Path("data/10_percent_data")
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    simple_transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor()])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir,
        test_dir=test_dir,
        train_transforms=simple_transform,
        test_transforms=simple_transform,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    )

    model, model_transform = model_builder.create_densenet(
        num_classes=2,
        device=device
    )

    imgs = list(test_dir.glob("*/*.jpeg"))
    img_path = random.sample(population=imgs, k=1)[0]
    img = Image.open(img_path)
    transformed_img = model_transform(img).unsqueeze(dim=0)

    model.eval()
    with torch.inference_mode():
        img_pred = model(transformed_img.to(device))
    label = torch.argmax(torch.softmax(img_pred, 1), 1)
    print(class_names[label])

    utils.pred_and_plot_image(model=model,
                              image_path=img_path,
                              class_names=class_names,
                              transform=model_transform)
    # utils.print_summary(model)


if __name__ == "__main__":
    main()
