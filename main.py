import torch
from torchvision import transforms, models

import data_setup
import utils

import os
from pathlib import Path
import random
from PIL import Image


def main():
    args = utils.parse_arguments()
    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs
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

    effnetb2_weights = models.EfficientNet_B2_Weights.DEFAULT
    effnetb2_transforms = effnetb2_weights.transforms
    effnetb2 = models.efficientnet_b2(weights=effnetb2_weights)

    densenet_weights = models.DenseNet121_Weights.DEFAULT
    densenet = models.densenet121(weights=densenet_weights)

    googlenet_weights = models.GoogLeNet_Weights.DEFAULT
    googlenet = models.GoogLeNet()


    imgs = list(test_dir.glob("*/*.jpeg"))
    img_path = random.sample(population=imgs, k=1)[0]
    img = Image.open(img_path)
    transformed_img = simple_transform(img).unsqueeze(dim=0)

    print(effnetb2.features[0][0])
    effnetb2.features[0][0] = torch.nn.Conv2d(1, 32,
                                           padding=(1, 1),
                                           bias=False,
                                           stride=(2, 2),
                                           kernel_size=(3, 3))
    print(effnetb2.features[0][0])
    effnetb2.to(device)
    effnetb2.eval()
    with torch.inference_mode():
        img_pred = effnetb2(transformed_img.to(device))
    label = torch.argmax(torch.softmax(img_pred, 1), 1)
    print(label)





if __name__ == "__main__":
    main()
