import torch

import data_setup
import utils
import model_builder
import engine # noqa 5501

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

    model, model_transform = model_builder.create_densenet(
        num_classes=2,
        device=device
    )

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir,
        test_dir=test_dir,
        train_transforms=model_transform,
        test_transforms=model_transform,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    )

    imgs = list(test_dir.glob("*/*.jpeg"))
    img_path = random.sample(population=imgs, k=1)[0]
    img = Image.open(img_path)
    transformed_img = model_transform(img).unsqueeze(dim=0)

    best_model_path = "models/Pretrained_densenet_20%_data_10_epochs.pth"
    model.load_state_dict(torch.load(f=best_model_path))

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

    # loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # results = engine.train(model=model,
    #                        train_dataloader=train_dataloader,
    #                        test_dataloader=test_dataloader,
    #                        loss_fn=loss_fn,
    #                        optimizer=optimizer,
    #                        writer=utils.create_writer(experiment_name="test",
    #                                                   model_name="xyz")
    #                        )
    # print(results)
    # utils.plot_loss_curves(results=results)


if __name__ == "__main__":
    main()
