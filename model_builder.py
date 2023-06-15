import torch
from torchvision import models, transforms


def create_effnetb2(num_classes: int,
                    device: torch.device = "cuda"):

    model_weights = models.EfficientNet_B2_Weights.DEFAULT
    model_transform = model_weights.transforms()
    model = models.efficientnet_b2(weights=model_weights).to(device)

    transform = transforms.Compose([transforms.Grayscale(3),
                                    model_transform])
    # # Set 1 color chanel
    # model.features[0][0] = torch.nn.Conv2d(in_channels=1,
    #                                        out_channels=32,
    #                                        kernel_size=(3, 3),
    #                                        stride=(2, 2),
    #                                        padding=(1, 1),
    #                                        bias=False)

    # Fit classfier to problem
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1408,
                        out_features=num_classes,
                        bias=True).to(device=device)
    )

    for param in model.features:
        param.requires_grad_ = False

    model.name = "effnetb2"
    return model, transform



