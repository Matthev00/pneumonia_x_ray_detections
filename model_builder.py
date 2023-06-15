import torch
from torchvision import models, transforms


def create_effnetb2(num_classes: int,
                    device: torch.device = "cuda"):

    model_weights = models.EfficientNet_B2_Weights.DEFAULT
    model_transform = model_weights.transforms()
    model = models.efficientnet_b2(weights=model_weights).to(device)

    transform = transforms.Compose([transforms.Grayscale(3),
                                    model_transform])

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


def create_densenet(num_classes: int,
                    device: torch.device = "cuda"):

    model_weights = models.DenseNet121_Weights.DEFAULT
    model_transform = model_weights.transforms()
    model = models.densenet121(weights=model_weights).to(device)

    transform = transforms.Compose([transforms.Grayscale(3),
                                    model_transform])

    # Fit classfier to problem
    model.classifier = torch.nn.Linear(
        in_features=1024,
        out_features=num_classes).to(device=device)

    for param in model.features:
        param.requires_grad_(False)

    model.name = "densenet"
    return model, transform


def create_googlenet(num_classes: int,
                     device: torch.device = "cuda"):

    weights = models.GoogLeNet_Weights.DEFAULT
    model_transform = weights.transforms()
    model = models.googlenet(weights).to(device)

    transform = transforms.Compose([transforms.Grayscale(3),
                                    model_transform])

    # Fit classfier to problem
    model.fc = torch.nn.Linear(
        in_features=1024,
        out_features=num_classes).to(device=device)

    for param in model.parameters():
        param.requires_grad_(False)
    model.fc.requires_grad_(True)

    model.name = "googlenet"
    return model, transform
