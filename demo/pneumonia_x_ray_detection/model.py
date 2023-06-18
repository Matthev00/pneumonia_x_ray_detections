import torch
from torchvision import models, transforms


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
