import gradio as gr
import os
import torch

from model import create_densenet
from timeit import default_timer as timer
from PIL import Image


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open("class_names.txt", "r") as filehandle:
        class_names = [class_name.strip() for class_name in filehandle.readlines()] # noqa 5501

    model, model_transform = create_densenet(num_classes=2,
                                             device=device)

    model.load_state_dict(torch.load(f='pretrained_model.pth',
                                     map_location=torch.device(device)))

    def predict(img: Image):

        start = timer()

        transformed_img = model_transform(img).unsqueeze(0).to(device)

        model.to(device)
        model.eval()
        with torch.inference_mode():

            y_pred = model(transformed_img)
            pred_probs = torch.softmax(y_pred, dim=1)

        pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))} # noqa 5501

        pred_time = round(timer() - start, 5)

        return pred_labels_and_probs, pred_time

    title = "Pneumonia Detection"
    description = "An Densenet121 feature extractor computer vision model to classify x_ray images in terms of pneumonia" # noqa 5501
    example_list = [["examples/" + example] for example in os.listdir("examples")] # noqa 5501

    demo = gr.Interface(fn=predict,
                        inputs=gr.Image(type="pil"),
                        outputs=[gr.Label(num_top_classes=1, label="Predictions"), # noqa 5501
                                 gr.Number(label="Prediction time (s)")],
                        examples=example_list,
                        description=description,
                        title=title,
                        examples_per_page=20)

    demo.launch()


if __name__ == "__main__":
    main()
