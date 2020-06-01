import streamlit as st
import numpy as np
import json
import random
import cv2
import torch
from PIL import Image
import os
from collections import Counter
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor


target_classes = ['cockle', 'tuatua', 'mussel']
dir_path = os.path.dirname(os.path.abspath(__file__))

# TRAIN_JSON = dir_path + "/data/aug_val/annotation.json"
# TRAIN_PATH = dir_path + "/data/aug_val"
# MODEL_WEIGHTS = "/Users/sunbowen/Desktop/shellfish_detection/data/logs/model_final.pth"
MODEL_WEIGHTS = dir_path + "/model/model_final.pth"
CONFIG_FILE = "model/config.yaml"

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


class ShellfishVisualizer(Visualizer):
    # Draw shellfish classes and count on images
    def draw_class_count(self, predictions):
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        class_names = self.metadata.get("thing_classes", None)
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
            counts = Counter(labels)
            text_string = ''
            x = self.output.height * 0.02
            y = self.output.width * 0.02
            for key in counts:
                text_string += str(key) + ': ' + str(counts[key]) + ' ' if text_string == '' else '\n' + str(
                    key) + ': ' + str(counts[key]) + ' '

            self.draw_text(text_string, (x, y), horizontal_alignment='left', color='green')


# @st.cache(allow_output_mutation=True)
def create_predictor(model_config, model_weights, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(model_config)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATASETS.TEST = ("shellfish",)
    predictor = DefaultPredictor(cfg)

    return cfg, predictor


def make_inference(image, model_config, model_weights, threshold=0.7, save=False):
    """
    Makes inference on image (single image) using model_config, model_weights and threshold.
    Returns image with n instance predictions drawn on.
    Params:
    -------
    image (str) : file path to target image
    model_config (str) : file path to model config in .yaml format
    model_weights (str) : file path to model weights
    threshold (float) : confidence threshold for model prediction, default 0.5
    save (bool) : if True will save image with predicted instances to file, default False
    """
    # Create predictor and model config
    cfg, predictor = create_predictor(model_config, model_weights, threshold)

    # Convert PIL image to array
    image = np.asarray(image)
    outputs = predictor(image)

    v = ShellfishVisualizer(img_rgb=image,
                            metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(thing_classes=target_classes),
                            scale=1)
    # Draw shellfish classes and counts on the top left of the image
    v.draw_class_count(outputs["instances"].to("cpu"))

    # Draw on predictions bbox and masks
    visualized = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    return visualized.get_image(), outputs["instances"]


def main():
    st.title("Shellfish Detection")
    st.write("## Upload image")
    uploaded_image = st.file_uploader("Please choose a png or jpg image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Make sure image is RGB
        image = image.convert("RGB")

        if st.button("Make a prediction"):
            with st.spinner("Predicting..."):
                custom_pred, preds = make_inference(
                    image=image,
                    model_config=CONFIG_FILE,
                    model_weights=MODEL_WEIGHTS
                )

                classes = np.array(preds.pred_classes)
                predicted_classes = [target_classes[i] for i in classes]
                st.write("Detected Shellfish:")
                for (key, value) in Counter(predicted_classes).items():
                    st.write(str(key) + ': ' + str(value))

                st.image(custom_pred, caption="Result", use_column_width=True)


if __name__ == "__main__":
    main()
