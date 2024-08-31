from transformers import pipeline, SamModel, SamProcessor
import torch
import numpy as np
import gradio as gr
from PIL import Image

# check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# we initialize model and processor
checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device=device)
sam_model = SamModel.from_pretrained("jadechoghari/robustsam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("jadechoghari/robustsam-vit-base")

def apply_mask(image, mask, color):
    """Apply a mask to an image with a specific color."""
    for c in range(3):  # Iterate over RGB channels
        image[:, :, c] = np.where(mask, color[c], image[:, :, c])
    return image

def query(image, texts, threshold):
    texts = texts.split(",")
    predictions = detector(
        image,
        candidate_labels=texts,
        threshold=threshold
    )
    
    image = np.array(image).copy()
    
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 165, 0),  # Orange
        (255, 0, 255)  # Magenta
    ]
    
    for i, pred in enumerate(predictions):
        score = pred["score"]
        if score > 0.5:
            box = [round(pred["box"]["xmin"], 2), round(pred["box"]["ymin"], 2),
                   round(pred["box"]["xmax"], 2), round(pred["box"]["ymax"], 2)]

            inputs = sam_processor(
                image,
                input_boxes=[[[box]]],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = sam_model(**inputs)

            mask = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0][0][0].numpy()
            
            color = colors[i % len(colors)]  # cycle through colors
            image = apply_mask(image, mask > 0.5, color)

    result_image = Image.fromarray(image)
    
    return result_image

title = """
# RobustSAM
"""

description = """
**Welcome to RobustSAM by Snap Research.**

This Space uses **RobustSAM**, a robust version of the Segment Anything Model (SAM) with improved performance on low-quality images while maintaining zero-shot segmentation capabilities.

Thanks to its integration with **OWLv2**, RobustSAM becomes text-promptable, allowing for flexible and accurate segmentation, even with degraded image quality.

Try the example or input an image with comma-separated candidate labels to see the enhanced segmentation results.

For better results, please check the [GitHub repository](https://github.com/robustsam/RobustSAM).
"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    
    gr.Interface(
        query,
        inputs=[gr.Image(type="pil", label="Image Input"), gr.Textbox(label="Candidate Labels"), gr.Slider(0, 1, value=0.05, label="Confidence Threshold")],
        outputs=gr.Image(type="pil", label="Segmented Image")
    )

demo.launch()
