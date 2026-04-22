import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from simple_lama_inpainting import SimpleLama

from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T

def process_meme_mitigation(image_path, text_to_remove, replacement_text, output_path="final_result.jpg"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_config = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_weights = "weights/groundingdino_swint_ogc.pth"
    
    dino_model = load_model(dino_config, dino_weights, device=device)
    simple_lama = SimpleLama()
    
    image_source, image_tensor = load_image(image_path)
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=f"the word {text_to_remove}",
        box_threshold=0.2,
        text_threshold=0.2,
        device=device
    )
    
    if len(boxes) == 0:
        print(f"Mot '{text_to_remove}' non détecté. Annulation.")
        return None

    h, w, _ = image_source.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for box in boxes:
        
        box = box * torch.Tensor([w, h, w, h])
        cx, cy, bw, bh = box.tolist()
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        x2, y2 = int(cx + bw/2), int(cy + bh/2)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    
    print(f"--- Effacement de '{text_to_remove}' ---")
    img_pil = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask).convert("L")
    cleaned_img = simple_lama(img_pil, mask_pil)
    
    print(f"--- Ajout du nouveau texte: '{replacement_text}' ---")
    draw = ImageDraw.Draw(cleaned_img)
    
    try:
        font = ImageFont.truetype("/Library/Fonts/Impact.ttf", 40)
    except:
        font = ImageFont.load_default()

    box_ref = boxes[0] * torch.Tensor([w, h, w, h])
    tx, ty = int(box_ref[0] - box_ref[2]/2), int(box_ref[1] - box_ref[3]/2)
    
    for adj in range(-2, 3):
        draw.text((tx+adj, ty), replacement_text, font=font, fill="black")
        draw.text((tx, ty+adj), replacement_text, font=font, fill="black")
    draw.text((tx, ty), replacement_text, font=font, fill="white")

    cleaned_img.save(output_path)
    print(f"Terminé ! Image sauvegardée sous : {output_path}")
    return output_path

# --- EXEMPLE D'UTILISATION ---
#if __name__ == "__main__":
    #process_meme_mitigation(
        #image_path="meme_with_text.jpg",
        #text_to_remove="agro",
        #replacement_text="calm"
    #)