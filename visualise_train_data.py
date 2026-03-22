import cv2
import os
import yaml
from pathlib import Path

with open("data/data.yaml", "r") as f:
    data = yaml.safe_load(f)
class_names = data["names"]


image_dir = "data/images/val"
label_dir = "data/labels/val"
output_dir = "train_data_visualizations"
os.makedirs(output_dir, exist_ok=True)


images = list(Path(image_dir).glob("*"))[:5]
for img_path in images:
    if not img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    label_path = Path(label_dir) / img_path.with_suffix(".txt").name

    if label_path.exists():
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")

        for label in labels:
            if not label:
                continue
            class_id, x_center, y_center, width, height = map(float, label.split())
            class_id = int(class_id)

            x_center *= w
            y_center *= h
            width *= w
            height *= h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    output_path = Path(output_dir) / f"gt_{img_path.name}"
    cv2.imwrite(str(output_path), img)
    print(f"Saved ground truth visualization to: {output_path}")