import json
from typing import List

# Classes the model is trained to detect.
# The index here is the class ID written into YOLO label files.
CLASSES = ['duckie', 'truck', 'sign']

# Images are resized to this square size before training.
IMAGE_SIZE = 416


def convert_labelme_json(json_path: str, img_w: int, img_h: int) -> List[str]:
    
    with open(json_path, 'r') as file:
        data = json.load(file)
        
    yolo = []
    
    for obj in data["shapes"]:
        if obj["label"] not in CLASSES:
            continue
        class_id = CLASSES.index(obj["label"])

        points = obj["points"]
        # extract from corner info
        xmin = min(points[0][0], points[1][0])
        xmax = max(points[0][0], points[1][0])
        ymin = min(points[0][1], points[1][1])
        ymax = max(points[0][1], points[1][1])
        
        # scale values
        xmin = xmin * IMAGE_SIZE / img_w
        xmax = xmax * IMAGE_SIZE / img_w
        ymin = ymin * IMAGE_SIZE / img_h
        ymax = ymax * IMAGE_SIZE / img_h
        
        # Compute normalized center and size
        cx = (xmin + xmax) / 2 / IMAGE_SIZE
        cy = (ymin + ymax) / 2 / IMAGE_SIZE
        w  = (xmax - xmin) / IMAGE_SIZE
        h  = (ymax - ymin) / IMAGE_SIZE
        yolo.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
           
    # {
    #   "label": "duckie",
    #   "shape_type": "rectangle",
    #   "points": [[120.0, 85.0], [210.0, 160.0]]
    # }
     
    # <class_id> <cx> <cy> <w> <h>
    
    return yolo
