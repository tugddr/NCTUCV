import os
import numpy as np
import json
import cv2

dir = "/mydrive/YOLO"
file = "result.json"

def convert(filepath,box):
  image = cv2.imread(filepath)
  box[0] *= int(image.shape[1])
  box[1] *= int(image.shape[0])
  box[2] *= int(image.shape[1])
  box[3] *= int(image.shape[0])
  x_diff = int(box[2]/2)
  y_diff = int(box[3]/2)
  box[0] = box[0]-x_diff
  box[1] = box[1]-y_diff
  box[2] += box[0]
  box[3] += box[1]
  return box


if __name__ =="__main__":
  with open('result.json', 'r', encoding='utf-8') as f:
    results = json.load(f)
  
  outputs = []
  for result in results:
    filepath = result["filename"]
    boxes = []
    scores = []
    labels = []
    dict = {}

    for obj in result["objects"]:
      img = filepath[10:-4]
      bb = obj["relative_coordinates"]
      box = [bb["center_x"],
            bb["center_y"],
            bb["width"],
            bb["height"]]
      box = np.array(box)
      box = convert(filepath,box)
      
      boxes.append((box[1],box[0],box[3],box[2]))

      scores.append(obj["confidence"])

      label = obj["class_id"]
      if label==0:
        label = 10
      labels.append(label)

    dict["bbox"] = boxes
    dict["score"] = scores
    dict["label"] = labels

    outputs.append([img,dict])
  f.close()

  outputs.sort(key=lambda x: x[0])
  out = []
  for o in outputs:
    out.append(o[1])

  with open('xx0616080.json', 'w', encoding='utf-8') as f:
    json.dump(out,f)
  f.close()
