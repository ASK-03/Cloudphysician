"""
 TODO: 
    1. get the .pt file of the yolo model and try to make it in such a way that it store output in our desired format
"""
from ultralytics import YOLO
from segmentation2polygon import segmentation2polygon, draw_points

model = YOLO("./model_yolo/best.pt")
results = model.predict("ex.jpeg")

mask = results[0].masks

mask = mask.xy
print(mask[0])
segmentation = segmentation2polygon(mask[0])
print(segmentation.reshape((-1, 2)))
#with open('mask.txt', 'w') as f:
    #f.write(mask[0])
draw_points(segmentation.reshape((-1, 2)))


