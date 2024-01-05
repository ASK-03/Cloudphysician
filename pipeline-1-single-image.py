from ultralytics import YOLO
from classifier_model import predict_monitor
from segmentation2polygon import segmentation2polygon, do_perspective_transformation
import cv2
from classifier_model import predict_monitor
import json
from extract_information import extract_information_from_image
import argparse

with open('data.json', 'r') as f:
	content = json.load(f)
parser = argparse.ArgumentParser(description="CloudPhysician")
parser.add_argument('-i', '--img', help='Path to image')

def extract_info(image):
	yolo_model = YOLO("./model_yolo/best.pt")
	yolo_result = yolo_model.predict(image)
	
	yolo_mask = yolo_result[0].masks.xy
	
	approx_polygon = segmentation2polygon(yolo_mask[0]).reshape((-1, 2))
	
	perspective_image = do_perspective_transformation(image, approx_polygon)
	
	class_number = predict_monitor(perspective_image)
	if class_number == "1":
		coordinates_dict = content['first']
	elif class_number =="2":
		coordinates_dict = content['second']
	elif class_number =="3":
		coordinates_dict = content['third']
	elif class_number =="4":
		coordinates_dict = content['fourth']
	else:
		exit(0)
	return extract_information_from_image(image, coordinates_dict)
	
	
	
if __name__ == '__main__':
	args=parser.parse_args()
	img = cv2.imread(args.img)
	print(extract_info(img))
	

