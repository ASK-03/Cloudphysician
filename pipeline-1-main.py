# Imports
import capture_img
from segmentation2polygon import segmentation2polygon, do_perspective_transformation
import queue
import logging
import threading
import cv2
from time import perf_counter, sleep
import os
import tensorflow as tf
import numpy as np
from time import perf_counter
import easyocr
from ultralytics import YOLO
from extract_information import extract_information_from_image

"""
TODO:
    - we have to add two options 
        [] - if only one image is given 
        [done] - if we have to do it using feed from webcam
    - the pipeline gets stopped if we get more number of points after segmentation add that.
"""

def initialize() -> None:
    """
    This function initializes the constant factors of the pipeline like the yolo model, logging, and to create a directory to store the images
    """
    format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.ERROR,
        format=format,
        handlers=[logging.FileHandler("pipeline-1.log")],
    )

    try:
        global model
        model = YOLO(model="./model_yolo/best.pt")
        logging.log(logging.INFO, "Loaded the YOLO model")
    except Exception as e:
        logging.log(logging.ERROR, e)
        logging.log(logging.ERROR, "Unable to load the YOLO model")
        exit(1)

    logging.log(level=logging.INFO, msg="Creating the directory structure!")
    if not os.path.exists("images"):
        capture_img.create_dir(path="images")
    if not os.path.exists("processed_images"):
        capture_img.create_dir(path="processed_images")
    logging.log(level=logging.DEBUG, msg=f"{os.getcwd()}")
    logging.log(level=logging.INFO, msg="Created the directory structure!")

    global reader
    reader = easyocr.Reader(["en"])

    global coordinates_dict
    coordinates_dict = {
        "2": {
            "heart_rate": [949, 58, 1082, 151],
            "map": [1209, 158, 1350, 230],
            "spo2": [942, 318, 1085, 423],
            "rr": [929, 420, 948, 530],
            "dbp": [1099, 159, 1178, 221],
            "sbp": [946, 162, 1058, 217],
        },
        "3": {
            "rr": [915, 532, 1247, 712],
            "dbp": [303, 562, 536, 750],
            "spo2": [937, 260, 1246, 445],
            "map": [563, 599, 801, 770],
            "heart_rate": [880, 41, 1240, 207],
            "sbp": [70, 550, 276, 765],
        },
        "4": {
            "map": [141, 378, 272, 447],
            "heart_rate": [71, 80, 415, 242],
            "spo2": [93, 469, 424, 605],
            "sbp": [18, 310, 195, 405],
            "rr": [303, 646, 452, 726],
            "dbp": [257, 317, 442, 398],
        },
        "1": {
            "map": [899, 581, 1108, 690],
            "heart_rate": [0, 0, 281, 185],
            "sbp": [681, 394, 955, 574],
            "spo2": [7, 345, 358, 573],
            "rr": [678, 48, 998, 249],
            "dbp": [1058, 386, 1252, 581],
        },
    }

    try:
        global classifier_model
        classifier_model = tf.keras.models.load_model(filepath="./classifier/model.h5")
        logging.log(logging.INFO, "Loaded the classifier model")
    except Exception as e:
        logging.log(logging.ERROR, e)
        logging.log(logging.ERROR, "Unable to load the classifier model")
        exit(1)

    sleep(2)


def get_segmentation(img_path: str):
    """
    This function returns the segmentation mask of the image
    Input: img_path: str
    Output: segmentation: np.ndarray
    """
    logging.log(logging.INFO, "Getting the segmentation mask of the image")
    start = perf_counter()
    results = model.predict(img_path)
    end = perf_counter()
    logging.log(
        logging.INFO,
        f"Time taken to get the segmentation mask of the image: {end-start}",
    )

    try:
        mask = results[0].masks
        mask = mask.xy
        segmentation = segmentation2polygon(mask[0])
        print(segmentation.reshape((-1, 2)))
        return segmentation.reshape((-1, 2))
    except Exception as e:
        logging.log(logging.ERROR, e)
        return None


def get_monitor_type(img):
    """
    This function returns the type of the image
    Input: img: np.ndarray
    Output: type: str
    """

    img = np.array(img).astype(dtype="float32") / 255.0
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    k = None
    if classifier_model is not None:
        preds = classifier_model.predict(x)
        k = np.argmax(preds)

    if k == 0:
        return "1"
    elif k == 1:
        return "2"
    elif k == 2:
        return "3"
    elif k == 3:
        return "4"
    else:
        return None


def main() -> None:
    img_queue = queue.Queue()

    img_capture_thread = threading.Thread(
        target=capture_img.capture_img, kwargs={"queue": img_queue}, daemon=True
    )
    img_capture_thread.start()

    while True:
        if 0xFF == ord("q"):
            break

        if not img_queue.empty():
            img_path = img_queue.get()
            logging.log(level=logging.DEBUG, msg=f"{img_queue.qsize()}")
            segmentation = get_segmentation(img_path=img_path)

            if segmentation is None or segmentation.shape[0] != 4:
                logging.log(
                    level=logging.WARNING,
                    msg="segmentation is either None or segmentataion is not apt.",
                )
                continue

            """
            TODO: check if the segmentation length is strictly equal to 4 or not if not then skip for now and create alert...
            """

            logging.log(
                level=logging.INFO, msg="Got the segmentation mask of the image"
            )

            if segmentation is not None:
                img = cv2.imread(filename=img_path)
                processed_image = do_perspective_transformation(
                    image=img, input_array=segmentation
                )
                logging.log(level=logging.INFO, msg="Got the processed image")

                monitor_type = get_monitor_type(processed_image)
                print(monitor_type)
                print(
                    extract_information_from_image(
                        img=processed_image,
                        coordinates_dict=coordinates_dict[monitor_type],
                        kwargs={"reader": reader},
                    )
                )

                try:
                    if not os.path.exists("processed_images"):
                        logging.log(
                            level=logging.INFO,
                            msg="Creating the directory to store the processed images",
                        )
                        os.mkdir("processed_images")
                    img_name = img_path.split("/")[-1]
                    cv2.imwrite(
                        filename=f"processed_images/{img_name}", img=processed_image
                    )
                    logging.log(level=logging.INFO, msg="Saved the processed image")

                except Exception as e:
                    logging.log(level=logging.ERROR, msg=e)
                    logging.log(
                        level=logging.ERROR, msg="Unable to save the processed image"
                    )
            img_queue.task_done()

    img_capture_thread.join()
    return


if __name__ == "__main__":
    initialize()
    main()
    del model
    exit(0)
