"""
    This is a python script to capture images from the camera
    and save them to the local directory in an interval of every 5 seconds.
"""

import cv2
import time
from datetime import datetime
import os
import logging


def create_dir(**kwargs) -> None:
    """
    This function creates a new directory if it does not exist
    the name of directory is images
    """
    import os

    if kwargs is not None and "path" in kwargs:
        path = kwargs["path"]
    else:
        path = "images"

    format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=format,
        handlers=[logging.FileHandler("pipeline-1-capture-img.log")],
    )

    if not os.path.exists(path):
        try:
            os.mkdir(path)
            logging.log(level=logging.INFO, msg=f"Directory {path} Created")
        except Exception as e:
            logging.log(level=logging.ERROR, msg=e)
            logging.log(level=logging.ERROR, msg=f"Unable to create directory {path}")
    else:
        logging.log(level=logging.INFO, msg=f"{path} directory already exists")


def capture_img(**kwargs) -> None:
    """
    This function captures images from the camera and saves them to the local directory
    in an interval of every 5 seconds.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.log(level=logging.ERROR, msg="Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if not os.path.exists(os.path.join(os.getcwd(), "images")):
        logging.log(
            level=logging.FATAL,
            msg="images directory not found, restart the script to create the directory",
        )
        exit(1)
    else:
        logging.log(level=logging.INFO, msg="images directory found")

    while True:
        ret, frame = cap.read()

        if ret == True:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"{os.getcwd()}/images/img_{timestamp}.jpg"

            cv2.imwrite(filename, frame)
            time.sleep(0.2)

            if kwargs is not None and "queue" in kwargs:
                queue = kwargs["queue"]
                queue.put(filename)

            logging.log(
                level=logging.INFO,
                msg=f"Image saved named img_{timestamp}.jpg in the directory images",
            )

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(5)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=format,
        handlers=[logging.FileHandler("pipeline-1.log")],
    )

    logging.log(level=logging.INFO, msg="Starting the pipeline")
    pipeline = (create_dir, capture_img)

    for func in pipeline:
        func()
