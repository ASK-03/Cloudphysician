# Imports
import capture_img
from segmentation2polygon import segmentation2polygon, do_perspective_transformation
import queue
import logging
import threading
import cv2
from time import perf_counter, sleep
import os

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
        level=logging.DEBUG,
        format=format,
        handlers=[logging.FileHandler("pipeline-1.log")],
    )

    from ultralytics import YOLO

    logging.log(logging.INFO, "Loading the YOLO model")
    global model
    model = YOLO("./model_yolo/best.pt")
    logging.log(logging.INFO, "Loaded the YOLO model")

    logging.log(level=logging.INFO, msg="Creating the directory structure!")
    if not os.path.exists("images"):
        capture_img.create_dir(path="images")
    if not os.path.exists("processed_images"):
        capture_img.create_dir(path="processed_images")
    logging.log(level=logging.DEBUG, msg=f"{os.getcwd()}")
    logging.log(level=logging.INFO, msg="Created the directory structure!")

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
            logging.log(level=logging.DEBUG, msg=f"Got the image path: {img_path}")
            segmentation = get_segmentation(img_path=img_path)
            logging.log(
                level=logging.INFO, msg="Got the segmentation mask of the image"
            )

            if segmentation is not None:
                img = cv2.imread(filename=img_path)
                processed_image = do_perspective_transformation(
                    image=img, input_array=segmentation
                )
                logging.log(level=logging.INFO, msg="Got the processed image")

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
