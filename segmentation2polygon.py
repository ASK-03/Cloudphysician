# Imports
import numpy as np
import cv2


# Functions
def get_segmentation():
    """
    HELPER FUNCTION
    This function returns the segmentation mask of the image
    """
    return "0 0.55 0.255556 0.548437 0.258333 0.521875 0.258333 0.520312 0.261111 0.515625 0.261111 0.514063 0.263889 0.50625 0.263889 0.504687 0.266667 0.496875 0.266667 0.495313 0.269444 0.489062 0.269444 0.4875 0.272222 0.476562 0.272222 0.475 0.275 0.448438 0.275 0.446875 0.277778 0.432813 0.277778 0.43125 0.280556 0.43125 0.705556 0.432813 0.708333 0.50625 0.708333 0.507812 0.705556 0.51875 0.705556 0.520312 0.702778 0.529688 0.702778 0.53125 0.7 0.546875 0.7 0.548437 0.702778 0.69375 0.702778 0.695312 0.705556 0.715625 0.705556 0.717188 0.708333 0.734375 0.708333 0.735937 0.705556 0.7375 0.705556 0.7375 0.7 0.739062 0.697222 0.739062 0.680556 0.740625 0.677778 0.740625 0.616667 0.742188 0.613889 0.742188 0.258333 0.740625 0.255556"


def convert_str_to_array(string) -> list[float]:
    """
    HELPER FUNCTION
    This function converts the string to a list of floats
    Input: string: string of points
    Output: list of floats
    """
    return [float(i) for i in string.split(" ")]


def get_height_and_width_of_img(img_path) -> dict[str, int]:
    """
    HELPER FUNCTION
    This function returns the height and width of the image
    Input: img_path: path to the image
    Output: dict of height and width
    """
    img = cv2.imread(img_path)
    h, w, c = img.shape
    return {"height": h, "width": w}


def from_normalised_to_pixel(img_size: dict, contour):
    """
    function converts the normalised segmentation mask to pixel values
    Input: img_size: dict of height and width of image
           contour: list of points
    Output: res: list of points
    """
    height = img_size["height"]
    width = img_size["width"]
    res = []

    for i in range(len(contour[1::2])):
        x = int(contour[2 * i + 1] * width)
        y = int(contour[2 * i + 2] * height)
        res.append((x, y))

    return res


def segmentation2polygon(segmentation):
    """
    function converts the segmentation mask from yolo V8 to a 4 sided polygon
    Input: segmentation: list of points
    Output: approx: list of points
    """

    points = np.array(segmentation, dtype=np.int32)

    closed = True

    peri = cv2.arcLength(points, True)
    approx = cv2.approxPolyDP(points, 0.02 * peri, closed)

    return approx


def pipeline():
    """
    Function is the pipeline for the segmentation to polygon conversion
    """
    segmentation = get_segmentation()
    segmentation = convert_str_to_array(segmentation)
    img_size = get_height_and_width_of_img(
        "ex.jpeg"
    )
    polygon = from_normalised_to_pixel(img_size, segmentation)
    polygon = segmentation2polygon(polygon)

    return polygon


def draw_points(polygon) -> None:
    """
    Function to draw the points on the image, it shows the image in the new window with points
    Input: polygon: list of points
    Output: None
    """
    img = cv2.imread(
        "ex.jpeg"
    )
    for i in polygon:
        cv2.circle(img, (i[0], i[1]), 5, (0, 0, 255), -1)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def do_perspective_transformation(image, input_array):
    """
    Function to do perspective transformation
    Input: image: image on which perspective transformation is to be done
           input_array: list of points
    Output: result: image after perspective transformation
    """
    height, width = image.shape[:2]

    input_array = np.array(input_array, dtype=np.float32)
    output_array = np.array(
        [(0, 0), (0, height), (width, height), (width, 0)], dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(input_array, output_array)
    result = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return result


if __name__ == "__main__":
    '''approx_polygon = np.array([[  42 , 12],
 [  72 , 692],
 [1252 , 684],
 [1236 ,  10]])

    print(len(approx_polygon))
    draw_points(approx_polygon)
    perspective = do_perspective_transformation(
        cv2.imread(
            "ex.jpeg"
        ),
        approx_polygon,
    )
    cv2.imshow("perspective", perspective)
    #cv2.imwrite('ex.jpeg', perspective)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    
    print(segmentation2polygon(get_segmentation()))
