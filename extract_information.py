import cv2
import easyocr
import numpy as np

def extract_information_from_image(img, coordinates_dict, **kwargs):

    def preprocess_image(image):
        image = cv2.medianBlur(image, 3)
        image = image * 0.7
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        alpha = 1.5
        beta = 0
        image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
        image = cv2.medianBlur(image, 5)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        return image

    
    allowed_list = " 0123456789(){}[]\'\"\\/.,_"
    def check(input:str):
        for letter in input:
            if not (letter in allowed_list):
    #             print('check list: ', letter)
                return '--'
        return input


    if kwargs.get("reader") is None:
        reader = easyocr.Reader(["en"])
    else:
        reader = kwargs.get("reader")

    img = preprocess_image(img)
    img = cv2.resize(img, (1280, 720))
    #cv2.imwrite("saved_test_3.jpg", img=img)

    #print(coordinates_dict)

    res = {}
    for key, value in coordinates_dict.items():
        temp = reader.readtext(img[value[1]:value[3], value[0]:value[2]], detail=0)
        print(temp)
        try:
            res[key] = (
                check(input=temp[0])
                if len(temp) >=1 else '--'
            )
        except:
            res[key] = '--'
    return res

if __name__ == "__main__":
    img = cv2.imread("ex.jpeg")
    print(extract_information_from_image(img, {
    "map": [141, 378, 272, 447],
    "heart_rate": [71, 80, 415, 242],
    "spo2": [93, 469, 424, 605],
    "sbp": [18, 310, 195, 405],
    "rr": [303, 646, 452, 726],
    "dbp": [257, 317, 442, 398]
  }))

