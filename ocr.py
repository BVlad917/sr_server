import cv2
import base64
from pytesseract import Output
from pytesseract import pytesseract


PATH_2_TESSERACT = "/usr/local/bin/tesseract"
pytesseract.tesseract_cmd = PATH_2_TESSERACT


def find_upscale_ratio(height, width, min_height=70, min_width=210):
    if height < min_height or width < min_width:
        return max(min_height / height, min_width / width)
    return None


def resize_image(image):
    height, width = image.shape[:2]
    ratio = find_upscale_ratio(height=height, width=width)
    if ratio is not None:
        new_height = int(height * ratio)
        new_width = int(width * ratio)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return resized_image

    return image


def run_tesseract_string_only(img):
    """
    Run Google Tesseract OCR on the given image and return the string prediction(s). Does not return
    bounding boxes or confidences.
    :param img: np array; RGB format image
    :return: the string of the prediction(s)
    """
    pred_str = pytesseract.image_to_string(img)  # get Tesseract output for the given image
    pred_str = parse_tesseract_ocr_output(pred_str)  # post-process Tesseract output
    return pred_str


def run_tesseract(img):
    """
    Run Google Tesseract OCR on the given image and return all the information provided by the OCR system.
    :param img: np array; RGB format image
    :return: a dictionary with the following keys:
        - "text": list of strings; the words predicted by the OCR system
        - "confidence": list of floats: confidence for each predicted word, provided by the OCR system
        - "box": list of 4-tuples containing the bounding box in the format (x0, y0, x1, y1). Defines the
        upper left and lower right points
    """
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    ratio = find_upscale_ratio(*img.shape[:2])
    texts, boxes = [], []
    num_boxes = len(d['level'])
    for i in range(num_boxes):
        text = parse_tesseract_ocr_output(d['text'][i])
        # conf = d['conf'][i] / 100. if d['conf'][i] >= 0 else d['conf'][i]  # normalized confidence
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        upper_left = (x, y)
        upper_right = (x + w, y)
        lower_right = (x + w, y + h)
        lower_left = (x, y + h)
        polygon = [*upper_left, *upper_right, *lower_right, *lower_left]
        if ratio is not None:
            polygon = list(map(lambda c: int(c * ratio), polygon))

        # if the detector worked but the text recognition model didn't find anything => skip
        if not len(text):
            continue

        boxes.append(polygon)
        texts.append(text)

    img = resize_image(img)
    _, img_bytes = cv2.imencode('.png', img)
    image_data = base64.b64encode(img_bytes).decode('utf-8')
    return {"image": image_data, "boxes": boxes, "texts": texts}


def parse_tesseract_ocr_output(string):
    """
    Post-process a string returned by Tesseract OCR in any way we want
    """
    # string = string.replace('€', 'e')
    # string = string.strip()
    # string = string.lower()
    # string = unidecode(string)
    return string


NAME_2_FN = {"tesseract": run_tesseract, "mmocr": None}


def get_ocr_fn(model_name):
    assert model_name in NAME_2_FN, "ERROR: Invalid OCR system requested."
    return NAME_2_FN[model_name]
