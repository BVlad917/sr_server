from pytesseract import Output
from pytesseract import pytesseract


PATH_2_TESSERACT = "/usr/local/bin/tesseract"
pytesseract.tesseract_cmd = PATH_2_TESSERACT


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
    texts, confidences, boxes = [], [], []
    num_boxes = len(d['level'])
    for i in range(num_boxes):
        text = parse_tesseract_ocr_output(d['text'][i])
        conf = d['conf'][i] / 100. if d['conf'][i] >= 0 else d['conf'][i]  # normalized confidence
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        (x_left, y_top, x_right, y_down) = (x, y, x + w, y + h)

        texts.append(text)
        confidences.append(conf)
        boxes.append((x_left, y_top, x_right, y_down))
    return {"text": texts, "confidence": confidences, "box": boxes}


def parse_tesseract_ocr_output(string):
    """
    Post-process a string returned by Tesseract OCR in any way we want
    """
    # string = string.replace('€', 'e')
    # string = string.strip()
    # string = string.lower()
    # string = unidecode(string)
    return string


NAME_2_FN = {"tesseract": run_tesseract_string_only, "mmocr": None}


def get_ocr_fn(model_name):
    assert model_name in NAME_2_FN, "ERROR: Invalid OCR system requested."
    return NAME_2_FN[model_name]
