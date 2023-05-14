from pytesseract import pytesseract


PATH_2_TESSERACT = "/usr/local/bin/tesseract"
pytesseract.tesseract_cmd = PATH_2_TESSERACT


def run_tesseract(img):
    """
    Run Google Tesseract OCR on the given image.
    :param img: np array; RGB format
    :return:
    """
    pred_str = pytesseract.image_to_string(img)  # get Tesseract output for the given image
    pred_str = parse_tesseract_ocr_output(pred_str)  # post-process Tesseract output
    return pred_str


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