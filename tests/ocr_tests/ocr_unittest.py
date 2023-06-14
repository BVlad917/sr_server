import os
import cv2
import sys
import glob
import unittest
import pytesseract


TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")

sys.path.append(os.path.join(TESTS_DIR, "../"))

from ocr import run_tesseract_string_only
PATH_2_TESSERACT = "/usr/local/bin/tesseract"
pytesseract.tesseract_cmd = PATH_2_TESSERACT


class OcrUnittest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_paths = glob.glob(os.path.join(TESTS_DIR, "test_images/ocr/*.png"))

    def test_tesseract(self):
        for p in self.img_paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tesseract_output = run_tesseract_string_only(img)
            self.assertGreater(len(tesseract_output), 0, msg=f"ERROR: No texts detected in image {p}")


if __name__ == '__main__':
    unittest.main()
