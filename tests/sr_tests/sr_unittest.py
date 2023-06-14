import logging
import os
import sys
import glob
import unittest

import cv2
import numpy as np

TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
SR_MODELS = ["swin_ir", "esrgan", "real_esrgan", "bsrgan", "swin2_sr"]
SR_SCALE_FACTOR = 4
MIN_PSNR_THRESHOLD = 15.0
MIN_SSIM_THRESHOLD = 0.35

sys.path.append(os.path.join(TESTS_DIR, "../"))

from sisr import get_sisr_forward_fn
from utils.calculate_psnr_ssim import calculate_psnr, calculate_ssim


class SRUnittest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sr_tests_imgs = glob.glob(os.path.join(TESTS_DIR, "test_images/sr/with_hr/*.png"))
        self.hr_imgs = [p for p in sr_tests_imgs if os.path.basename(p).split('.')[0].split('_')[-1] == "hr"]
        self.lr_imgs = [p for p in sr_tests_imgs if os.path.basename(p).split('.')[0].split('_')[-1] == "lr"]

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_matching_lr_hr(self):
        lr_paths = sorted(self.lr_imgs)
        hr_paths = sorted(self.hr_imgs)
        self.assertEqual(len(hr_paths), len(lr_paths), msg="ERROR: Every LR image must have a corresponding HR image.")
        hr_img_numbers = np.array([int(os.path.basename(p).split('.')[0].split('_')[-2]) for p in hr_paths])
        lr_img_numbers = np.array([int(os.path.basename(p).split('.')[0].split('_')[-2]) for p in lr_paths])
        self.assertTrue(np.all(hr_img_numbers == lr_img_numbers), msg="ERROR: LR/HR image numbers don't match.")

    def _test_hr_similarity_score(self, lr_imgs_paths, hr_imgs_paths, sr_model, similarity_fn, similarity_min_threshold):
        for lr_path, hr_path in zip(sorted(lr_imgs_paths), sorted(hr_imgs_paths)):
            lr_img, hr_img = map(cv2.imread, (lr_path, hr_path))  # BGR
            lr_img, hr_img = map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (lr_img, hr_img))  # RGB
            self.assertEqual(lr_img.shape[0] * SR_SCALE_FACTOR, hr_img.shape[0])
            self.assertEqual(lr_img.shape[1] * SR_SCALE_FACTOR, hr_img.shape[1])

            sr_img = sr_model(lr_img)
            self.assertEqual(lr_img.shape[0] * SR_SCALE_FACTOR, sr_img.shape[0])
            self.assertEqual(lr_img.shape[1] * SR_SCALE_FACTOR, sr_img.shape[1])
            sim = similarity_fn(img1=hr_img, img2=sr_img, crop_border=SR_SCALE_FACTOR)
            self.assertGreater(sim, similarity_min_threshold, msg=f"Similarity score too small for image {hr_path}.\n "
                                                                  f"Similarity score=({sim:.2f}), threshold={similarity_min_threshold:.2f}.")

    def test_hr_pnsr_all_models(self):
        for model_name in SR_MODELS:
            print(f"Testing SR model '{model_name.replace('_', ' ').upper()}'...")
            sr_model = get_sisr_forward_fn(model_name)
            self._test_hr_similarity_score(self.lr_imgs, self.hr_imgs, sr_model, calculate_psnr, MIN_PSNR_THRESHOLD)

    def test_hr_ssim_all_models(self):
        for model_name in SR_MODELS:
            print(f"Testing SR model '{model_name.replace('_', ' ').upper()}'...")
            sr_model = get_sisr_forward_fn(model_name)
            self._test_hr_similarity_score(self.lr_imgs, self.hr_imgs, sr_model, calculate_ssim, MIN_SSIM_THRESHOLD)
