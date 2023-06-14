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

        # images which have a high-resolution ground truth provided
        with_hr_imgs = glob.glob(os.path.join(TESTS_DIR, "test_images/sr/with_hr/*.png"))
        self.with_hr_imgs_hr = [p for p in with_hr_imgs if os.path.basename(p).split('.')[0].split('_')[-1] == "hr"]
        self.with_hr_imgs_lr = [p for p in with_hr_imgs if os.path.basename(p).split('.')[0].split('_')[-1] == "lr"]

        # images without a high-resolution ground truth provided, so we test them against previous SR runs
        with_sr_imgs = glob.glob(os.path.join(TESTS_DIR, "test_images/sr/with_sr/*.jpg"))
        self.with_sr_imgs_sr = [p for p in with_sr_imgs if os.path.basename(p).split('.')[0].split('_')[-1] == "sr"]
        self.with_sr_imgs_lr = [p for p in with_sr_imgs if os.path.basename(p).split('.')[0].split('_')[-1] == "lr"]

    def test_matching_lr_hr(self):
        lr_paths = sorted(self.with_hr_imgs_lr)
        hr_paths = sorted(self.with_hr_imgs_hr)
        self.assertEqual(len(hr_paths), len(lr_paths), msg="ERROR: Every LR image must have a corresponding HR image.")
        hr_img_numbers = np.array([int(os.path.basename(p).split('.')[0].split('_')[-2]) for p in hr_paths])
        lr_img_numbers = np.array([int(os.path.basename(p).split('.')[0].split('_')[-2]) for p in lr_paths])
        self.assertTrue(np.all(hr_img_numbers == lr_img_numbers), msg="ERROR: LR/HR image numbers don't match.")

    def test_matching_lr_sr(self):
        lr_paths = sorted(self.with_sr_imgs_lr)
        sr_paths = sorted(self.with_sr_imgs_sr)
        self.assertEqual(len(sr_paths), len(lr_paths), msg="ERROR: Every LR image must have a corresponding SR image.")
        sr_img_numbers = np.array([int(os.path.basename(p).split('.')[0].split('_')[-2]) for p in sr_paths])
        lr_img_numbers = np.array([int(os.path.basename(p).split('.')[0].split('_')[-2]) for p in lr_paths])
        self.assertTrue(np.all(sr_img_numbers == lr_img_numbers), msg="ERROR: LR/SR image numbers don't match.")

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
            print(f"Testing SR model '{model_name.replace('_', ' ').upper()}' for ICDAR-2015...")
            sr_model = get_sisr_forward_fn(model_name)
            self._test_hr_similarity_score(self.with_hr_imgs_lr, self.with_hr_imgs_hr, sr_model, calculate_psnr, MIN_PSNR_THRESHOLD)

    def test_hr_ssim_all_models(self):
        for model_name in SR_MODELS:
            print(f"Testing SR model '{model_name.replace('_', ' ').upper()}' for ICDAR-2015...")
            sr_model = get_sisr_forward_fn(model_name)
            self._test_hr_similarity_score(self.with_hr_imgs_lr, self.with_hr_imgs_hr, sr_model, calculate_ssim, MIN_SSIM_THRESHOLD)

    def test_sr_against_old_runs(self):
        sr_model = get_sisr_forward_fn(SR_MODELS[0])
        for lr_path, sr_path in zip(sorted(self.with_sr_imgs_lr), sorted(self.with_sr_imgs_sr)):
            lr_img, sr_img = map(cv2.imread, (lr_path, sr_path))  # BGR
            lr_img, sr_img = map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (lr_img, sr_img))  # RGB
            self.assertEqual(lr_img.shape[0] * SR_SCALE_FACTOR, sr_img.shape[0])
            self.assertEqual(lr_img.shape[1] * SR_SCALE_FACTOR, sr_img.shape[1])

            current_sr_img = sr_model(lr_img)
            self.assertEqual(current_sr_img.shape[0], sr_img.shape[0])
            self.assertEqual(current_sr_img.shape[1], sr_img.shape[1])
            ssim = calculate_ssim(img1=current_sr_img, img2=sr_img, crop_border=SR_SCALE_FACTOR)
            self.assertGreaterEqual(ssim, 0.99, msg=f"Similarity score too small for image {sr_path}.\n "
                                                    f"Similarity score=({ssim:.2f}); Images should be the same.")


if __name__ == '__main__':
    unittest.main()
