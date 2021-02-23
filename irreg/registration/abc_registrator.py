from abc import ABC, abstractmethod
import os

import numpy as np
from PIL import Image as PilImage
import SimpleITK as sitk

from irreg.normalizations import min_max_scaler, standard_scaler
from irreg.registration.evaluator import registration_errors
from irreg.registration.utils import (
    landmarks_csv_reader,
    blend_images,
)


class ABCRegistrator(ABC):
    def __init__(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        image_name: str,
        results_path: str,
        initial_transform_path: str = None,
        optimized_transform: sitk.Transform = sitk.AffineTransform(2),
    ):
        self.fixed_npy = fixed_image
        self.fixed_sitk = sitk.GetImageFromArray(fixed_image)
        self.moving_npy = moving_image
        self.moving_sitk = sitk.GetImageFromArray(moving_image)

        self.image_name = image_name

        self.results_path = results_path
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if initial_transform_path and os.path.exists(initial_transform_path):
            self.initial_transform = sitk.ReadTransform(initial_transform_path)
        else:
            self.initial_transform = None

        self.optimized_transform = optimized_transform

        self.tre_output_plot_path = os.path.join(
            self.results_path, f"{self.image_name}_tre.png"
        )
        self.registration_path = os.path.join(
            self.results_path, f"{self.image_name}_reg.png"
        )
        self.transform_path = os.path.join(
            self.results_path, f"{self.image_name}_transform.txt"
        )

        self.errors = {}
        self.time = None

        # indicated whether the registration failed or not
        self.failed = 0

    def evaluate(self, fixed_test_landmarks_path, moving_test_landmarks_path):
        fixed_test_landmarks, moving_test_landmarks = landmarks_csv_reader(
            fixed_test_landmarks_path, moving_test_landmarks_path, is_flat=False
        )
        (
            pre_errors_mean,
            pre_errors_std,
            pre_errors_min,
            pre_errors_max,
            errors,
        ) = registration_errors(
            self.final_transform,
            fixed_test_landmarks,
            moving_test_landmarks,
            display_errors=True,
            filename=self.tre_output_plot_path,
        )

        self.errors["errors"] = errors
        self.errors["mean"] = pre_errors_mean
        self.errors["std"] = pre_errors_std
        self.errors["min"] = pre_errors_min
        self.errors["max"] = pre_errors_max
        self.errors["rtre"] = errors / np.sqrt(
            self.moving_sitk.GetWidth() ** 2 + self.moving_sitk.GetHeight() ** 2
        )

    @property
    def final_transform(self):
        if self.initial_transform:
            # Compose cascaded transform
            transform = sitk.CompositeTransform(
                [self.initial_transform, self.optimized_transform]
            )
            transform.FlattenTransform()
        else:
            transform = self.optimized_transform

        return transform

    def write_results(self):
        img_sitk = blend_images(self.fixed_sitk, self.moving_sitk, self.final_transform)
        PilImage.fromarray(
            min_max_scaler(
                standard_scaler(sitk.GetArrayViewFromImage(img_sitk)), 0, 255
            ).astype("uint8")
        ).save(self.registration_path)
        self.final_transform.WriteTransform(self.transform_path)

    @abstractmethod
    def register(self):
        pass
