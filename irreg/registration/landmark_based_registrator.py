import time

import numpy as np
import SimpleITK as sitk

from irreg.registration.abc_registrator import ABCRegistrator
from irreg.registration.utils import landmarks_csv_reader


class LandmarkBasedRegistrator(ABCRegistrator):
    def __init__(
        self,
        fixed_landmarks_path: str,
        moving_landmarks_path: str,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        image_name: str,
        results_path: str,
        initial_transform_path: str = None,
        optimized_transform: sitk.Transform = sitk.AffineTransform(2),
        reference_image: sitk.Image = sitk.Image(),
        landmarks_reader=landmarks_csv_reader,
    ):
        super().__init__(
            fixed_image,
            moving_image,
            image_name,
            results_path,
            initial_transform_path=initial_transform_path,
            optimized_transform=optimized_transform,
        )
        fixed_fiducial_points_flat, moving_fiducial_points_flat = landmarks_reader(
            fixed_landmarks_path, moving_landmarks_path
        )
        self.fixed_landmarks = fixed_fiducial_points_flat
        self.moving_landmarks = moving_fiducial_points_flat

        self.reference_image = reference_image

    def register(self):
        start = time.time()

        self.optimized_transform = sitk.LandmarkBasedTransformInitializer(
            self.optimized_transform,
            self.fixed_landmarks,
            self.moving_landmarks,
            referenceImage=self.reference_image,
        )

        self.time = time.time() - start
