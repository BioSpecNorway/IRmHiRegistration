from typing import Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk


def blend_images(
    fixed_sitk: sitk.Image, moving_sitk: sitk.Image, transform: sitk.Transform = None
) -> sitk.Image:
    if not transform:
        transform = sitk.Transform()

    moving_sitk_resampled = sitk.Resample(
        moving_sitk,
        fixed_sitk,
        transform,
        sitk.sitkLinear,
        0.0,
        moving_sitk.GetPixelID(),
    )

    resulting_image = sitk.Compose(
        moving_sitk_resampled, fixed_sitk, fixed_sitk
    )  # red and cyan colors

    return resulting_image


def landmarks_csv_reader(
    fixed_landmarks_filename: str, moving_landmarks_filename: str, is_flat: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    # columns = id,x,y
    df_fixed = pd.read_csv(fixed_landmarks_filename)
    df_fixed = df_fixed.set_index(df_fixed.columns[0])

    df_moving = pd.read_csv(moving_landmarks_filename)
    df_moving = df_moving.set_index(df_moving.columns[0])

    if is_flat:
        return (
            df_fixed.values.ravel().astype(float).tolist(),
            df_moving.values.ravel().astype(float).tolist(),
        )
    return df_fixed.values.astype(float), df_moving.values.astype(float)
