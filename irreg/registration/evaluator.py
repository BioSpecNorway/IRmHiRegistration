from typing import Tuple

import numpy as np
from numpy import linalg
import SimpleITK as sitk


def registration_errors(
    tx: sitk.Transform,
    reference_fixed_point_list: tuple,
    reference_moving_point_list: tuple,
    display_errors: bool = False,
    min_err: float = None,
    max_err: float = None,
    figure_size: tuple = (18, 14),
    filename: str = None,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Distances between points transformed by the given transformation and their
    location in another coordinate system. When the points are only used to
    evaluate registration accuracy (not used in the registration) this is the
    Target Registration Error (TRE).

    Args:
        tx (SimpleITK.Transform): The transform we want to evaluate.
        reference_fixed_point_list (list(tuple-like)): Points in fixed image
                                                       cooredinate system.
        reference_moving_point_list (list(tuple-like)): Points in moving image
                                                        cooredinate system.
        display_errors (boolean): Display a 3D figure with the points from
                                  reference_fixed_point_list color corresponding
                                  to the error.
        min_err, max_err (float): color range is linearly stretched between min_err
                                  and max_err. If these values are not given then
                                  the range of errors computed from the data is used.
        figure_size (tuple): Figure size in inches.

    Returns:
     (mean, std, min, max, errors) (float, float, float, float, [float]):
      TRE statistics and original TREs.
    """
    transformed_fixed_point_list = [
        tx.TransformPoint(p) for p in reference_fixed_point_list
    ]

    errors = np.array(
        [
            linalg.norm(np.array(p_fixed) - np.array(p_moving))
            for p_fixed, p_moving in zip(
                transformed_fixed_point_list, reference_moving_point_list
            )
        ]
    )
    min_errors = np.min(errors)
    max_errors = np.max(errors)

    if display_errors:
        import matplotlib.pyplot as plt
        import matplotlib

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111)
        if not min_err:
            min_err = min_errors
        if not max_err:
            max_err = max_errors

        collection = ax.scatter(
            list(np.array(transformed_fixed_point_list).T)[0],
            list(np.array(transformed_fixed_point_list).T)[1],
            marker="o",
            c=errors,
            vmin=min_err,
            vmax=max_err,
            cmap=matplotlib.cm.coolwarm,
            label="transformed fixed points",
        )
        plt.colorbar(collection, shrink=0.8)

        ax.scatter(
            list(np.array(reference_moving_point_list).T)[0],
            list(np.array(reference_moving_point_list).T)[1],
            marker="o",
            c="green",
            label="reference moving points",
        )

        plt.title(
            f"registration errors (mean+-std): {np.mean(errors):.2f}+-{np.std(errors):.2f}",
            x=0.7,
            y=1.05,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.legend()

        if filename:
            plt.savefig(filename)
        plt.close()

    return np.mean(errors), np.std(errors), min_errors, max_errors, errors
