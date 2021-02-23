from irreg.normalizations import min_max_scaler, standard_scaler
import time
import os
from abc import abstractmethod
import warnings

import numpy as np
from PIL import Image as ImagePil
import matplotlib.pyplot as plt
from IPython.display import clear_output
import SimpleITK as sitk

from irreg.registration.abc_registrator import ABCRegistrator
from irreg.registration.utils import blend_images


class SITKRegistrator(ABCRegistrator):
    def __init__(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        image_name: str,
        results_path: str,
        initial_transform_path: str = None,
        optimized_transform: sitk.Transform = sitk.AffineTransform(2),
        debug: bool = False,
    ):
        super().__init__(
            fixed_image,
            moving_image,
            image_name,
            results_path,
            initial_transform_path=initial_transform_path,
            optimized_transform=optimized_transform,
        )
        self.registration_method = sitk.ImageRegistrationMethod()
        self.metric_plot_output_path = os.path.join(
            self.results_path, f"{self.image_name}_MI.png"
        )
        self.debug = debug
        self.stop_condition_description = None


    @property
    def bspline_scale_factors(self):
        return None

    @abstractmethod
    def register_init(self):
        pass

    def register(self):
        """Register fixed and moving images.
        """
        self.register_init()

        if self.initial_transform:
            self.registration_method.SetMovingInitialTransform(self.initial_transform)

        if (
            self.optimized_transform.GetName() == "BSplineTransform"
            and self.bspline_scale_factors
        ):
            self.registration_method.SetInitialTransformAsBSpline(
                self.optimized_transform, scaleFactors=self.bspline_scale_factors
            )
        else:
            self.registration_method.SetInitialTransform(self.optimized_transform)

        # Connect all of the observers so that we can perform plotting during registration.
        self.registration_method.AddCommand(sitk.sitkStartEvent, self.start_plot)
        self.registration_method.AddCommand(sitk.sitkEndEvent, self.end_plot)
        self.registration_method.AddCommand(
            sitk.sitkMultiResolutionIterationEvent, self.update_multires_iterations
        )
        self.registration_method.AddCommand(sitk.sitkIterationEvent, self.plot_values)

        start = time.time()
        try:
            self.registration_method.Execute(self.fixed_sitk, self.moving_sitk)
        except Exception as e:
            self.failed = 1
            # reset the optimized transform
            self.optimized_transform = sitk.AffineTransform(2)
            warnings.warn(
                f"Exception during a registration procedure in sitk. Message : {str(e)}"
            )
            warnings.warn(
                f"Results path: {self.results_path}. Image name: {self.image_name}."
            )
        finally:
            self.stop_condition_description = (
                self.registration_method.GetOptimizerStopConditionDescription()
            )
        self.time = time.time() - start

    # Callback invoked when the StartEvent happens, sets up our new data.
    def start_plot(self):
        self.metric_values = []
        self.multires_iterations = []

    # Callback invoked when the EndEvent happens, do cleanup of data and figure.
    def end_plot(self):
        # render metric plot and return it as a file
        if not self.debug:
            plt.figure(figsize=(10, 10), dpi=300)
            plt.tight_layout()

        plt.plot(self.metric_values, "r")
        # filter boundary case when a new multires level was checked but no iteration was executed, e.g.
        # len(self.metric_values) is 5, while self.multires_iterations is [0, 5, 5].
        self.multires_iterations = list(
            filter(lambda x: x < len(self.metric_values), self.multires_iterations)
        )
        plt.plot(
            self.multires_iterations,
            [self.metric_values[index] for index in self.multires_iterations],
            "b*",
        )

        plt.xlabel("Iteration Number", fontsize=12)
        plt.ylabel("Metric Value", fontsize=12)

        plt.savefig(self.metric_plot_output_path)

        # Close figure, we don't want to get a duplicate of the plot latter on.
        plt.close()

    # Callback invoked when the IterationEvent happens, update our data and display new figure.
    def plot_values(self):
        self.metric_values.append(self.registration_method.GetMetricValue())

        if (
            self.registration_method.GetOptimizerIteration() % 10 != 0
        ) or not self.debug:
            return

        # Clear the output area (wait=True, to reduce flickering), and plot current data
        clear_output(wait=True)
        # Plot the similarity metric values

        _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 10))

        img_sitk = blend_images(self.fixed_sitk, self.moving_sitk, self.final_transform)
        img = min_max_scaler(
            standard_scaler(sitk.GetArrayViewFromImage(img_sitk)), 0, 255
        ).astype("uint8")

        img = np.asarray(
            ImagePil.fromarray(img).resize((img.shape[1] / 4, img.shape[0] / 4))
        )

        ax1.imshow(img)

        ax2.plot(self.metric_values, "r")
        ax2.plot(
            self.multires_iterations,
            [self.metric_values[index] for index in self.multires_iterations],
            "b*",
        )

        ax2.set_xlabel("Iteration Number", fontsize=12)
        ax2.set_ylabel("Metric Value", fontsize=12)

        plt.show()

    # Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
    # metric_values list.
    def update_multires_iterations(self):
        self.multires_iterations.append(len(self.metric_values))
