import json
import os

from jsonschema import validate
import SimpleITK as sitk
import numpy as np

from irreg.registration.sitk_registrator import SITKRegistrator

SAMPLING_STRATEGY_MAP = {
    "none": sitk.ImageRegistrationMethod.NONE,
    "regular": sitk.ImageRegistrationMethod.REGULAR,
    "random": sitk.ImageRegistrationMethod.RANDOM,
}

INTERPOLATOR_MAP = {
    "none": sitk.sitkLinear,
    "linear": sitk.sitkLinear,
    "nearest_neighbor": sitk.sitkNearestNeighbor,
}

OPTIMIZER_MAP_PARAMS = {
    "never": sitk.ImageRegistrationMethod.Never,
    "once": sitk.ImageRegistrationMethod.Once,
    "each_iteration": sitk.ImageRegistrationMethod.EachIteration,
}

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schemas', 'sitk.schema')

def load_and_validate_params(params_path, schema_path=SCHEMA_PATH):
    with open(params_path, "r") as params_file, open(schema_path, "r"
    ) as schema_file:
        params = json.load(params_file)
        schema = json.load(schema_file)

    # check schema validity
    validate(params, schema)

    return params


class SitkRegistratorFromJson(SITKRegistrator):
    def __init__(
        self,
        params: dict,
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
            debug=debug
        )
        self.params = params

    @property
    def metric_sampling_percentage(self):
        if "metric_sampling_percentage" in self.params:
            return {
                "percentage": self.params["metric_sampling_percentage"]["percentage"],
                "seed": self.params["metric_sampling_percentage"]["seed"],
            }
        return None

    @property
    def mattes_mutual_information_metric(self):
        if "mattes_mutual_information_metric" in self.params:
            return {
                "numberOfHistogramBins": self.params[
                    "mattes_mutual_information_metric"
                ]["number_of_histogram_bins"]
            }
        return None

    @property
    def sampling_strategy(self):
        if "sampling_strategy" in self.params:
            return SAMPLING_STRATEGY_MAP[self.params["sampling_strategy"]]
        return SAMPLING_STRATEGY_MAP["none"]

    @property
    def interpolator(self):
        if "interpolator" in self.params:
            return INTERPOLATOR_MAP[self.params["interpolator"]]
        return INTERPOLATOR_MAP["none"]

    @property
    def gradient_descent_optimizer(self):
        if "gradient_descent_optimizer" in self.params:
            return {
                "learningRate": self.params["gradient_descent_optimizer"][
                    "learning_rate"
                ],
                "numberOfIterations": self.params["gradient_descent_optimizer"][
                    "number_of_iterations"
                ],
                "convergenceMinimumValue": self.params["gradient_descent_optimizer"][
                    "convergence_minimum_value"
                ],
                "convergenceWindowSize": self.params["gradient_descent_optimizer"][
                    "convergence_window_size"
                ],
                "estimateLearningRate": OPTIMIZER_MAP_PARAMS[
                    self.params["gradient_descent_optimizer"]["estimate_learning_rate"]
                ],
            }
        return None

    @property
    def conjugate_gradient_line_search_optimizer(self):
        if "conjugate_gradient_line_search_optimizer" in self.params:
            return {
                "learningRate": self.params["conjugate_gradient_line_search_optimizer"][
                    "learning_rate"
                ],
                "numberOfIterations": self.params[
                    "conjugate_gradient_line_search_optimizer"
                ]["number_of_iterations"],
                "convergenceMinimumValue": self.params[
                    "conjugate_gradient_line_search_optimizer"
                ]["convergence_minimum_value"],
                "convergenceWindowSize": self.params[
                    "conjugate_gradient_line_search_optimizer"
                ]["convergence_window_size"],
                "lineSearchLowerLimit": self.params[
                    "conjugate_gradient_line_search_optimizer"
                ]["line_search_lower_limit"],
                "lineSearchUpperLimit": self.params[
                    "conjugate_gradient_line_search_optimizer"
                ]["line_search_upper_limit"],
                "lineSearchEpsilon": self.params[
                    "conjugate_gradient_line_search_optimizer"
                ]["line_search_epsilon"],
                "lineSearchMaximumIterations": self.params[
                    "conjugate_gradient_line_search_optimizer"
                ]["line_search_maximum_iterations"],
                "estimateLearningRate": OPTIMIZER_MAP_PARAMS[
                    self.params["conjugate_gradient_line_search_optimizer"][
                        "estimate_learning_rate"
                    ]
                ],
                "maximumStepSizeInPhysicalUnits": self.params[
                    "conjugate_gradient_line_search_optimizer"
                ]["maximum_step_size_in_physical_units"],
            }
        return None

    @property
    def lbfgsb_optimizer(self):
        if "lbfgsb_optimizer" in self.params:
            return {
                "gradientConvergenceTolerance": self.params["lbfgsb_optimizer"][
                    "gradient_convergence_tolerance"
                ],
                "numberOfIterations": self.params["lbfgsb_optimizer"][
                    "number_of_iterations"
                ],
            }
        return None

    @property
    def lbfgs2_optimizer(self):
        if "lbfgs2_optimizer" in self.params:
            return {
                "solutionAccuracy": self.params["lbfgs2_optimizer"][
                    "solution_accuracy"
                ],
                "numberOfIterations": self.params["lbfgs2_optimizer"][
                    "number_of_iterations"
                ],
                "hessianApproximateAccuracy": self.params["lbfgs2_optimizer"][
                    "hessian_approximate_accuracy"
                ],
                "deltaConvergenceDistance": self.params["lbfgs2_optimizer"][
                    "delta_convergence_distance"
                ],
                "deltaConvergenceTolerance": self.params["lbfgs2_optimizer"][
                    "delta_convergence_tolerance"
                ],
                "lineSearchMaximumEvaluations": self.params["lbfgs2_optimizer"][
                    "line_search_maximum_evaluations"
                ],
                "lineSearchMinimumStep": self.params["lbfgs2_optimizer"][
                    "line_search_minimum_step"
                ],
                "lineSearchMaximumStep": self.params["lbfgs2_optimizer"][
                    "line_search_maximum_step"
                ],
                "lineSearchAccuracy": self.params["lbfgs2_optimizer"][
                    "line_search_accuracy"
                ],
            }
        return None

    @property
    def optimizer_scales_from_physical_shift(self):
        if "optimizer_scales_from_physical_shift" in self.params:
            return {
                "centralRegionRadius": self.params[
                    "optimizer_scales_from_physical_shift"
                ]["central_region_radius"],
                "smallParameterVariation": self.params[
                    "optimizer_scales_from_physical_shift"
                ]["small_parameter_variation"],
            }
        return None

    @property
    def multi_resolution_framework(self):
        if "multi_resolution_framework" in self.params:
            params = {
                "shrinkFactors": self.params["multi_resolution_framework"][
                    "shrink_factors_per_level"
                ],
                "smoothingSigmas": self.params["multi_resolution_framework"][
                    "smoothing_sigmas_per_level"
                ],
            }
            if "bspline_scale_factors" in self.params["multi_resolution_framework"]:
                params["scaleFactors"] = self.params["multi_resolution_framework"][
                    "bspline_scale_factors"
                ]
            return params
        return None

    @property
    def bspline_scale_factors(self):
        return (
            self.multi_resolution_framework["scaleFactors"]
            if self.multi_resolution_framework
            and "scaleFactors" in self.multi_resolution_framework
            else None
        )

    @property
    def smoothing_sigmas_are_specified_in_physical_units(self):
        if "smoothing_sigmas_are_specified_in_physical_units" in self.params:
            return self.params["smoothing_sigmas_are_specified_in_physical_units"]
        return None

    @property
    def bspline_transform_initializer(self):
        if "bspline_transform_initializer" in self.params:
            return {
                "transformDomainMeshSize": self.params["bspline_transform_initializer"][
                    "mesh_size"
                ],
                "order": self.params["bspline_transform_initializer"]["order"],
            }
        return None

    def register_init(self):
        if self.metric_sampling_percentage:
            self.registration_method.SetMetricSamplingPercentage(
                **self.metric_sampling_percentage
            )

        if self.mattes_mutual_information_metric:
            self.registration_method.SetMetricAsMattesMutualInformation(
                **self.mattes_mutual_information_metric
            )

        self.registration_method.SetMetricSamplingStrategy(self.sampling_strategy)

        self.registration_method.SetInterpolator(self.interpolator)

        if self.gradient_descent_optimizer:
            self.registration_method.SetOptimizerAsGradientDescent(
                **self.gradient_descent_optimizer
            )

        if self.conjugate_gradient_line_search_optimizer:
            self.registration_method.SetOptimizerAsConjugateGradientLineSearch(
                **self.conjugate_gradient_line_search_optimizer
            )

        if self.lbfgsb_optimizer:
            self.registration_method.SetOptimizerAsLBFGSB(**self.lbfgsb_optimizer)

        if self.lbfgs2_optimizer:
            self.registration_method.SetOptimizerAsLBFGS2(**self.lbfgs2_optimizer)

        if self.optimizer_scales_from_physical_shift:
            self.registration_method.SetOptimizerScalesFromPhysicalShift(
                **self.optimizer_scales_from_physical_shift
            )

        if self.multi_resolution_framework:
            self.registration_method.SetShrinkFactorsPerLevel(
                self.multi_resolution_framework["shrinkFactors"]
            )
            self.registration_method.SetSmoothingSigmasPerLevel(
                self.multi_resolution_framework["smoothingSigmas"]
            )

        if self.smoothing_sigmas_are_specified_in_physical_units:
            self.registration_method.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(
                self.smoothing_sigmas_are_specified_in_physical_units
            )

        if self.bspline_transform_initializer:
            self.optimized_transform = sitk.BSplineTransformInitializer(
                image1=self.fixed_sitk,
                transformDomainMeshSize=self.bspline_transform_initializer[
                    "transformDomainMeshSize"
                ],
                order=self.bspline_transform_initializer["order"],
            )
