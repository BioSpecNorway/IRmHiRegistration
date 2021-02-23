## Remarks

1. We have encountered issues when writing a huge histological image to a file using PIL. For such a case, pyvips worked like a charm. However, for simplicity, we excluded pyvips from the dependencies of the current project.
2. In the original paper, we tested different grayscale transformations of H&E images. The luminance conversion based on the linear formula performed equally well compared to more sophisticated approaches based on deconvolution. Therefore, we didn't include deconvolution-based grayscale transformations.
3. Registration API (ABCRegistrator, LandmarkRegistrator, etc.) was created to solve the author's needs. If you have suggestions on how to improve it please create PR.
