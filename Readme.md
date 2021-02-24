## Remarks

1. We have encountered issues when writing a huge histological image to a file using PIL. For such a case, pyvips worked like a charm. However, for simplicity, we excluded pyvips from the dependencies of the current project.
2. In the original paper, we tested different grayscale transformations of H&E images. The luminance conversion based on the linear formula performed equally well compared to more sophisticated approaches based on deconvolution. Therefore, we didn't include deconvolution-based grayscale transformations.
3. Registration API (ABCRegistrator, LandmarkRegistrator, etc.) was created to solve the author's needs. If you have suggestions on how to improve it please create PR.

## Citation
If you find this repository useful in your research, please consider citing the following paper:

```
@article{trukhan2020grayscale,
  title={Grayscale representation of infrared microscopy images by extended multiplicative signal correction for registration with histological images},
  author={Trukhan, Stanislau and Tafintseva, Valeria and T{\o}ndel, Kristin and Gro{\ss}erueschkamp, Frederik and Mosig, Axel and Kovalev, Vassili and Gerwert, Klaus and Kohler, Achim},
  journal={Journal of biophotonics},
  volume={13},
  number={8},
  pages={e201960223},
  year={2020},
  publisher={Wiley Online Library}
}
```
