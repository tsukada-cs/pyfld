# pyfld
[![Build Status](https://github.com/tsukada-cs/pyfld/workflows/CI/badge.svg)](https://github.com/tsukada-cs/pyfld/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/tsukada-cs/pyfld/badge.svg)](https://coveralls.io/github/tsukada-cs/pyfld?branch=main)
[![PyPI version](https://img.shields.io/pypi/v/pyfld.svg)](https://pypi.python.org/pypi/pyfld/)

Python package for detecting line segments from images.

In order to extract line segments, Lee et al. (2014) devised a simple but reliable extractor inspired from Bay et al. (2005). Lee et al. (2014) described it as follows.
> Given an image, Canny edges are detected first and the system extracts line segments as follows: At an edge pixel the extractor connects a straight line with a neighboring one, and continues fitting lines and extending to the next edge pixel until it satisfies co-linearity with the current line segment. If the extension meets a high curvature, the extractor returns the current segment only if it is longer than 20 pixels, and repeats the same steps until all the edge pixels are consumed. Then with the segments, the system incrementally merges two segments with length weight if they are overlapped or closely located and the difference of orientations is sufficiently small.

This package is designed to allow fine tuning of parameters based on this approach.

## Instration
The currently recommended method of installation is via pip:
```shell
pip install pyfld
```

pyfld can also be installed by cloning the GitHub repository:
```shell
git clone https://github.com/tsukada-cs/pyfld
cd pyfld
pip install .
```

## Dependencies
* numpy >= 1.17.3
* opencv-coontrib-python >= 2.4

## Sample Usage
Standard use case:
```python
import numpy as np
from PIL import Image

from pyfld import FastLineDetector

img = Image.open("sample.png")
img = np.asarray(img.convert("L"))

length_threshold = 10
distance_threshold = 1.41421356
canny_th1 = 50
canny_th2 = 50
canny_aperture_size = 3
do_merge = False

fld = FastLineDetector(length_threshold, distance_threshold, canny_th1, canny_th2, canny_aperture_size, do_merge)
(x1, y1, x2, y2), _ = fld.detect(img)
```

If the `img` is already binarized, set `canny_aperture_size=0`. Then, the Canny method is not used, and line segment detection is performed directly on the input image.


Example of line segment visualization:
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(img, cmap="gray")
ax.plot([x1, x2], [y1, y2], c="r")
plt.show()
```
<img width="349" alt="FLD_output" src="https://user-images.githubusercontent.com/45615081/97328052-f47fd700-18b8-11eb-844f-949790c4aa5e.png">

## Reference
* J. Han Lee, S. Lee, G. Zhang, J. Lim, W. Kyun Chung, I. Hong Suh. "Outdoor place recognition in urban environments using straight lines." In 2014 IEEE International Conference on Robotics and Automation (ICRA), pp.5550–5557. IEEE, 2014. [[Link to PDF]](http://cvlab.hanyang.ac.kr/~jwlim/files/icra14linerec.pdf)
* H. Bay, V. Ferraris, and L. Van Gool, “Wide-Baseline Stereo Matching with Line Segments.” In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), vol.1, no., pp.329-336, June 2005. [[Link to PDF]](https://homes.esat.kuleuven.be/~konijn/publications/2005/CVPR-HB-05.pdf)
