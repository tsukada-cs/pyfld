# pyfld
FastLineDetector implemented in Python.


## Description
In order to extract line segments, Lee et al., (2014) devised a simple but reliable extractor inspired from Bay et al., (2005).  
Lee et al., (2014) described it as follows.
> Given an image, Canny edges are detected first and the system extracts line segments as follows: At an edge pixel the extractor connects a straight line with a neighboring one, and continues fitting lines and extending to the next edge pixel until it satisfies co-linearity with the current line segment. If the extension meets a high curvature, the extractor returns the current segment only if it is longer than 20 pixels, and repeats the same steps until all the edge pixels are consumed. Then with the segments, the system incrementally merges two segments with length weight if they are overlapped or closely located and the difference of orientations is sufficiently small.

This package is designed to allow fine tuning of parameters based on this approach.

## Reference
* J. Han Lee, S. Lee, G. Zhang, J. Lim, W. Kyun Chung, I. Hong Suh. "Outdoor place recognition in urban environments using straight lines." In 2014 IEEE International Conference on Robotics and Automation (ICRA), pp.5550–5557. IEEE, 2014. [[Link to PDF]](http://cvlab.hanyang.ac.kr/~jwlim/files/icra14linerec.pdf)
* H. Bay, V. Ferraris, and L. Van Gool, “Wide-Baseline Stereo Matching with Line Segments.” In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), vol.1, no., pp.329-336, June 2005.
