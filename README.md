# pyfld
FastLineDetector implemented in Python.


## Description
In order to extract line segments, Lee et al., (2014) devised a simple but reliable extractor.
Lee et al., (2014) described it as follows.
> Given an image, Canny edges are detected first and the system extracts line segments as follows: At an edge pixel the extractor connects a straight line with a neighboring one, and continues fitting lines and extending to the next edge pixel until it satisfies co-linearity with the current line segment. If the extension meets a high curvature, the extractor returns the current segment only if it is longer than 20 pixels, and repeats the same steps until all the edge pixels are consumed. Then with the segments, the system incrementally merges two segments with length weight if they are overlapped or closely located and the difference of orientations is sufficiently small.

This package is designed to allow fine tuning of parameters based on this approach.

## Reference
Jin Han Lee, Sehyung Lee, Guoxuan Zhang, Jongwoo Lim, Wan Kyun Chung, and Il Hong Suh. Outdoor place recognition in urban environments using straight lines. In 2014 IEEE International Conference on Robotics and Automation (ICRA), pages 5550–5557. IEEE, 2014. [[Link to PDF]](http://cvlab.hanyang.ac.kr/~jwlim/files/icra14linerec.pdf)
