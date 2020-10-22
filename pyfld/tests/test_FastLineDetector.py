import unittest
from unittest import mock

import numpy as np

from pyfld import FastLineDetector, Point, Segment

class TestFastLineDetector(unittest.TestCase):
    """Test the FastLineDetector class."""
    def test__init__(self):
        with self.assertRaises(ValueError):
            fld = FastLineDetector(length_threshold=0)
        with self.assertRaises(ValueError):
            fld = FastLineDetector(distance_threshold=-1)
        with self.assertRaises(ValueError):
            fld = FastLineDetector(canny_th1=-1)
        with self.assertRaises(ValueError):
            fld = FastLineDetector(canny_th2=-1)

    def test_FLD_const_color(self, *mocks):
        fld = FastLineDetector()
        img = np.ones([30,30]).astype(np.uint8)
        segments = fld.detect(img)
        self.assertEqual(len(segments), 0)
    
    def test_FLD_line(self, *mocks):
        fld = FastLineDetector()
        img = np.zeros([30,30]).astype(np.uint8)
        img[5:-5, 20] = 255
        segments = fld.detect(img)
        self.assertEqual(len(segments), 2) # 2 because of Gibbs effect

    def test_FLD_edge_line(self, *mocks):
        fld = FastLineDetector(canny_aperture_size=0)
        img = np.zeros([30,30]).astype(np.uint8)
        img[5:-5, 20] = 255
        segments = fld.detect(img)
        self.assertEqual(len(segments), 1)
    
    def test_FLD_merge_lines(self, *mocks):
        fld = FastLineDetector(length_threshold=2, canny_aperture_size=0, do_merge=True)
        img = np.zeros([30,30]).astype(np.uint8)
        img[5:13, 15] = 255
        img[16:-5, 15] = 255
        segments = fld.detect(img)
        self.assertEqual(len(segments), 1)

    def test_get_chained_points_1(self, *mocks):
        fld = FastLineDetector()
        img = np.zeros([5,5])
        img[:,2] = 1
        ans_points = [Point(2,0), Point(2,1), Point(2,2), Point(2,3), Point(2,4)]
        points, img = fld.get_chained_points(img, Point(2,0))
        self.assertEqual(points, ans_points)
        self.assertTrue(np.all(img == np.zeros_like(img)))

    def test_get_chained_points_2(self, *mocks):
        fld = FastLineDetector()
        img = np.array([
            [0,0,1,0,0],
            [0,1,0,0,0],
            [0,1,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
        ])
        ans_points = [Point(2,0), Point(1,1), Point(1,2), Point(0,3), Point(0,4)]
        points, img = fld.get_chained_points(img, Point(2,0))
        self.assertEqual(points, ans_points)
        self.assertTrue(np.all(img == np.zeros_like(img)))

    def test_get_chained_points_3(self, *mocks):
        fld = FastLineDetector()
        img = np.array([
            [0,0,1,0,0],
            [0,1,0,1,0],
            [0,1,0,1,0],
            [1,0,0,0,1],
            [1,0,0,0,1],
        ])
        ans_points = [Point(2,0), Point(3,1), Point(3,2), Point(4,3), Point(4,4)]
        ans_img = np.array([
            [0,0,0,0,0],
            [0,1,0,0,0],
            [0,1,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
        ])
        points, img = fld.get_chained_points(img, Point(2,0))
        self.assertEqual(points, ans_points)
        self.assertTrue(np.all(img == ans_img))

    def test_get_chained_points_4(self, *mocks):
        fld = FastLineDetector()
        img = np.eye(5)
        img += np.fliplr(np.eye(5))
        ans_points = [Point(0,0), Point(1,1), Point(2,2), Point(3,3), Point(4,4)]
        ans_img = np.array([
            [0,0,0,0,1],
            [0,0,0,1,0],
            [0,0,0,0,0],
            [0,1,0,0,0],
            [1,0,0,0,0],
        ])
        points, img = fld.get_chained_points(img, Point(0,0))
        self.assertEqual(points, ans_points)
        self.assertTrue(np.all(img == ans_img))

    def test_get_chained_points_5(self, *mocks):
        fld = FastLineDetector()
        img = np.array([
            [1,0,0,0,0,0],
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,1,1],
        ])
        ans_points = [Point(0,0), Point(0,1), Point(1,2), Point(1,3),
                      Point(2,4), Point(3,5), Point(4,5), Point(5,5)]
        points, img = fld.get_chained_points(img, Point(0,0))
        self.assertEqual(points, ans_points)
        self.assertTrue(np.all(img == np.zeros_like(img)))

    def test_extract_segments_1(self, *mocks):
        fld = FastLineDetector(length_threshold=1, distance_threshold=0)
        points = [Point(0,0), Point(0,1), Point(1,2), Point(1,3),
                  Point(2,4), Point(3,5), Point(4,5), Point(5,5)]
        segs = [Segment(0,0,0,1), Segment(1,2,1,3), Segment(2,4,3,5), Segment(4,5,5,5)]
        segments = fld.extract_segments(points)
        self.assertTrue(np.all(np.round(segments) == np.array(segs)))

    def test_extract_segments_2(self, *mocks):
        fld = FastLineDetector(length_threshold=2, distance_threshold=0)
        points = [Point(0,0), Point(0,1), Point(1,2), Point(1,3),
                  Point(2,4), Point(3,5), Point(4,5), Point(5,5)]
        segs = [Segment(1,3,3,5)]
        segments = fld.extract_segments(points)
        self.assertTrue(np.all(np.round(segments) == np.array(segs)))

    def test_extract_segments_3(self, *mocks):
        fld = FastLineDetector(length_threshold=10, distance_threshold=0)
        points = [Point(0,0), Point(0,1), Point(1,2), Point(1,3),
                  Point(2,4), Point(3,5), Point(4,5), Point(5,5)]
        segments = fld.extract_segments(points)
        self.assertEqual(segments, [])

    def test_dist_point_line_1(self, *mocks):
        p1 = Point(0,0)
        p2 = Point(0,4)
        p3 = Point(2,2)
        dist = FastLineDetector.dist_point_line(p1, p2, p3)
        self.assertEqual(dist, 2)

    def test_dist_point_line_2(self, *mocks):
        p1 = Point(0,0)
        p2 = Point(0,0.1)
        p3 = Point(2,2)
        dist = FastLineDetector.dist_point_line(p1, p2, p3)
        self.assertEqual(dist, 2)

    def test_incidet_point_1(self, *mocks):
        p1 = Point(0,0)
        p2 = Point(4,4)
        p3 = Point(0,4)
        pt = FastLineDetector.get_incident_point(p1, p2, p3)
        self.assertEqual(pt, Point(2,2))
        
    def test_incidet_point_2(self, *mocks):
        p1 = Point(0,0)
        p2 = Point(0.1,0.1)
        p3 = Point(0,4)
        pt = FastLineDetector.get_incident_point(p1, p2, p3)
        self.assertEqual(pt, Point(2,2))
    
    def test_merge_segments_1(self, *mocks):
        seg1 = Segment(0,0,3,0)
        seg2 = Segment(5,0,10,0)
        fld = FastLineDetector()
        seg_merged = fld.merge_segments(seg1, seg2)
        self.assertEqual(seg_merged, Segment(0,0,10,0))

    def test_merge_segments_2(self, *mocks):
        seg1 = Segment(0,0,3,3)
        seg2 = Segment(5,5,10,10)
        fld = FastLineDetector()
        seg_merged = fld.merge_segments(seg1, seg2)
        self.assertEqual(seg_merged, Segment(0,0,10,10))

    def test_merge_segments_3(self, *mocks):
        seg1 = Segment(0,0,10,10)
        seg2 = Segment(40,40,50,50)
        fld = FastLineDetector()
        seg_merged = fld.merge_segments(seg1, seg2)
        self.assertEqual(seg_merged, None)