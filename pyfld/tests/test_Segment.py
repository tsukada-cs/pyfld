import unittest

from pyfld import Point, Segment

class TestSegment(unittest.TestCase):
    """Test the Segment class."""
    def test_property(self):
        x1 = 0
        x2 = 1
        y1 = 2
        y2 = 3
        seg = Segment(x1, y1, x2, y2)
        self.assertEqual(seg.p1, Point(x1, y1))
        self.assertEqual(seg.p2, Point(x2, y2))

    def test_swap_x(self):
        x1 = 0
        x2 = 1
        y1 = 2
        y2 = 3
        seg = Segment(x1, y1, x2, y2)
        seg.swap_x()
        self.assertEqual(seg, Segment(x2, y1, x1, y2))

    def test_swap_x(self):
        x1 = 0
        x2 = 1
        y1 = 2
        y2 = 3
        seg = Segment(x1, y1, x2, y2)
        seg.swap_y()
        self.assertEqual(seg, Segment(x1, y2, x2, y1))