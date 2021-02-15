import unittest

from pyfld import Point, Segment

class TestSegment(unittest.TestCase):
    """Test the Segment class."""
    def test_property(self):
        x1 = 0
        y1 = 0
        x2 = 10
        y2 = 0
        seg = Segment(x1, y1, x2, y2)
        self.assertEqual(seg.p1, Point(x1, y1))
        self.assertEqual(seg.p2, Point(x2, y2))
        self.assertEqual(seg.length, 10)
        self.assertEqual(seg.angle, 0)

    def test_swap_x(self):
        x1 = 0
        x2 = 1
        y1 = 2
        y2 = 3
        seg = Segment(x1, y1, x2, y2)
        seg.swap_x()
        self.assertEqual(seg, Segment(x2, y1, x1, y2))

    def test_swap_y(self):
        x1 = 0
        x2 = 1
        y1 = 2
        y2 = 3
        seg = Segment(x1, y1, x2, y2)
        seg.swap_y()
        self.assertEqual(seg, Segment(x1, y2, x2, y1))
    
    def test_x_length(self):
        x1 = 0
        x2 = 1
        y1 = 2
        y2 = 3
        seg = Segment(x1, y1, x2, y2)
        self.assertEqual(seg.x_length, 1)

    def test_y_length(self):
        x1 = 0
        x2 = 1
        y1 = 2
        y2 = 3
        seg = Segment(x1, y1, x2, y2)
        self.assertEqual(seg.y_length, 1)