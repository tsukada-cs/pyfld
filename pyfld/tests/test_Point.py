import unittest

from pyfld import Point

class TestPoint(unittest.TestCase):
    """Test the Point class."""
    def test_property(self):
        x = 0
        y = 1
        seg = Point(x, y)
        self.assertEqual(seg.x, x)
        self.assertEqual(seg.y, y)
