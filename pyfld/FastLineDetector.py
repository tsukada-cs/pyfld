import copy

import cv2
import numpy as np

from pyfld.Exceptions import LineNotFound, PointChainNotFound


class Point(list):
    def __init__(self, x, y):
        super().__init__([x,y])
    @property
    def x(self):
        return self[0]
    @property
    def y(self):
        return self[1]

class Segment(list):
    def __init__(self, x1, y1, x2, y2):
        super().__init__([x1, y1, x2, y2])
    @property
    def x1(self):
        return self[0]
    @property
    def y1(self):
        return self[1]
    @property
    def x2(self):
        return self[2]
    @property
    def y2(self):
        return self[3]
    @property
    def angle(self):
        return np.arctan2(self.y2 - self.y1, self.x2 - self.x1)
    @property
    def length(self):
        return np.sqrt((self.x2-self.x1)**2 + (self.y2-self.y1)**2)
    @property
    def p1(self):
        return Point(self.x1, self.y1)
    @property
    def p2(self):
        return Point(self.x2, self.y2)
    def swap_x(self):
        self[0], self[2] = self[2], self[0]
    def swap_y(self):
        self[1], self[3] = self[3], self[1]

class FastLineDetector:    
    def __init__(self, length_threshold=10, distance_threshold=1.414213562,
                canny_th1=50.0, canny_th2=50.0, canny_aperture_size=3, do_merge=False):
        """
        Parameters
        ----------
         length_threshold : int, default 10
            Segment shorter than this will be discarded.
         distance_threshold : int or float, default 1.41421356
            A point placed from a hypothesis line segment farther than this will be regarded as an outlier.
         canny_th1 : int or float, default 50
            First threshold for hysteresis procedure in cv2.canny().
         canny_th2 : int or float, default 50
            Second threshold for hysteresis procedure in cv2.canny().
         canny_aperture_size : int, default 3
            Aperturesize for the sobel operator in cv2.canny().
            If zero, Canny() is not applied and the input image is taken as an edge image. 
         do_merge : bool, default False
            If true, incremental merging of segments will be perfomred.
        """
        if (length_threshold <= 0):
            raise ValueError("length_threshold must be positive.")
        if (distance_threshold < 0):
            raise ValueError("distance_threshold must not be negative.")
        if (canny_th1 < 0):
            raise ValueError("canny_th1 must not be negative.")
        if (canny_th2 < 0):
            raise ValueError("canny_th2 must not be negative.")

        self.length_threshold = length_threshold
        self.distance_threshold = distance_threshold
        self.canny_th1 = canny_th1
        self.canny_th2 = canny_th2
        self.canny_aperture_size = canny_aperture_size
        self.do_merge = do_merge

    def detect(self, image):
        """
        Detect lines in the input image.
        
        Parameters
        ----------
        image : numpy ndarray
            A grayscale input image (dtype=np.uint8).

        Reterns
        -------
        lines : numpy ndarray
            A vector of specifying the beginning and ending point of a line.
            Where vector is (x1, y1, x2, y2), point 1 is the start, point 2 is the end.
            Returned lines are directed so that the brighter side is placed on left.
        """
        image = np.array(image)
        segments = self.line_detection(image)
        if segments == []:
            raise LineNotFound("The image has no line segments")
        return np.array(segments).T

    def line_detection(self, src):
        self._h, self._w = src.shape
        if self.canny_aperture_size == 0:
            canny = src
        else:
            canny = cv2.Canny(image=src, threshold1=self.canny_th1, threshold2=self.canny_th2, apertureSize=self.canny_aperture_size)
        
        canny[:5, :5] = 0
        canny[self._h-5:, self._w-5:] = 0

        segments_all = []
        if np.all(canny == 0):
            return segments_all

        segments_tmp = []
        for r in range(self._h):
            for c in range(self._w):
                # Skip for non-seeds
                if canny[r,c] == 0:
                    continue
                # Found seeds
                pt = Point(c,r)
                # Get point chain
                points, canny = self.get_chained_points(canny, pt)

                if len(points) - 1 < self.length_threshold:
                    points = []
                    continue

                segments = self.extract_segments(points, xmin=0, xmax=self._w-1, ymin=0, ymax=self._h-1)

                if len(segments) == 0:
                    points = []
                    continue

                for seg in segments:
                    if seg.length < self.length_threshold:
                        continue
                    if (seg.x1 <= 4 and seg.x2 <= 4) or (seg.y1 <= 4 and seg.y2 <= 4) or (seg.x1 >= self._w-5 and seg.x2 >= self._w-5) or (seg.y1 >= self._h-5 and seg.y2 >= self._h-5):
                        continue
                    if self.do_merge is False:
                        segments_all.append(seg)
                    segments_tmp.append(seg)
                points = []
                segments = []
        
        if self.do_merge is False:
            return segments_all
        ith = len(segments_tmp) - 1
        jth = ith - 1
        while (ith >= 1 or jth >= 0):
            seg1 = segments_tmp[ith]
            seg2 = segments_tmp[jth]
            seg_merged = self.merge_segments(seg1, seg2)
            if seg_merged is None:
                jth -= 1
            else:
                segments_tmp[jth] = seg_merged
                del segments_tmp[ith]
                ith -= 1
                jth = ith - 1
            if jth < 0:
                ith -= 1
                jth = ith - 1
        segments_all = segments_tmp
        return segments_all

    def get_point_chain(self, img):
        """
        Parameters
        ----------
        img : numpy ndarray
            Input edge image.
        
        Reterns
        -------
        point_chain : list
            Extracted point chain.
        """
        point_chain = []
        if np.all(img == 0):
            raise PointChainNotFound("The image has no point chain")

        # rs, cs = np.where(img != 0)
        # for r, c in zip(rs, cs):
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                # Skip for non-seeds
                if img[r,c] == 0:
                    continue
                # Found seeds
                pt = Point(c,r)
                # Get point chain
                points, img = self.get_chained_points(img, pt)
                point_chain.append(points)
        return point_chain

    def get_chained_points(self, img, pt):
        """
        Parameters
        ----------
        img : numpy ndarray
            Input edge image.
        pt : Point
            Start point to search chained points.
        
        Reterns
        -------
        points : list of Point
            Extracted chained points.
        img : numpy ndarray
            Points removed image.
        """
        if img[pt.y, pt.x] == 0:
            raise ValueError("img[pt.y, pt.x] must not be 0")
        points = [pt]
        img[pt.y, pt.x] = 0

        direction = 0.0
        step = 0
        getting = True
        while(getting):
            pt, direction = self.get_chained_point(img, pt, direction, step)
            if pt is None:
                getting = False
                break
            points.append(pt)
            step += 1
            img[pt.y, pt.x] = 0
        return points, img

    def get_chained_point(self, img, pt, direction, step):
        """
        Find the neighboring edge point.
        Edge points closer to the direction are given priority.
        However, when step=0, it does not depend on the direction.
        Returns (None, None) if no edge point is found.

        Parameters
        ----------
        img : numpy ndarray
            Input edge image.
        pt : Point
            Start point to search the neighboring edge point.
        direction : int or float
            Previous search direction.
        step : int
            Current number of searches.
        
        Reterns
        -------
        point : Point
            Extracted neighbor chained point.
        direction : int or float
            Recalculated direction.
        """
        def direction_fixer(dir):
            if dir <= 180:
                return dir
            return dir - 360
        indices = {0:(1,1), 45:(1,0), 90:(1,-1), 135:(0,-1),
                   180:(-1,-1), 225:(-1,0), 270:(-1,1), 315:(0,1)} # Clockwise from lower right

        min_dir_diff = 315.0
        for i in (0, 45, 90, 135, 180, 225, 270, 315):
            ci = pt.x + indices[i][1]
            ri = pt.y + indices[i][0]
            if ri < 0 or ri == img.shape[0] or ci < 0 or ci == img.shape[1]:
                continue
            if img[ri, ci] == 0:
                continue
            if step == 0:
                chained_pt = Point(ci, ri)
                direction = direction_fixer(i)
                return chained_pt, direction
            if step > 0:
                curr_dir = direction_fixer(i)
                dir_diff = abs(curr_dir - direction)
                dir_diff = abs(direction_fixer(dir_diff))
                if dir_diff <= min_dir_diff:
                    min_dir_diff = dir_diff
                    consistent_pt = Point(ci, ri)
                    consistent_direction = direction_fixer(i)

        if min_dir_diff < 90.0:
            chained_pt = consistent_pt
            direction = (direction * step + consistent_direction) / (step + 1)
            return chained_pt, direction
        return None, None

    def extract_segments(self, points, xmin=0, xmax=None, ymin=0, ymax=None):
        """
        Extract segments from point chain.

        Parameters
        ----------
        points : list of Point
            Point chain for extraction.
        xmin : int or float, default 0
            Minimum x. Segment's x smaller than xmin are rounded to xmin.
        xmax : int or float, optional
            Minimum x. Segment's x larger than xmax are rounded to xmax.
        ymin : int or float, default 0
            Minimum y. Segment's y smaller than ymin are rounded to ymin.
        ymin : int or float, optional
            Minimum y. Segment's y smaller than ymax are rounded to ymax.
        """
        segments = []
        total = len(points)
        skip = 0
        for i in range(total-self.length_threshold):
            if skip > 0:
                skip -= 1
                continue

            ps = points[i]
            pe = points[i+self.length_threshold]

            is_line = True
            l_points = [ps]

            for j in range(1, self.length_threshold):
                pt = Point(points[i+j].x, points[i+j].y)
                dist = self.dist_point_line(ps, pe, pt)
                if dist > self.distance_threshold:
                    is_line = False
                    break
                l_points.append(pt)

            # Line check fail, test next point
            if is_line is False:
                continue

            l_points.append(pe)

            vx, vy, x, y = cv2.fitLine(np.array(l_points).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            a = Point(x.item(), y.item())
            b = Point(x.item() + vx.item(), y.item() + vy.item())
            ps = self.get_incident_point(a, b, ps, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

            # Extending line
            for j in range(self.length_threshold+1, total-i):
                pt = Point(points[i+j].x, points[i+j].y)
                dist = self.dist_point_line(a, b, pt)
                if dist > self.distance_threshold:
                    vx, vy, x, y = cv2.fitLine(np.array(l_points).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
                    a = Point(x.item(), y.item())
                    b = Point(x.item() + vx.item(), y.item() + vy.item())
                    dist2nd = self.dist_point_line(a, b, pt)
                    if dist2nd > self.distance_threshold:
                        j -= 1
                        break
                pe = pt
                l_points.append(pt)
            vx, vy, x, y = cv2.fitLine(np.array(l_points).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            a = Point(x.item(), y.item())
            b = Point(x.item() + vx.item(), y.item() + vy.item())
            e1 = self.get_incident_point(a, b, ps, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            e2 = self.get_incident_point(a, b, pe, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            segments.append(Segment(e1.x, e1.y, e2.x, e2.y))
            if self.length_threshold == 1:
                j = 1
            skip = j
        return segments

    @staticmethod
    def dist_point_line(p1, p2, p3):
        """
        Calcurate the distance between the line (p1 to p2) and point (p3).
        """
        u = np.array([p2.x - p1.x, p2.y - p1.y])
        v = np.array([p3.x - p1.x, p3.y - p1.y])
        l = abs(np.cross(u, v)/ np.linalg.norm(u))
        return l

    @staticmethod
    def get_incident_point(p1, p2, p3, xmin=0, xmax=None, ymin=0, ymax=None):
        """
        Calcurate the incident point on the line (p1 to p2) from point (p3).
        """
        l = np.cross([p1.x, p1.y, 1.0], [p2.x, p2.y, 1.0])
        lh = [l[0], l[1], 0.0]
        xk = [p3.x, p3.y, 1.0]

        lk = np.cross(xk, lh)
        xk = np.cross(lk, l)

        xk = xk/xk[2]

        if xk[0] < xmin:
            pt_x = xmin
        elif xmax is not None and xk[0] > xmax:
            pt_x = xmax
        else:
            pt_x = xk[0]

        if xk[1] < ymin:
            pt_y = ymin
        elif ymax is not None and xk[1] > ymax:
            pt_y = ymax
        else:
            pt_y = xk[1]

        return Point(pt_x, pt_y)

    def adjust_left_of_segment_to_be_higher(self, src, seg, gap=1, num_points=10):
        """
        Adjust the left side of the segment to be higher value.
        The line segment is divided into num_points and evaluated by the average
        of the values at vertically separated positions by a gap.

        Parameters
        ----------
        src : numpy ndarray
            Source image
        seg : Segment
            Segment to adjust.
        gap : int or float, default 1
            Distance from the segment when getting the value.
        num_points : int, default 10
            Divide the line segment into num_points and get the value.
        
        Reterns
        -------
        seg : Segment
            Adjusted segment.
        """
        if seg.x1 == 0 and seg.x2 == 0 and seg.y1 == 0 and seg.y2 == 0:
            return seg
        start = seg.p1
        end = seg.p2

        dx = end.x - start.x
        dy = end.y - start.y

        x10 = np.linspace(seg.x1, seg.x2, num_points)
        y10 = np.linspace(seg.y1, seg.y2, num_points)

        x10L = np.round(x10 + gap * np.cos(seg.angle + np.pi/2)).astype(int)
        y10L = np.round(y10 + gap * np.sin(seg.angle + np.pi/2)).astype(int)
        x10L_inboard = np.logical_and(x10L >= 0, x10L < src.shape[1])
        y10L_inboard = np.logical_and(y10L >= 0, y10L < src.shape[0])
        x10L = x10L[x10L_inboard * y10L_inboard]
        y10L = y10L[x10L_inboard * y10L_inboard]
        left = src[y10L, x10L].mean()

        x10R = np.round(x10 - gap * np.cos(seg.angle + np.pi/2)).astype(int)
        y10R = np.round(y10 - gap * np.sin(seg.angle + np.pi/2)).astype(int)
        x10R_inboard = np.logical_and(x10R >= 0, x10R < src.shape[1])
        y10R_inboard = np.logical_and(y10R >= 0, y10R < src.shape[0])
        x10R = x10R[x10R_inboard * y10R_inboard]
        y10R = y10R[x10R_inboard * y10R_inboard]
        right = src[y10R, x10R].mean()

        if right > left:
            seg.swap_x()
            seg.swap_y()

        return seg

    def merge_segments(self, seg1, seg2):
        """
        Merge segments if some conditions are passed.

        Parameters
        ----------
        seg1 : Segment
            Segment 1.
        seg2 : Segment
            Segment 2.

        Reterns
        -------
        seg_merged : Segment
            Merged segment.
        """
        p1 = Point(seg1.x1, seg1.y1)
        p2 = Point(seg1.x2, seg1.y2)
        
        seg1mid = Point((seg1.x1 + seg1.x2)/2, (seg1.y1 + seg1.y2)/2)
        seg2mid = Point((seg2.x1 + seg2.x2)/2, (seg2.y1 + seg2.y2)/2)

        seg1len = np.sqrt((seg1.x1 - seg1.x2)**2 + (seg1.y1 - seg1.y2)**2)
        seg2len = np.sqrt((seg2.x1 - seg2.x2)**2 + (seg2.y1 - seg2.y2)**2)

        mid_dist = np.sqrt((seg1mid.x - seg2mid.x)**2 + (seg1mid.y - seg2mid.y)**2)
        ang_diff = abs(seg1.angle - seg2.angle)

        dist = self.dist_point_line(p1, p2, seg2mid)
        
        if dist <= self.distance_threshold * 2 and mid_dist <= seg1len/2 + seg2len/2 + 20 and\
           ang_diff <= np.deg2rad(5.0):
           seg_merged = self.merge_lines(seg1, seg2)
           return seg_merged
        else:
            return None
    
    @staticmethod
    def merge_lines(seg1, seg2):
        """
        Merge line segments.

        Parameters
        ----------
        seg1 : Segment
            Segment 1.
        seg2 : Segment
            Segment 2.

        Reterns
        -------
        seg_merged : Segment
            Merged segment.
        """
        ax = seg1.x1
        ay = seg1.y1
        bx = seg1.x2
        by = seg1.y2

        cx = seg2.x1
        cy = seg2.y1
        dx = seg2.x2
        dy = seg2.y2

        dlix = bx - ax
        dliy = by - ay
        dljx = dx - cx
        dljy = dy - cy

        li = np.sqrt(dlix**2 + dliy**2)
        lj = np.sqrt(dljx**2 + dljy**2)

        xg = (li * (ax + bx) + lj * (cx + dx)) / (2 * (li + lj))
        yg = (li * (ay + by) + lj * (cy + dy)) / (2 * (li + lj))

        if dlix == 0:
            thi = np.pi/2
        else:
            thi = np.arctan(dliy / dlix)

        if dljx == 0:
            thj = np.pi/2
        else:
            thj = np.arctan(dljy / dljx) 

        if np.abs(thi - thj) <= np.pi / 2:
            thr = (li * thi + lj * thj) / (li + lj)
        else:
            tmp = thj - np.pi * (thj / abs(thj))
            thr = li * thi + lj * tmp
            thr /= (li + lj)
        
        axg = (ay - yg) * np.sin(thr) + (ax - xg) * np.cos(thr)
        bxg = (by - yg) * np.sin(thr) + (bx - xg) * np.cos(thr)
        cxg = (cy - yg) * np.sin(thr) + (cx - xg) * np.cos(thr)
        dxg = (dy - yg) * np.sin(thr) + (dx - xg) * np.cos(thr)
        
        delta1xg = min([axg, bxg, cxg, dxg])
        delta2xg = max([axg, bxg, cxg, dxg])

        delta1x = delta1xg * np.cos(thr) + xg
        delta1y = delta1xg * np.sin(thr) + yg
        delta2x = delta2xg * np.cos(thr) + xg
        delta2y = delta2xg * np.sin(thr) + yg

        seg_merged = Segment(delta1x, delta1y, delta2x, delta2y)
        return seg_merged