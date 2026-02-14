import numpy as np
import math


class GapCornerDetector:
    def __init__(
        self,
        car_width=0.28,
        safety_margin=0.2,
        gap_distance=2.0,
        corner_threshold=0.45,
        max_lidar_dist=6.0,
        radians_per_elem=0.00436332312998582,
        preprocess_conv_size=3,
        range_offset=180,
    ):
        self.car_width = car_width
        self.safety_margin = safety_margin + car_width / 2
        self.gap_distance = gap_distance
        self.corner_threshold = corner_threshold
        self.max_lidar_dist = max_lidar_dist
        self.radians_per_elem = radians_per_elem
        self.preprocess_conv_size = preprocess_conv_size
        self.range_offset = range_offset

    # -------------------------------
    # Preprocessing
    # -------------------------------
    def preprocess_lidar(self, ranges):
        """Returns front FOV ranges, left→right"""

        ranges = np.array(ranges)

        # Front 180°
        proc = ranges[self.range_offset:-self.range_offset]

        proc[np.isinf(proc)] = self.max_lidar_dist
        proc[np.isnan(proc)] = self.max_lidar_dist

        # Smoothing
        kernel = np.ones(self.preprocess_conv_size) / self.preprocess_conv_size
        proc = np.convolve(proc, kernel, mode="valid")

        proc = np.clip(proc, 0, self.max_lidar_dist)

        center_index = len(proc) // 2

        # Flip so left → right
        return proc[::-1], center_index

    # -------------------------------
    # Corner detection
    # -------------------------------
    def detect_corner(self, ranges):
        """Simple near-wall latch"""

        ranges = np.array(ranges)

        right = ranges[:self.range_offset]
        left = ranges[-self.range_offset:]

        return (
            np.any(right < self.corner_threshold)
            or np.any(left < self.corner_threshold)
        )

    # -------------------------------
    # Gap detection
    # -------------------------------
    def detect_gap(self, proc_ranges, center_index):
        best_index = None
        best_width = 0.0
        best_score = -np.inf

        i = 0
        n = len(proc_ranges)

        while i < n:

            if proc_ranges[i] <= self.gap_distance:
                i += 1
                continue

            start = i
            while i < n and proc_ranges[i] > self.gap_distance:
                i += 1
            end = i - 1

            gap_len = end - start + 1
            mid = (start + end) // 2
            mid_dist = proc_ranges[mid]

            gap_width = gap_len * mid_dist * self.radians_per_elem

            # Reject narrow gaps
            if gap_width < self.safety_margin:
                continue

            # Best point in gap (deepest)
            slice_ = proc_ranges[start:end + 1]
            max_dist = np.max(slice_)
            best_idxs = np.where(slice_ >= (max_dist - 0.1))[0]
            local_best = start + int(np.median(best_idxs))

            center_penalty = abs(local_best - center_index)
            score = max_dist - 0.01 * center_penalty

            if score > best_score:
                best_score = score
                best_index = local_best
                best_width = gap_width

        return best_index, best_width

    # -------------------------------
    # Public API
    # -------------------------------
    def analyze(self, ranges):
        """
        Main entry point for state machine
        """

        proc_ranges, center_index = self.preprocess_lidar(ranges)

        is_cornering = self.detect_corner(ranges)

        best_idx, best_width = self.detect_gap(proc_ranges, center_index)

        return {
            "has_gap": best_idx is not None,
            "best_gap_index": best_idx,
            "best_gap_width": best_width,
            "is_cornering": is_cornering,
        }