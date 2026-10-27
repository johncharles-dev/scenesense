"""Thin adapter so Ultralytics' ByteTrack can track any detector's boxes (here RT-DETR)."""
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


class _Dets:
    """The minimal 'results' interface BYTETracker reads: conf, cls, xywh, len(), boolean indexing."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls
        self.xywh = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                                     xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Dets(self.xyxy[idx], self.conf[idx], self.cls[idx])


class ByteTrack:
    def __init__(self, frame_rate=12, high=0.25, low=0.1, new=0.25, buffer=30, match=0.8):
        args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=high, track_low_thresh=low,
                               new_track_thresh=new, track_buffer=buffer, match_thresh=match, fuse_score=True)
        self.tracker = BYTETracker(args)

    def update(self, xyxy, conf, cls):
        """Feed one frame's detections; returns (xyxy, track_id, score, cls) for tracked objects."""
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        out = self.tracker.update(_Dets(xyxy, np.asarray(conf, np.float32), np.asarray(cls, np.float32)))
        if len(out) == 0:
            return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), np.zeros(0, int)
        return out[:, :4], out[:, 4].astype(int), out[:, 5], out[:, 6].astype(int)