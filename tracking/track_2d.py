import numpy as np


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reactivate = 4

class TrackState3Dor2D:
    Tracking_3D = 1
    Tracking_2D = 2

class Track_2D:
    def __init__(self,pose, kf_2d,track_id,min_frame,max_frame):
        self.pose=pose
        self.kf_2d=kf_2d
        self.track_id_2d = track_id  #
        self.hits = 1
        self.age = 1
        self.state = TrackState.Tentative
        self.is3D_or_2D_track = TrackState3Dor2D.Tracking_2D  # 2D tracking
        self.time_since_update = 0
        self.fusion_time_update = 0
        self.n_init = min_frame    # 连续n_init帧被检测到，状态就被设为confirmed
        self._max_age = max_frame  # 一个跟踪对象丢失多少帧后会被删去（删去之后将不再进行特征匹配）

    def update_2d(self, det):
        self.kf_2d.update(det.bbox)
        self.pose=np.concatenate(self.kf_2d.kf.x[:4],axis=0)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        if self.hits>= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative
        if  self.fusion_time_update >= 3:
            self.state = TrackState.Reactivate

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate
        elif self.time_since_update >= 1 and self.state != TrackState.Reactivate:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Reactivate and self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def predict_2d(self, trk_2d):
        self.pose = trk_2d.predict()

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def x1y1x2y2(self):
        """
        Get current position in bounding box format `(min x, miny, max x, max y)`.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def to_tlwh(self):
        """
        Get current position in bounding box format `(top left x, top left y, width, height)`.
        Returns
        """
        ret = self.pose[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret