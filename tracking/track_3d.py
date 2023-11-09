'''
  3D track management
  Reactivate: When a confirmed trajectory is occluded and in turn cannot be associated with any detections for several frames, it 
  is then regarded as a reappeared trajectory.
'''
import numpy as np


class TrackState:
    Tentative = 1  # 不确定的
    Confirmed = 2  # 确定的
    Deleted = 3    # 删除
    Reactivate = 4  # 重新激活


class TrackState3Dor2D:
    Tracking_3D = 1
    Tracking_2D = 2


class Track_3D:
    def __init__(self, pose, kf_3d, track_id_3d, n_init, max_age, additional_info, feature=None):
        self.pose = pose
        self.kf_3d = kf_3d
        self.track_id_3d = track_id_3d
        self.hits = 1  # 成功跟踪次数
        self.age = 1
        self.state = TrackState.Tentative
        self.n_init = n_init
        self._max_age = max_age
        self.is3D_or_2D_track = TrackState3Dor2D.Tracking_3D
        self.additional_info = additional_info
        self.time_since_update = 0  # 消失的时间
        self.fusion_time_update = 0

    def predict_3d(self, trk_3d):
        self.pose = trk_3d.predict()

    def update_3d(self, detection_3d):
        self.kf_3d.update(detection_3d.bbox, detection_3d.additional_info[6])
        self.additional_info = detection_3d.additional_info
        # self.pose = self.kf_3d.ckf.x[:7]
        self.pose = np.concatenate((self.kf_3d.ckf.x[:4].squeeze(), self.kf_3d.pose), axis=0)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative
        if self.fusion_time_update >= 2:
            self.state = TrackState.Reactivate

    def unmatch_update_3d(self):
        self.kf_3d.unmatch_update()
        self.pose = np.concatenate((self.kf_3d.ckf.x[:4].squeeze(), self.kf_3d.pose), axis=0)

    def state_update(self):
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate
        elif self.time_since_update >= 1 and self.state != TrackState.Reactivate:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Reactivate and self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def fusion_state(self):
        if self.fusion_time_update >= 2:
            self.state = TrackState.Deleted

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_track_id_3d(self):
        track_id_3d = self.track_id_3d
        return track_id_3d
