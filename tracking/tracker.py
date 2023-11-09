# Author: wangxy
# Emial: 1393196999@qq.com

import numpy as np

from tracking.cost_function import convert_3dbox_to_8corner
from tracking.cost_function import iou3d, iou_batch
from tracking.filter3d import KalmanBoxTracker, _CKF, CKF
from tracking.kalman_fileter_2d import Kalman_2D
from tracking.matching import associate_detections_to_trackers_fusion, linear_assignment
from tracking.track_2d import Track_2D
from tracking.track_3d import Track_3D


def associate_detections_to_tracks(tracks, detections, threshold):
    track_indices = list(range(len(tracks)))
    detection_indices = list(range(len(detections)))
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t, trk in enumerate(tracks):
        for d, det in enumerate(detections):
            iou_matrix[t, d] = iou_batch(trk.x1y1x2y2(), det.to_x1y1x2y2())  # det: 8 x 3, trk: 8 x 3

    matches = []
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 0]:
            unmatched_trackers.append(t)

    # Filter out those pairs with small IoU
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < threshold:
            unmatched_detections.append(m[1])
            unmatched_trackers.append(m[0])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_trackers), np.array(unmatched_detections)


def cost_cos(a, b):
    theta_a = a[3]
    l_a = a[4]
    w_a = a[5]
    h_a = a[6]
    theta_b = b[3]
    l_b = b[4]
    w_b = b[5]
    h_b = b[6]

    result = (theta_a * theta_b + l_a * l_b + w_a * w_b + h_a * h_b) / (
            np.sqrt(theta_a ** 2 + l_a ** 2 + w_a ** 2 + h_a ** 2) * np.sqrt(
        theta_b ** 2 + l_b ** 2 + w_b ** 2 + h_b ** 2))

    return result


def cost_mse(image, det, trk):
    l_det = det.additional_info[4] - det.additional_info[2]
    w_det = det.additional_info[5] - det.additional_info[3]
    l_trk = trk.additional_info[4] - trk.additional_info[2]
    w_trk = trk.additional_info[5] - trk.additional_info[3]

    center_det = np.array((det.additional_info[2] + l_det / 2, det.additional_info[3] + w_det / 2))
    center_trk = np.array((trk.additional_info[2] + l_trk / 2, trk.additional_info[3] + w_trk / 2))

    box_l = int(min(l_det, l_trk))
    box_w = int(min(w_det, w_trk))
    boxA = np.array(
        (center_det[0] - box_l / 2, center_det[1] - box_w / 2, center_det[0] + box_l / 2, center_det[1] + box_w / 2))
    boxB = np.array(
        (center_trk[0] - box_l / 2, center_trk[1] - box_w / 2, center_trk[0] + box_l / 2, center_trk[1] + box_w / 2))

    imgA = np.zeros((box_w, box_l, 3))
    imgB = np.zeros((box_w, box_l, 3))
    results = 0
    for l in range(box_l):
        for w in range(box_w):
            imgA[w, l, :] = image[int(boxA[1]) + w, int(boxA[0]) + l, :]
            imgB[w, l, :] = image[int(boxB[1]) + w, int(boxB[0]) + l, :]
            results += (imgA[w, l, 0] - imgB[w, l, 0]) ** 2 + (imgA[w, l, 1] - imgB[w, l, 1]) ** 2 + (
                    imgA[w, l, 2] - imgB[w, l, 2]) ** 2

    results = results / (box_l * box_w)
    results = np.exp(-1 / results)
    return results


def distance(detection, tracker):
    dist = np.sqrt(
        (detection[0] - tracker[0]) ** 2 + (detection[1] - tracker[1]) ** 2 + (detection[2] - tracker[2]) ** 2)
    return 1 / (1 + dist)


def associate(detections, trackers, pose):
    for i, t in enumerate(trackers):
        if len(trackers[i].pose) == 4:
            trackers[i].pose = np.concatenate((trackers[i].pose, pose), axis=0)

    dets_8corner = [convert_3dbox_to_8corner(det_tmp.bbox) for det_tmp in detections]
    if len(dets_8corner) > 0:
        dets_8corner = np.stack(dets_8corner, axis=0)
    else:
        dets_8corner = []

    trks_8corner = [convert_3dbox_to_8corner(trk_tmp.pose) for trk_tmp in trackers]
    if len(trks_8corner) > 0:
        trks_8corner = np.stack(trks_8corner, axis=0)
    if (len(trks_8corner) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(dets_8corner)), np.empty((0, 8, 3), dtype=int)

    iou_matrix = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float64)

    for d, det in enumerate(dets_8corner):
        for t, trk in enumerate(trks_8corner):
            # mse=cost_mse(image, detections[d], trackers[t])
            cos = cost_cos(detections[d].bbox, trackers[t].pose)
            dist = distance(detections[d].bbox, trackers[t].pose)
            iou = iou3d(det, trk)[0]
            s_d = detections[d].additional_info[6]
            s_t = trackers[t].additional_info[6]
            cost = (s_d ** 2 + s_t ** 2) / (s_d + s_t)
            w1 = 0.3 + 0.2 * cost
            w2 = 1 - w1 * 2
            iou_matrix[d, t] = w1 * (iou + dist) + w2 * cos
            if iou == 0:
                if dist <= 0.2:
                    iou_matrix[d, t] = -1000

    matches = []
    if min(iou_matrix.shape) > 0:
        matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(dets_8corner):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trks_8corner):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < 0:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def _associate(detections, trackers, pose, idx):
    for i, t in enumerate(trackers):
        if len(trackers[i].pose) == 4:
            trackers[i].pose = np.concatenate((trackers[i].pose, pose), axis=0)

    dets_8corner = [convert_3dbox_to_8corner(det_tmp.bbox) for det_tmp in detections]
    if len(dets_8corner) > 0:
        dets_8corner = np.stack(dets_8corner, axis=0)
    else:
        dets_8corner = []

    tracker = []
    for i in idx:
        tracker.append(trackers[i])

    trks_8corner = [convert_3dbox_to_8corner(trk_tmp.pose) for trk_tmp in tracker]
    if len(trks_8corner) > 0:
        trks_8corner = np.stack(trks_8corner, axis=0)
    if (len(trks_8corner) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(dets_8corner)), np.empty((0, 8, 3), dtype=int)

    iou_matrix = np.zeros((len(dets_8corner), len(idx)), dtype=np.float64)

    for d, det in enumerate(dets_8corner):
        for t, trk in enumerate(trks_8corner):
            # mse=cost_mse(image, detections[d], trackers[t])
            cos = cost_cos(detections[d].bbox, trackers[t].pose)
            dist = distance(detections[d].bbox, trackers[t].pose)
            iou = iou3d(det, trk)[0]
            s_d = detections[d].additional_info[6]
            s_t = trackers[t].additional_info[6]
            cost = (s_d ** 2 + s_t ** 2) / (s_d + s_t)
            w1 = 0.3 + 0.2 * cost
            w2 = 1 - w1 * 2
            iou_matrix[d, t] = w1 * (iou + dist) + w2 * cos
            if iou == 0:
                if dist <= 0.2:
                    iou_matrix[d, t] = -1000

    matches = []
    if min(iou_matrix.shape) > 0:
        matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(dets_8corner):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trks_8corner):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < 0:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        for i, m in enumerate(matches):
            matches[i, 1] = idx[matches[i, 1]]
    for i, um in enumerate(unmatched_trackers):
        unmatched_trackers[i] = idx[unmatched_trackers[i]]

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Tracker:
    def __init__(self, max_age, n_init):
        self.max_age = max_age
        self.n_init = n_init
        self.tracks_3d = []
        self.tracks_2d = []
        self.track_id_3d = 0  # The id of 3D track is represented by an even number.
        self.track_id_2d = 1  # The id of 3D track is represented by an odd number.
        self.unmatch_tracks_3d = []

    def predict_3d(self):
        for track in self.tracks_3d:
            track.predict_3d(track.kf_3d)

    def predict_2d(self):
        for track in self.tracks_2d:
            track.predict_2d(track.kf_2d)

    def update(self, detection_3D_fusion, detection_3D_only, calib_file, iou_threshold):
        # 1st Level of Association
        if len(self.tracks_3d) == 0:
            matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_detections_to_trackers_fusion(
                detection_3D_fusion, self.tracks_3d, iou_threshold=0.01)

        else:

            # matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = _associate_detections_to_trackers_fusion(detection_3D_fusion, self.tracks_3d, self.kf_3d.pose, iou_threshold=0.01)
            matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate(detection_3D_fusion,
                                                                                                 self.tracks_3d,
                                                                                                 self.kf_3d.pose)
            """
            _matched_fusion_idx, _unmatched_dets_fusion_idx, _unmatched_trks_fusion_idx = _associate(detection_3D_only,
                                                                                                     self.tracks_3d,
                                                                                                     self.kf_3d.pose,
                                                                                                     unmatched_trks_fusion_idx)
            """
            for detection_idx, track_idx in matched_fusion_idx:
                self.tracks_3d[track_idx].update_3d(detection_3D_fusion[detection_idx])
                self.tracks_3d[track_idx].state = 2
                self.tracks_3d[track_idx].fusion_time_update = 0
            """
            for _detection_idx, _track_idx in _matched_fusion_idx:
                self.tracks_3d[_track_idx].update_3d(detection_3D_only[_detection_idx])
                self.tracks_3d[_track_idx].state = 2
                self.tracks_3d[_track_idx].fusion_time_update = 0
            """
            for track_idx in unmatched_trks_fusion_idx:
                self.tracks_3d[track_idx].unmatch_update_3d()
                self.tracks_3d[track_idx].fusion_time_update += 1
                self.tracks_3d[track_idx].mark_missed()
        for detection_idx in unmatched_dets_fusion_idx:
            self._initiate_track_3d(detection_3D_fusion[detection_idx])

        self.tracks_3d = [t for t in self.tracks_3d if not t.is_deleted()]

    def initiate_track_3d(self, detection):
        self.kf_3d = KalmanBoxTracker(detection.bbox, detection.additional_info[6])
        self.additional_info = detection.additional_info
        # pose=self.kf_3d.ckf.x[:7]
        pose = np.concatenate((self.kf_3d.kf.x[:4, 0].squeeze(), self.kf_3d.pose), axis=0)
        self.tracks_3d.append(
            Track_3D(pose, self.kf_3d, self.track_id_3d, self.n_init, self.max_age, self.additional_info))
        self.track_id_3d += 2

    def _initiate_track_3d(self, detection):
        self.kf_3d = CKF(detection.bbox, detection.additional_info[6])
        self.additional_info = detection.additional_info
        # pose=self.kf_3d.ckf.x[:7]
        pose = np.concatenate((self.kf_3d.ckf.x[:4].squeeze(), self.kf_3d.pose), axis=0)
        self.tracks_3d.append(
            Track_3D(pose, self.kf_3d, self.track_id_3d, self.n_init, self.max_age, self.additional_info))
        self.track_id_3d += 2

    def _initiate_track_2d(self, detection):
        self.kf_2d = Kalman_2D(detection.tlwh)  # (top_left_x,top_left_y,w,h)
        pose = np.concatenate(self.kf_2d.kf.x[:4], axis=0)
        self.tracks_2d.append(Track_2D(pose, self.kf_2d, self.track_id_2d, self.n_init, self.max_age))
        self.track_id_2d += 2
