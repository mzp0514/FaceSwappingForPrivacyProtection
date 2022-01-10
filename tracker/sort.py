"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function

import lib.utils as utils
import numpy as np
from tracker.data_association import associate_detections_to_trackers
from tracker.kalman_tracker import KalmanBoxTracker
from deepface import DeepFace
import cv2

logger = utils.Logger("MOT")


class Sort:
    def __init__(self, max_age=100000, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, img_size, root_dic, additional_attribute_list,
               predict_num):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE:as in practical realtime MOT, the detector doesn't run on every single frame
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()  # kalman predict ,very fast ,<1ms
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if dets != []:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
                dets, trks)
            # print("matched det:", len(matched), "unmatched det:",
            #       len(unmatched_dets), len(dets))
            # print("unmatched trk:", len(unmatched_trks), len(trks))

            unmatched_dets_cp = unmatched_dets.copy()
            unmatched_trks_cp = unmatched_trks.copy()

            for i in unmatched_dets_cp:
                for j in unmatched_trks_cp:
                    det_face = additional_attribute_list[i][0]
                    faces = [
                        attr[0]
                        for attr in self.trackers[j].face_additional_attribute
                    ]
                    similar_cnt = 0
                    unsimilar_cnt = 0
                    is_matched = False
                    for f in faces:
                        result = DeepFace.verify(det_face,
                                                 f,
                                                 model_name="Facenet",
                                                 enforce_detection=False)
                        print(result)
                        if result['verified']:
                            similar_cnt += 1
                            if similar_cnt > min(0.1 * len(faces), 2):
                                matched = np.append(matched, [[i, j]], axis=0)
                                is_matched = True
                                unmatched_dets = np.delete(
                                    unmatched_dets,
                                    np.where(unmatched_dets == i))
                                unmatched_trks = np.delete(
                                    unmatched_trks,
                                    np.where(unmatched_trks == j))
                                break
                        else:
                            unsimilar_cnt += 1
                            if unsimilar_cnt > min(0.1 * len(faces), 2):
                                break
                    if is_matched:
                        break
                    # exit()
            unmatched_dets_cp = unmatched_dets.copy()

            for i in unmatched_dets_cp:
                det_face = additional_attribute_list[i]
                print(det_face[1], det_face[2], det_face[3], det_face[4])
                if det_face[2] >= 1.4 or det_face[4] >= 1:
                    unmatched_dets = np.delete(unmatched_dets,
                                               np.where(unmatched_dets == i))
                print(unmatched_dets)

                cv2.imwrite("./data/tmp/p.png", det_face[0])
                for j in unmatched_trks:
                    attr = self.trackers[j].face_additional_attribute
                    for p, f in enumerate(attr):
                        cv2.imwrite("./data/tmp/%d_%d.png" % (j, p), f[0])

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])
                    det_face = additional_attribute_list[d[0]]
                    if det_face[2] < 1.4 and det_face[4] < 1:
                        trk.face_additional_attribute.append(
                            det_face)

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :])
                trk.face_additional_attribute.append(
                    additional_attribute_list[i])
                logger.info("new Tracker: {0}".format(trk.id + 1))
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if dets == []:
                trk.update([])
            d = trk.get_state()
            if (trk.time_since_update <
                    1) and (trk.hit_streak >= self.min_hits
                            or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(
                    1, -1))  # +1 as MOT benchmark requires positive
            i -= 1

            # remove dead tracklet
            if trk.time_since_update >= self.max_age or trk.predict_num >= 10 or d[
                    2] < 0 or d[3] < 0 or d[0] > img_size[1] or d[
                        1] > img_size[0]:
                print("----------------", trk.time_since_update,
                      trk.predict_num, d, "---------------")
                if len(trk.face_additional_attribute) >= 5:
                    utils.save_to_file(root_dic, trk)
                logger.info('remove tracker: {0}'.format(trk.id + 1))
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def write(self, root_dic):

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if len(trk.face_additional_attribute) >= 5:
                utils.save_to_file(root_dic, trk)
            logger.info('remove tracker: {0}'.format(trk.id + 1))
            self.trackers.pop(i)
