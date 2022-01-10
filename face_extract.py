import argparse
import os
from time import time

import align.detect_face as detect_face
import cv2
import numpy as np
from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir
from project_root_dir import project_dir
from tracker.sort import Sort
from insightface_func.face_detect_crop_multi import Face_detect_crop

logger = Logger()


def main():
    global colours, img_size
    args = parse_args()
    video_path = args.video_path
    output_path = args.output_path
    no_display = args.no_display
    detect_interval = args.detect_interval  # you need to keep a balance between performance and fluency
    margin = args.margin  # if the face is big in your video ,you can set it bigger for tracking easiler
    scale_rate = args.scale_rate  # if set it smaller will make input frames smaller
    show_rate = args.show_rate  # if set it smaller will dispaly smaller frames
    face_score_threshold = args.face_score_threshold
    crop_size = args.crop_size

    mkdir(output_path)
    # for display
    if not no_display:
        colours = np.random.rand(32, 3)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    logger.info('Start track and extract......')

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=None)

    filename = os.path.split(video_path)[-1]
    directoryname = os.path.join(output_path, filename.split('.')[0])
    logger.info('video_path:{}'.format(video_path))
    cam = cv2.VideoCapture(video_path)
    c = 0
    while True:
        final_faces = []
        additional_attribute_list = []
        ret, frame = cam.read()
        if not ret:
            logger.warning("ret false")
            break
        if frame is None:
            logger.warning("frame drop")
            break

        frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)

        if c % detect_interval == 0:
            img_size = np.asarray(frame.shape)[0:2]

            faces, points = app.detect(frame)

            face_nums = faces.shape[0]
            if face_nums > 0:
                face_list = []
                for i, item in enumerate(faces):
                    score = round(faces[i][4], 6)
                    if score > face_score_threshold:

                        face_list.append(item)

                        # use 5 face landmarks  to judge the face is front or side

                        facial_landmarks = np.squeeze(points[i, :])
                        if args.face_landmarks:
                            for (x, y) in facial_landmarks:
                                cv2.circle(frame, (int(x), int(y)), 3,
                                            (0, 255, 0), -1)

                        cropped = app.align(frame, points[i, :], crop_size)

                        # cv2.imwrite(str(i) + '.png', cropped)

                        dist_rate, high_ratio_variance, width_rate = judge_side_face(
                            np.array(facial_landmarks))
                        # print("----------------------")
                        # print(dist_rate, high_ratio_variance, width_rate)
                        # print("----------------------")
                        # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                        item_list = [
                            cropped, score, dist_rate, high_ratio_variance,
                            width_rate
                        ]
                    
                        additional_attribute_list.append(item_list)

                final_faces = np.array(face_list)

        trackers = tracker.update(final_faces, img_size, directoryname,
                                    additional_attribute_list,
                                    detect_interval)

        c += 1

        for d in trackers:
            if not no_display:
                d = d.astype(np.int32)
                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]),
                                colours[d[4] % 32, :] * 255, 3)
                if final_faces != []:
                    cv2.putText(frame, 'ID : %d  DETECT' % (d[4]),
                                (d[0] - 10, d[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                colours[d[4] % 32, :] * 255, 2)
                    cv2.putText(frame, 'DETECTOR', (5, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (1, 1, 1),
                                2)
                else:
                    cv2.putText(frame, 'ID : %d' % (d[4]),
                                (d[0] - 10, d[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                colours[d[4] % 32, :] * 255, 2)

        if not no_display:
            frame = cv2.resize(frame, (0, 0), fx=show_rate, fy=show_rate)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    tracker.write(directoryname)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        help='Path to the raw video.',
        default='videos')
    parser.add_argument('--output_path',
                        type=str,
                        help='Path to save face',
                        default='facepics')
    parser.add_argument('--detect_interval',
                        help='how many frames to make a detection',
                        type=int,
                        default=1)
    parser.add_argument('--margin',
                        help='add margin for face',
                        type=int,
                        default=10)
    parser.add_argument('--scale_rate',
                        help='Scale down or enlarge the original video img',
                        type=float,
                        default=1.0)
    parser.add_argument('--show_rate',
                        help='Scale down or enlarge the imgs drawn by opencv',
                        type=float,
                        default=1)
    parser.add_argument('--crop_size',
                        help='Scale down or enlarge the imgs drawn by opencv',
                        type=float,
                        default=224)
    parser.add_argument(
        '--face_score_threshold',
        help='The threshold of the extracted faces,range 0<x<=1',
        type=float,
        default=0.88)
    parser.add_argument(
        '--face_landmarks',
        help='Draw five face landmarks on extracted face or not ',
        action="store_true")
    parser.add_argument('--no_display',
                        help='Display or not',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
