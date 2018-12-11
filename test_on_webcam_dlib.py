import sys, os, argparse
import numpy as np
import cv2
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
import mxnet.gluon.data.vision
import datasets
import mxnet.gluon.model_zoo as model_zoo
from mxnet import ndarray as nd
import datetime
import hopenet
from PIL import Image

import datasets, hopenet, utils

from skimage import io
import dlib

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    args = parser.parse_args()
    return args

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

if __name__ == '__main__':
    args = parse_args()
    batch_size = 1
    gpu = args.gpu_id
    ctx = mx.gpu(gpu)
    # snapshot_path = args.snapshot
    # video_path = args.video_path

    snapshot_path = "./data/2018-12-08-16-29-08-458567"
    face_model_path = "D:\\CV\\dlib-models-master\\mmod_human_face_detector.dat"


    # ResNet50 structure
    model = hopenet.Hopenet(model_zoo.vision.BottleneckV1, [3, 4, 6, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(face_model_path)

    print('Loading snapshot.')
    # Load snapshot
    model = gluon.nn.SymbolBlock.imports(os.path.join(snapshot_path, "hopenet-4-symbol.json"), ['data'], os.path.join(snapshot_path, "hopenet-4-0000.params"), ctx=ctx)

    print('Loading data.')
    transformations = transforms.Compose([transforms.Resize(240),
            transforms.RandomResizedCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    print('Ready to test network.')

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = np.float32(idx_tensor)

    video = cv2.VideoCapture(0)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    frame_num = 1
    color_green = (0, 255, 0)
    line_width = 1

    while True:

        ret,frame = video.read()
        if ret == False:
            break

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Dlib detect
        dets = cnn_face_detector(cv2_frame, 1)

        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = int(max(x_min, 0)); y_min = int(max(y_min, 0))
            x_max = int(min(frame.shape[1], x_max)); y_max = int(min(frame.shape[0], y_max))
            # Crop image
            img = cv2_frame[y_min:y_max,x_min:x_max]
            img = Image.fromarray(img)

            # Transform
            img = transformations(mx.nd.array(img))
            img = mx.nd.expand_dims(img, axis=0)
            img = img.as_in_context(ctx)

            yaw, pitch, roll = model(img)

            yaw_predicted = softmax(yaw.asnumpy(), axis=1)
            pitch_predicted = softmax(pitch.asnumpy(), axis=1)
            roll_predicted = softmax(roll.asnumpy(), axis=1)

            yaw_predicted = np.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = np.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = np.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            cv2.rectangle(frame, (det.rect.left(), det.rect.top()), (det.rect.right(), det.rect.bottom()), color_green, line_width)
            cv2.putText(frame, "yaw: " + "{:7.2f}".format(yaw_predicted[0]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), thickness=2)
            cv2.putText(frame, "pitch: " + "{:7.2f}".format(pitch_predicted[0]), (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), thickness=2)
            cv2.putText(frame, "roll: " + "{:7.2f}".format(roll_predicted[0]), (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), thickness=2)

        cv2.imshow('my webcam', frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()
