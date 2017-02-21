"""
Minimum example of how to run face detection.

Should be called from facenet root directory with one image path as argument.
"""

import sys
import os
sys.path.append('../')
import facenet.src.align as align
import facenet.src.align.detect_face as face_detector
from scipy import misc
import argparse
import tensorflow as tf
import numpy as np

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
pnet = None
rnet = None
onet = None

def init_nets():
  """
  Initialize the three MT-CNN networks.
  """
  global pnet, rnet, onet
  with tf.Graph().as_default():
    # use CPU by default
    cpu_config = tf.ConfigProto(
      device_count={'GPU': 0},
      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0),
      log_device_placement=False
    )
    sess = tf.Session(config=cpu_config)
    with sess.as_default():
      pnet, rnet, onet = align.detect_face.create_mtcnn(sess, './data/')

def detect_image(image_path):
  """

  :param image_path:
  :return:
  """
  img = misc.imread(image_path)
  bounding_boxes, _ = face_detector.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
  print bounding_boxes
  return bounding_boxes

def show_face_bbox(image_path, bboxes):
  import cv2
  img = cv2.imread(image_path)
  for bbox in bboxes:
    cv2.rectangle(img,
                  (int(np.round(bbox[0])),
                   int(np.round(bbox[1]))),
                  (int(np.round(bbox[2])),
                   int(np.round(bbox[3]))),
                  (0, 255, 0), 2)
  cv2.imshow('Face', img)
  cv2.waitKey(0)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Detect faces in an image.')
  parser.add_argument('image_path', help='path of image to detect faces in')

  args = parser.parse_args()
  print(args)
  init_nets()

  if os.path.isdir(args.image_path): # directory of images
    for image_name in os.listdir(args.image_path):
      try:
        image_path = os.path.join(args.image_path, image_name)
        bounding_boxes = detect_image(image_path)
        show_face_bbox(image_path, bounding_boxes)
      except Exception as err:
        print 'Could not process',image_path,'[{}]'.format(err)

  else: # single image
    bounding_boxes = detect_image(args.image_path)
    show_face_bbox(args.image_path, bounding_boxes)
