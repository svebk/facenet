from __future__ import print_function
import numpy as np
import tensorflow as tf
from mtcnn_utils import create_mtcnn, detect_face, imresample

class MTCNNFaceDetector():

  def __init__(self, conf=None):
    """
    Initialize a MTCNN Face detector.

    :param conf: conf could be used to overwrite default values:
                 'minsize', 'threshold', 'factor' and 'data_dir'
    """
    self.pnet = None
    self.rnet = None
    self.onet = None
    # default values that can be overwritten from conf
    #self.minsize = 20  # minimum size of face
    #self.minsize = 40  # minimum size of face
    self.minsize = 48  # minimum size of face
    #self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    #self.threshold = [0.65, 0.75, 0.8]  # three steps's threshold
    #self.threshold = [0.7, 0.75, 0.85]  # three steps's threshold
    #self.threshold = [0.6, 0.7, 0.85]  # three steps's threshold
    #self.threshold = [0.65, 0.75, 0.85]  # three steps's threshold
    self.threshold = [0.7, 0.75, 0.85]  # three steps's threshold
    #self.factor = 0.709  # scale factor
    #self.factor = 0.75  # scale factor
    #self.factor = 0.8  # scale factor
    #self.factor = 0.85  # scale factor
    self.factor = 0.8  # scale factor
    #self.data_dir = '../../data/'
    self.data_dir = '../data/'
    # process conf
    if conf is not None:
      # could check types here...
      if 'minsize' in conf:
        self.minsize = conf['minsize']
      if 'threshold' in conf:
        self.threshold = conf['threshold']
      if 'factor' in conf:
        self.factor = conf['factor']
      if 'data_dir' in conf:
        self.data_dir = conf['data_dir']
    # initialize networks
    self.init_nets()

  def init_nets(self):
    """
    Initialize the three MT-CNN networks.
    """
    with tf.Graph().as_default():
      # tf config could be specified from conf
      self.sess = tf.Session()
      with self.sess.as_default():
        self.pnet, self.rnet, self.onet = create_mtcnn(self.sess, self.data_dir)

  def detect_faces(self, img):
    """
    Detect faces in image.

    :param img: img in memory as a PIL Image
    :return bounding_boxes: list of faces bounding boxes and detection scores
    """
    arr_img = np.array(img)
    bounding_boxes, _ = detect_face(arr_img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
    return bounding_boxes