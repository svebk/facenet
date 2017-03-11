"""
Minimum example of how to run face detection.

Should be called from facenet root directory with one image path as argument.
"""

import os
import cStringIO
import sys
import time
import json
import base64
import argparse
import numpy as np
import happybase
import tensorflow as tf

from PIL import Image
import boto3
s3_res = boto3.resource('s3')
s3_clt = boto3.client('s3')

sys.path.append('../')
import src.align as align
import src.align.detect_face as face_detector
from src.align.detect_face import PNet, ONet, RNet
import src.facenet as facenet

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
pnet = None
rnet = None
onet = None
images_placeholder = None
embeddings = None

data_dir = '../../data/'
skip_processed = False

def get_image_from_s3(s3_url):
  try:
    s3_spl = s3_url.split("/")
    try:
      bucket_pos = s3_spl.index('s3.amazonaws.com')+1
    except:
      print "{} is not a s3 url.".format(s3_url)
      return None
    bucket = s3_spl[bucket_pos]
    key = '/'.join(s3_spl[bucket_pos+1:])
    buffer = cStringIO.StringIO()
    s3_clt.download_fileobj(bucket, key, buffer)
    img = Image.open(buffer)
    # # for debugging
    # s3_img_obj = s3_res.Object(bucket, key)
    # ctype = s3_img_obj.content_type
    # nb_bytes = s3_img_obj.content_length
    # #print bucket, key, ctype, nb_bytes, img.size
    return img
  except Exception as err:
    print "Could not download image from {}. Error was: {}".format(s3_url, err)
    return None

def detect_image_memory(sess, img):
  """

  :param img: img in memory
  :return:
  """
  with sess.as_default():
    bounding_boxes, _ = face_detector.detect_face(np.array(img), minsize, pnet, rnet, onet, threshold, factor)
    #print "Found {} faces.".format(len(bounding_boxes)), bounding_boxes
    return bounding_boxes

def show_face_bbox_memory(img, bboxes):
  from PIL import ImageDraw
  draw = ImageDraw.Draw(img)
  for bbox in bboxes:
    draw.rectangle([(int(np.round(bbox[0])),
                     int(np.round(bbox[1]))),
                    (int(np.round(bbox[2])),
                     int(np.round(bbox[3])))],
                    outline=(0, 255, 0))
  img.show()

def crop_face_with_margin(img, bboxes, target_size=182, margin=44):
  crops = []
  for det in bboxes:
    bb = np.zeros(4, dtype=np.int32)
    w_ar = (det[2] - det[0]) / target_size
    h_ar = (det[3] - det[1]) / target_size
    bb[0] = np.maximum(det[0] - margin * w_ar / 2 , 0)
    bb[1] = np.maximum(det[1] - margin * h_ar / 2, 0)
    bb[2] = np.minimum(det[2] + margin * w_ar / 2, img.size[1])
    bb[3] = np.minimum(det[3] + margin * h_ar / 2, img.size[0])
    print bb
    cropped = img.crop((bb[0],bb[1],bb[2],bb[3]))
    crops.append(cropped.resize((target_size, target_size), resample=Image.BILINEAR))
  return crops

def show_crops(crops):
  for crop in crops:
    crop.show()

def create_mtcnn_sess(sess, model_path):
  with tf.variable_scope('pnet'):
    data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
    pnet = PNet({'data': data})
    pnet.load(os.path.join(model_path, 'det1.npy'), sess)
  with tf.variable_scope('rnet'):
    data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
    rnet = RNet({'data': data})
    rnet.load(os.path.join(model_path, 'det2.npy'), sess)
  with tf.variable_scope('onet'):
    data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
    onet = ONet({'data': data})
    onet.load(os.path.join(model_path, 'det3.npy'), sess)

  pnet_fun = lambda img: sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0': img})
  rnet_fun = lambda img: sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0': img})
  onet_fun = lambda img: sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'),
                                  feed_dict={'onet/input:0': img})
  return pnet_fun, rnet_fun, onet_fun

def init_nets(sess):
  """
  Initialize the three MT-CNN networks.
  """
  with sess.graph.as_default():
    global pnet, rnet, onet
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, data_dir)

def init_facenet(sess, model_dir=os.path.join(data_dir,'20170131-234652')):
  with sess.as_default(), sess.graph.as_default():
    meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))
    facenet.load_model(model_dir, meta_file, ckpt_file)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

def compute_facefeat(sess, faceimg):
  # Get embeddings
  with sess.as_default():
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    phase_train = sess.graph.get_tensor_by_name("phase_train:0")
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    facecrop = facenet.prewhiten(np.expand_dims(facenet.crop(np.asarray(faceimg), False, 160), axis=0))
    emb = sess.run([embeddings], feed_dict={images_placeholder: facecrop, phase_train: False})
    print emb
    return emb

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Detect faces in an image.')
  parser.add_argument('-t', '--hbase_table', default='escorts_images_sha1_infos_dev', help='hbase_table of image to detect faces in')
  parser.add_argument('-c', '--column_url', default='info:s3_url', help='hbase_table column of image url')
  parser.add_argument('-d', '--column_detection', default='info:face_det_mtcnn', help='hbase_table column to store detections')
  parser.add_argument('-r', '--row_start', default=None, help='hbase_table row to start detection from')
  parser.add_argument('-m', '--max_images', default=10, help='maximum number of detections to perform')
  parser.add_argument('-s', '--show', default=False, action='store_true', help='show detections')

  args = parser.parse_args()
  print(args)

  ## initialization
  det_graph = tf.Graph()
  sess_det = tf.Session(graph=det_graph)
  start_init_nets = time.time()
  init_nets(sess_det)
  print "Detection networks initialized in {}s.".format(time.time() - start_init_nets)
  rec_graph = tf.Graph()
  sess_rec = tf.Session(graph=rec_graph)
  start_init_facenet = time.time()
  init_facenet(sess_rec)
  print "Facenet network initialized in {}s.".format(time.time() - start_init_facenet)

  nb_images = 0
  start_time = time.time()
  done = False
  pool = happybase.ConnectionPool(size=2, host='10.1.94.57')
  while not done:
    with pool.connection() as connection:
      #try:
      hbase_tab = connection.table(args.hbase_table)
      b = hbase_tab.batch()
      for one_row in hbase_tab.scan(row_start=args.row_start):
        if args.column_detection in one_row[1] and skip_processed:
          print "Skipping image {} already processed.".format(one_row[0])
        else:
          print nb_images, one_row[0],
          start_time_one_img = time.time()
          s3_url = one_row[1][args.column_url]
          print s3_url,
          img = get_image_from_s3(s3_url)
          if img is not None:
            if len(img.size) == 2:
              rgbimg = Image.new("RGB", img.size)
              rgbimg.paste(img)
              img = rgbimg
            bounding_boxes = detect_image_memory(sess_det, img)
            crops = crop_face_with_margin(img, bounding_boxes)
            print crops
            if args.show:
              show_face_bbox_memory(img, bounding_boxes)
              show_crops(crops)
            out_dict = {}
            dict_bbox = {}
            for i,bbox in enumerate(bounding_boxes):
              face_id = one_row[0]+'_'+'-'.join([str(int(x)) for x in bbox[:4]])
              out_dict['info:facecropb64_'+face_id] = base64.b64encode(crops[i].tobytes())
              start_compute_facefeat = time.time()
              facefeat = compute_facefeat(sess_rec, crops[i])
              print facefeat[0].shape
              print "Face feature computed in {}s.".format(time.time() - start_compute_facefeat)
              out_dict['info:facefeatb64_' + face_id] = base64.b64encode(facefeat[0].data)
              dict_bbox[face_id] = {}
              dict_bbox[face_id]['bbox'] = ','.join([str(x) for x in bbox[:4]])
              dict_bbox[face_id]['score'] = str(bbox[4])
            out_dict[args.column_detection] = json.dumps(dict_bbox)
            #print out_dict
            b.put(one_row[0], out_dict)
            nb_images += 1
            print "Found {} faces in {}s.".format(len(bounding_boxes), time.time() - start_time_one_img)
            b.send()
        if nb_images % 10 == 0:
          b.send()
        if nb_images >= int(args.max_images):
          print "Processed {} images in {}s.".format(nb_images, time.time() - start_time)
          done = True
          break
      b.send()
        # except Exception as err:
        #   print "Main loop failed with error: {}".format(err)
        #   # try to reinitialize connection pool. Could test if err is TTransportException
        #   pool = happybase.ConnectionPool(size=2, host='10.1.94.57')
        #   time.sleep(5)

  # if os.path.isdir(args.image_path): # directory of images
  #   for image_name in os.listdir(args.image_path):
  #     try:
  #       image_path = os.path.join(args.image_path, image_name)
  #       bounding_boxes = detect_image(image_path)
  #       show_face_bbox(image_path, bounding_boxes)
  #     except Exception as err:
  #       print 'Could not process',image_path,'[{}]'.format(err)
  #
  # else: # single image
  #   bounding_boxes = detect_image(args.image_path)
  #   show_face_bbox(args.image_path, bounding_boxes)
