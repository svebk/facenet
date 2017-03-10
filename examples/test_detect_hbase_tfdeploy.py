"""
Minimum example of how to run face detection.

Should be called from facenet root directory with one image path as argument.
"""

import os
import cStringIO
import sys
import time
import json
import argparse
import numpy as np
import happybase
import tfdeploy as td

sys.path.append('../')
import src.align.detect_face as face_detector

from PIL import Image
import boto3
s3_res = boto3.resource('s3')
s3_clt = boto3.client('s3')

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
pnet = None
rnet = None
onet = None
skip_exist = False

def get_image_from_s3(s3_url):
  try:
    s3_spl = s3_url.split("/")
    bucket_pos = s3_spl.index('s3.amazonaws.com')+1
    bucket = s3_spl[bucket_pos]
    key = '/'.join(s3_spl[bucket_pos+1:])
    buffer = cStringIO.StringIO()
    s3_clt.download_fileobj(bucket, key, buffer)
    img = Image.open(buffer)
    #s3_img_obj = s3_res.Object(bucket, key)
    #ctype = s3_img_obj.content_type
    #nb_bytes = s3_img_obj.content_length
    #print bucket, key, ctype, nb_bytes, img.size
    return img
  except Exception as err:
    print "Could not download image from {}. Error was: {}".format(s3_url, err)
    return None

def init_nets():
  """
  Initialize the three MT-CNN networks.
  """
  global pnet, rnet, onet
  pnet_model = td.Model('pnet.pkl')
  rnet_model = td.Model('rnet.pkl')
  onet_model = td.Model('onet.pkl')
  print pnet_model.roots
  print rnet_model.roots
  print onet_model.roots

  # define all nets i/o
  pnet_input, pnet_output1, pnet_output2 = pnet_model.get('pnet/input:0', 'pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0')
  print pnet_input, pnet_output1, pnet_output2
  rnet_input, rnet_output1, rnet_output2 = rnet_model.get('rnet/input:0', 'rnet/conv5-2/conv5-2:0', 'rnet/prob1:0')
  print rnet_input, rnet_output1, rnet_output2
  onet_input, onet_output1, onet_output2, onet_output3 = onet_model.get('onet/input:0', 'onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0')
  print onet_input, onet_output1, onet_output2, onet_output3

  # define all nets processing function
  pnet = lambda img: (pnet_output1.eval({pnet_input: img}), pnet_output2.eval({pnet_input: img}))
  rnet = lambda img: (rnet_output1.eval({rnet_input: img}), rnet_output2.eval({rnet_input: img}))
  onet = lambda img: (onet_output1.eval({onet_input: img}), onet_output2.eval({onet_input: img}), onet_output3.eval({onet_input: img}))


def detect_image_memory(img):
  """

  :param img: img in memory
  :return:
  """
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
  pool = happybase.ConnectionPool(size=2, host='10.1.94.57')
  # to write detections
  start_init_nets = time.time()
  init_nets()
  print "Detection networks initialized in {}s.".format(time.time() - start_init_nets)
  nb_images = 0
  start_time = time.time()

  done = False
  while not done:
    with pool.connection() as connection:
      try:
        hbase_tab = connection.table(args.hbase_table)
        b = hbase_tab.batch()
        for one_row in hbase_tab.scan(row_start=args.row_start):
          if args.column_detection in one_row[1] and skip_exist:
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
              bounding_boxes = detect_image_memory(img)
              if args.show:
                show_face_bbox_memory(img, bounding_boxes)
              dict_bbox = {}
              for bbox in bounding_boxes:
                face_id = one_row[0]+'_'+'-'.join([str(int(x)) for x in bbox[:4]])
                dict_bbox[face_id] = {}
                dict_bbox[face_id]['bbox'] = ','.join([str(x) for x in bbox[:4]])
                dict_bbox[face_id]['score'] = str(bbox[4])
              #b.put(one_row[0], {args.column_detection: json.dumps(dict_bbox)})
              nb_images += 1
              print "Found {} faces in {}s.".format(len(bounding_boxes), time.time() - start_time_one_img)
          #if nb_images % 10 == 0:
          #  b.send()
          if nb_images >= int(args.max_images):
            print "Processed {} images in {}s.".format(nb_images, time.time() - start_time)
            done = True
            break
        #b.send()
      except Exception as err:
        print "Main loop failed with error: {}".format(err)
        # try to reinitialize connection pool. Could test if err is TTransportException
        pool = happybase.ConnectionPool(size=2, host='10.1.94.57')
        time.sleep(5)

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
