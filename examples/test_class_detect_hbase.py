"""
Minimum example of how to run face detection.

Should be called from facenet root directory with one image path as argument.
"""


import time
import json
import base64
import argparse
import happybase
from PIL import Image

skip_processed = False

from src.detectors.mtcnn import MTCNNFaceDetector
from src.images.image_io import get_image_from_s3, crop_face_with_margin, show_face_bbox_memory, show_crops


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
  mtfd = MTCNNFaceDetector()
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
              bounding_boxes = mtfd.detect_faces(img)
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
