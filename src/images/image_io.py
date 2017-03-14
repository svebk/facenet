from PIL import Image
import numpy as np
import cStringIO
import boto3

s3_res = boto3.resource('s3')
s3_clt = boto3.client('s3')


# data I/O
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

# Image drawing/showing
def draw_face_bbox(img, bboxes):
  from PIL import ImageDraw
  draw = ImageDraw.Draw(img)
  for bbox in bboxes:
    draw.rectangle([(int(np.round(bbox[0])),
                     int(np.round(bbox[1]))),
                    (int(np.round(bbox[2])),
                     int(np.round(bbox[3])))],
                   outline=(0, 255, 0))


def show_face_bbox_memory(img, bboxes):
  draw_face_bbox(img, bboxes)
  img.show()

def show_crops(crops):
  for crop in crops:
    crop.show()


# Image cropping for recognition
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