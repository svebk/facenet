from flask import Flask, request, render_template, make_response
from flask_restful import Resource, Api

import sys
import json
import time
import argparse
from datetime import datetime

sys.path.append('../')
from src.detectors.mtcnn import MTCNNFaceDetector
from src.images.image_io import get_image_from_s3, get_image_from_url, draw_face_bbox

app = Flask(__name__)
app.secret_key = "secret_key"
app.config['SESSION_TYPE'] = 'filesystem'

api = Api(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Headers', 'Keep-Alive,User-Agent,If-Modified-Since,Cache-Control,x-requested-with,Content-Type,origin,authorization,accept,client-security-token')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response

global_conf_file = '../conf/detect_api_conf.json'
global_detector = None
global_start_time = None

class APIResponder(Resource):


    def __init__(self):
        self.detector = global_detector
        self.start_time = global_start_time
        self.nb_processed = 0
        self.nb_detected = 0
        # define valid options
        self.valid_options = ["return_images"]
        self.list_endpoints = ["detect_from_URL", "detect_from_B64", "view_detect_from_URL", "view_detect_from_B64"]


    def get(self, mode):
        print("[get] received parameters: {}".format(request.args.keys()))
        query = request.args.get('data')
        options = request.args.get('options')
        print("[get] received data: {}".format(query))
        print("[get] received options: {}".format(options))
        if query:
            return self.process_query(mode, query, options)
        else:
            return self.process_mode(mode)
 

    def put(self, mode):
        return self.put_post(mode)


    def post(self, mode):
        return self.put_post(mode)        


    def put_post(self, mode):
        print("[put/post] received parameters: {}".format(request.form.keys()))
        print("[put/post] received request: {}".format(request))
        query = request.form['data']
        try:
            options = request.form['options']
        except:
            options = None
        print("[put/post] received data of length: {}".format(len(query)))
        print("[put/post] received options: {}".format(options))
        if not query:
            return {'error': 'no data received'}
        else:
            return self.process_query(mode, query, options)


    def process_mode(self, mode):
        if mode == "status":
            return self.status()
        else:
            return {'error': 'unknown_mode: '+str(mode)+'. Did you forget to give \'data\' parameter?'}


    def process_query(self, mode, query, options=None):
        if mode == "detect_from_URL":
            return self.detect_from_URL(query, options)
        elif mode == "detect_from_B64":
            return self.detect_from_B64(query, options)
        # for debugging or showcasing
        elif mode == "view_detect_from_URL":
            return self.view_detect_from_URL(query, options)
        elif mode == "view_detect_from_B64":
            return self.view_detect_from_B64(query, options)
        else:
            return {'error': 'unknown_mode: '+str(mode)}


    def get_options_dict(self, options):
        errors = []
        options_dict = dict()
        if options:
            try:
                options_dict = json.loads(options)
            except Exception as inst:
                err_msg = "[get_options: error] Could not load options from: {}. {}".format(options, inst)
                print(err_msg)
                errors.append(err_msg)
            for k in options_dict:
                if k not in self.valid_options:
                    err_msg = "[get_options: error] Unkown option {}".format(k)
                    print(err_msg)
                    errors.append(err_msg)
        return options_dict, errors


    def detect_from_URL(self, query, options=None):
        query_urls = query.split(',')
        options_dict, errors = self.get_options_dict(options)
        print options_dict
        return_images = False
        if 'return_images' in options_dict and options_dict['return_images']:
            return_images = True
        output = {}
        for image_url in query_urls:
            # download images
            try:
                img = get_image_from_s3(image_url)
                if img is None:
                    print "We will download with request."
                    img = get_image_from_url(image_url)
            except:
                print "Could not download image from",image_url
                output[image_url] = []
            # detect in each image
            bboxes = self.detector.detect_faces(img)
            out_img = {}
            for bbox in bboxes:
                out_img['-'.join([str(x) for x in bbox[:4]])] = bbox[4]
            output[image_url] = {'detections': out_img}
            if return_images:
                output[image_url]['img'] = img
        return output

    def detect_from_B64(self, query, options=None):
        import cStringIO
        import base64
        from PIL import Image
        query_b64s = query.split(',')
        options_dict, errors = self.get_options_dict(options)
        print options_dict
        return_images = False
        if 'return_images' in options_dict and options_dict['return_images']:
            return_images = True
        output = {}
        for i,image_b64 in enumerate(query_b64s):
            # load image
            data_sio = cStringIO(base64.b64decode())
            img = Image.open(data_sio)
            # detect in each image
            bboxes = self.detector.detect_faces(img)
            out_img = {}
            for bbox in bboxes:
                out_img['-'.join([str(x) for x in bbox[:4]])] = bbox[4]
            # save with image id as we don't want to push image b64 back...
            output[str(i)] = {'detections': out_img}
            if return_images:
                output[str(i)]['img'] = img
        return output


    def status(self):
        status_dict = {'status': 'OK'}
        status_dict['API_start_time'] = self.start_time.isoformat(' ')
        status_dict['API_uptime'] = str(datetime.now()-self.start_time)
        status_dict['face_detected'] = self.nb_detected
        status_dict['face_image_processed'] = self.nb_processed
        status_dict['endpoints'] = ','.join(self.list_endpoints)
        return status_dict


    def view_detect_from_X(self, detections):
        import base64
        import cStringIO
        from PIL import Image
        detect_images = []
        print detections
        for one_det in detections:
            print one_det
            img = detections[one_det]['img']
            bboxes_dict = detections[one_det]['detections']
            bboxes = []
            for key in bboxes_dict:
                bboxes.append([float(x) for x in key.split('-')])
            print bboxes
            # we should draw the detections on the image
            draw_face_bbox(img, bboxes)
            # and b64 encode them, buffer saving needed for proper display
            buffer = cStringIO.StringIO()
            img.save(buffer, format=img.format)
            detect_images.append((Image.MIME[img.format], base64.b64encode(buffer.getvalue())))
        #print detect_images
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('view_detections.html', detect_images=detect_images, noblur=True),
                             200,
                             headers)

    # http://localhost:5001/cu_face_detect/view_detect_from_URL?data=https://i.vimeocdn.com/video/443476895_1280x720.jpg
    def view_detect_from_URL(self, query, options=None):
        # TODO: we should add return_images to potential existing options
        detections = self.detect_from_URL(query, "{\"return_images\":\"1\"}")
        return self.view_detect_from_X(detections)


    def view_detect_from_B64(self, query, options=None):
        # TODO: we should add return_images to potential existing options
        detections = self.detect_from_B64(query, "{\"return_images\":\"1\"}")
        return self.view_detect_from_X(detections)


api.add_resource(APIResponder, '/cu_face_detect/<string:mode>')


if __name__ == '__main__':
    start_init = time.time()
    parser = argparse.ArgumentParser(description='Setup face detection API')
    parser.add_argument('-c', '--conf_file', default='../conf/detect_api_conf.json',
                        help='configuration file for the API')
    args = parser.parse_args()
    print(args)

    # read in conf file to get parameters
    global_conf = {}
    global_conf['api_port'] = 5001
    try:
        global_conf = json.load(open(args.conf_file,'rt'))
    except IOError:
        print "Could not read configuration file",args.conf_file

    # load detector once at startup
    global_detector = MTCNNFaceDetector()
    global_start_time = datetime.now()

    from gevent.wsgi import WSGIServer
    http_server = WSGIServer(('', global_conf['api_port']), app)
    print "API started in {}s. Start serving...".format(time.time() - start_init)
    http_server.serve_forever()
