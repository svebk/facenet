# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os

#import cv2 # Only for debug visualization...
import numpy as np
import tensorflow as tf
from scipy import misc

try:
    import align.detect_face
    import facenet
except:
    from facenet.src import align
    from facenet.src.align import detect_face

gpu_memory_fraction = 0.3
facenet_model_checkpoint = os.path.dirname(__file__) + "/../model_checkpoints/20170512-110547"
classifier_model = os.path.dirname(__file__) + "/../model_checkpoints/my_classifier_1.pkl"
debug = False


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        faces = self.detect.find_faces(image)

        for i, face in enumerate(faces):
            if debug:
                import cv2
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            face.name = self.identifier.identify(face)

        return faces


class Identifier:
    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, face):
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            return self.class_names[best_class_indices[0]]


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces

    def find_faces_vggface2(self, image):
        faces = []
        vggface2_rs = 256
        vggface2_ts = 224
        fm = (vggface2_rs - vggface2_ts) / 2

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            # bounding box is then extended by a factor 0.3 (except the extension outside image)
            # to include the whole head, which is used as network input
            fwm = (bb[2] - bb[0]) * 0.3
            fhm = (bb[3] - bb[1]) * 0.3
            face.bounding_box[0] = np.maximum(bb[0] - fwm / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - fhm / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + fwm / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + fhm / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            # First resize to 256 then take central crop of 224
            tmp_face_image = misc.imresize(cropped, (vggface2_rs, vggface2_rs), interp='bilinear')
            face.image = tmp_face_image[fm:vggface2_ts+fm, fm:vggface2_ts+fm, :]

            faces.append(face)

        return faces


    def find_faces_arcface(self, image, **kwargs):

        from skimage import transform as trans

        # If we were to do alignment based on landmarks
        do_align = bool(kwargs.get('align', False))
        M = None
        # Face image size
        image_size = []
        # Should we add a default? Which value 112,112?
        #str_image_size = kwargs.get('image_size', '')
        str_image_size = kwargs.get('image_size', '112,112')
        if len(str_image_size) > 0:
            image_size = [int(x) for x in str_image_size.split(',')]
            if len(image_size) == 1:
                image_size = [image_size[0], image_size[0]]
            assert len(image_size) == 2
            assert image_size[0] == 112
            assert image_size[0] == 112 or image_size[1] == 96
        margin = kwargs.get('margin', 44)
        #margin = kwargs.get('margin', 32)

        faces = []
        bounding_boxes, points = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)

        #print("bounding_boxes: ", bounding_boxes.shape, bounding_boxes)
        #print("points: ", points.shape, points)

        for bbi, det in enumerate(bounding_boxes):
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)
            face.bounding_box[0] = np.maximum(det[0] - margin / 2, 0)
            face.bounding_box[1] = np.maximum(det[1] - margin / 2, 0)
            face.bounding_box[2] = np.minimum(det[2] + margin / 2, image.shape[1])
            face.bounding_box[3] = np.minimum(det[3] + margin / 2, image.shape[0])

            if do_align:
                import cv2
                # Beware of stacking inducing weird reshape...
                landmark = points[:,bbi].reshape(2,5).T
                src = np.array([
                    [30.2946, 51.6963],
                    [65.5318, 51.5014],
                    [48.0252, 71.7366],
                    [33.5493, 92.3655],
                    [62.7299, 92.2041]], dtype=np.float32)
                if image_size[1] == 112:
                    src[:, 0] += 8.0
                dst = landmark.astype(np.float32)

                #print("src: ", src.shape, src)
                #print("dst: ", dst.shape, dst)

                tform = trans.SimilarityTransform()
                tform.estimate(dst, src)
                M = tform.params[0:2, :]
                face.image = cv2.warpAffine(image, M, (image_size[1], image_size[0]), borderValue=0.0)
            else:
                cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
                if len(image_size) > 0:
                    cropped = trans.resize(cropped, (image_size[1], image_size[0]))
                face.image = cropped

            faces.append(face)

        return faces
