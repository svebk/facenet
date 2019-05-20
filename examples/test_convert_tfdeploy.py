import tensorflow as tf
import tfdeploy as td
import os
import sys
sys.path.append('../src')

from align.detect_face import ONet, RNet, PNet

# setup tfdeploy (only when creating models)
td.setup(tf)

# build your graph
sess = tf.Session()

# build ONet
model_path = '../../data/'
with tf.variable_scope('onet'):
  data_onet = tf.placeholder(tf.float32, (None,48,48,3), 'input')
  onet = ONet({'data':data_onet})
  onet.load(os.path.join(model_path, 'det3.npy'), sess)

with tf.variable_scope('rnet'):
  data_rnet = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
  rnet = RNet({'data': data_rnet})
  rnet.load(os.path.join(model_path, 'det2.npy'), sess)

with tf.variable_scope('pnet'):
  data_pnet = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
  pnet = PNet({'data': data_pnet})
  pnet.load(os.path.join(model_path, 'det1.npy'), sess)

sess.run(tf.global_variables_initializer())

# create a tfdeploy model and save it to disk
#print onet.layers
model = td.Model()
onet_outputs = ['conv6-2', 'conv6-3', 'prob1']
for output in onet_outputs:
  model.add(onet.layers[output], sess)
print len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='onet')),tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='onet')
print len(model.roots),model.roots
model.save("onet.pkl")

#print rnet.layers
model = td.Model()
rnet_outputs = ['conv5-2', 'prob1']
for output in rnet_outputs:
  model.add(rnet.layers[output], sess)
print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnet')
print model.roots
model.save("rnet.pkl")

#print pnet.layers
model = td.Model()
pnet_outputs = ['conv4-2', 'prob1']
for output in pnet_outputs:
  model.add(pnet.layers[output], sess)
print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pnet')
print model.roots
model.save("pnet.pkl")