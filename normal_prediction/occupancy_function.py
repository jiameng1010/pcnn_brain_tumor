import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime
import yaml
import os
import sys
from pyhocon import ConfigFactory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
from liver_data import ProbeProvider
from probe_network_no_grid_old import ProbeNetwork
import datetime
#from octree_log import octree_generator

#import pointnet_part_seg as model
with open('../confs/training.yaml', 'r') as f:
    cfg = yaml.load(f)

pv = ProbeProvider(cfg)
# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=None, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=200, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=150, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--config', type=str, default='pointconv_probe', help='Config file name')
parser.add_argument('--expname', type=str, default='probe', help='Config file name')
parser.add_argument('--hypothesis', type=str, default='', help='Config file name')
FLAGS = parser.parse_args()

conf = ConfigFactory.parse_file('../confs/{0}.conf'.format(FLAGS.config))

os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(FLAGS.gpu)

#hdf5_data_dir = os.path.join(BASE_DIR, 'segmentation_data/hdf5_data')

# MAIN SCRIPT
point_num = cfg['training']['num_points1'] + cfg['training']['num_points2']
probe_num = cfg['training']['num_sample_points']
if FLAGS.batch == None:
    batch_size = cfg['training']['batch_size']

#color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
#color_map = json.load(open(color_map_file, 'r'))

#all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
#fin = open(all_obj_cats_file, 'r')
#lines = [line.rstrip() for line in fin.readlines()]
#all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
#fin.close()

#all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
NUM_CATEGORIES = 2
#NUM_PART_CATS = len(all_cats)

print('#### Batch Size: {0}'.format(batch_size))
print('#### Point Number: {0}'.format(point_num))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))

DECAY_STEP = 16881 * 20  ##td
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-5

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.001
MOMENTUM = 0.9
TRAINING_EPOCHES = FLAGS.epoch
#print('### Training epoch: {0}'.format(TRAINING_EPOCHES))


def printout(flog, data):
	print(data)
	flog.write(data + '\n')

def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    probepoints_ph = tf.placeholder(tf.float32, shape=(batch_size, probe_num, 3))
    #input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, NUM_CATEGORIES))
    groundtruth_ph = tf.placeholder(tf.int32, shape=(batch_size, probe_num))
    return pointclouds_ph, probepoints_ph, groundtruth_ph

def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot

class OccFunction():
    def __init__(self, saved_model_dir, input_PC=None):
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(FLAGS.gpu)):
                tf.set_random_seed(1000)
                pointclouds_ph, probepoints_ph, gt_ph = placeholder_inputs()
                normal_gt = tf.placeholder(tf.float32, shape=(batch_size, probe_num, 3))
                is_training_ph = tf.placeholder(tf.bool, shape=())

                batch = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(
                                BASE_LEARNING_RATE,     # base learning rate
                                batch * batch_size,     # global_var indicating the number of steps
                                DECAY_STEP,             # step size
                                DECAY_RATE,             # decay rate
                                staircase=True          # Stair-case or continuous decreasing
                                )
                learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)

                bn_momentum = tf.train.exponential_decay(
                          BN_INIT_DECAY,
                          batch*batch_size,
                          BN_DECAY_DECAY_STEP,
                          BN_DECAY_DECAY_RATE,
                          staircase=True)
                bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

                lr_op = tf.summary.scalar('learning_rate', learning_rate)
                batch_op = tf.summary.scalar('batch_number', batch)
                bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)


                network = ProbeNetwork(conf.get_config('network'))
                seg_pred, feature = network.get_network_model(pointclouds_ph, probepoints_ph, \
                        is_training=is_training_ph, bn_decay=bn_decay, batch_size=batch_size, num_point=point_num, weight_decay=FLAGS.wd)

                # model.py defines both classification net and segmentation net, which share the common global feature extractor network.
                # In model.get_loss, we define the total loss to be weighted sum of the classification and segmentation losses.
                # Here, we only train for segmentation network. Thus, we set weight to be 1.0.
                loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res  \
                    = network.get_loss(seg_pred, gt_ph)

                #soft = tf.nn.softmax(seg_pred)
                soft = tf.slice(seg_pred, [0, 0, 0], [batch_size, point_num, 1]) - tf.slice(seg_pred, [0, 0, 1], [batch_size, point_num, 1])
                surface_loss = tf.reduce_mean(soft)
                normal_pred = -tf.gradients(soft, probepoints_ph)[0]
                laplacian = tf.gradients(normal_pred, probepoints_ph)
                laplacian = tf.squeeze(tf.reduce_sum(laplacian, axis=-1))
                #normal_metric, normal_loss = network.get_normal_loss(normal_pred, normal_gt)


                total_training_loss_ph = tf.placeholder(tf.float32, shape=())
                total_testing_loss_ph = tf.placeholder(tf.float32, shape=())

                seg_training_loss_ph = tf.placeholder(tf.float32, shape=())
                seg_testing_loss_ph = tf.placeholder(tf.float32, shape=())

                seg_training_acc_ph = tf.placeholder(tf.float32, shape=())
                seg_testing_acc_ph = tf.placeholder(tf.float32, shape=())
                seg_testing_acc_avg_cat_ph = tf.placeholder(tf.float32, shape=())

                train_variables = tf.trainable_variables()

                trainer = tf.train.AdamOptimizer(learning_rate)
                train_op = trainer.minimize(loss, var_list=train_variables, global_step=batch)

                #trainer2 = tf.train.AdamOptimizer(learning_rate)
                #train_normal_op = trainer2.minimize(normal_loss, var_list=train_variables, global_step=batch)
                #surface_op = trainer2.minimize(surface_loss, var_list=train_variables, global_step=batch)

            saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.Session(config=config)

            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess,saved_model_dir)

        self.sess = sess
        self.tensors=[pointclouds_ph, probepoints_ph, is_training_ph, seg_pred, normal_pred, feature, laplacian]
        if input_PC != None:
            self.pointcloud = np.tile(np.expand_dims(input_PC, axis=0), [batch_size, 1, 1])

    def update_PC(self, point_cloud):
        self.pointcloud = np.tile(np.expand_dims(point_cloud, axis=0), [batch_size, 1, 1])

    def eval(self, points, threshold=None, normal=False):
        if not isinstance(threshold, np.ndarray):
            threshold = np.zeros(shape=points.shape[0])
        batch_points = (batch_size * probe_num)
        num_of_querry = points.shape[0]
        num_of_runs = int(num_of_querry / batch_points)
        pred_all = np.zeros(shape=[num_of_querry])
        pred_all_value = np.zeros(shape=[num_of_querry])
        pred_all_normal = np.zeros(shape=[num_of_querry, 3])
        pred_all_laplacian = np.zeros(shape=[num_of_querry])
        for i in range(num_of_runs + 1):
            if i == num_of_runs:
                feed_points = np.zeros(shape=(batch_points, 3))
                feed_threshold = np.zeros(shape=(batch_points))
                feed_points[:(num_of_querry - (i * batch_points)), :] = points[(i * batch_points):, :]
                feed_threshold[:(num_of_querry - (i * batch_points))] = threshold[(i * batch_points):]
            else:
                feed_points = points[(i * batch_points):((i + 1) * batch_points), :]
                feed_threshold = threshold[(i * batch_points):((i + 1) * batch_points)]
            feed_points = np.reshape(feed_points, [batch_size, probe_num, 3])
            feed_dict = {
                self.tensors[0]: self.pointcloud,
                self.tensors[1]: feed_points,
                self.tensors[2]: False,
            }

            if not normal:
                pred, normal_pred = self.sess.run([self.tensors[3], self.tensors[4]], feed_dict=feed_dict)
                pred = pred[:, :, 0] - pred[:, :, 1]
                pred = np.squeeze(-pred)
                feed_threshold = np.reshape(feed_threshold, newshape=[batch_size, probe_num])
                pred_value = 1 * ((pred) < feed_threshold)
                pred_value = np.squeeze(pred_value)
                pred_value = np.reshape(pred_value, [-1])
                if i == num_of_runs:
                    pred_all[(i * batch_points):] = pred_value[:(num_of_querry - (i * batch_points))]
                    pred_all_normal[(i * batch_points):, :] = normal_pred[:(num_of_querry - (i * batch_points)), :]
                    pred_all_value[(i * batch_points):] = np.reshape(pred - feed_threshold, [-1])[
                                                          :(num_of_querry - (i * batch_points))]
                else:
                    pred_all[(i * batch_points):((i + 1) * batch_points)] = pred_value
                    pred_all_normal[(i * batch_points):((i + 1) * batch_points), :] = normal_pred
                    pred_all_value[(i * batch_points):((i + 1) * batch_points)] = np.reshape(pred - feed_threshold,
                                                                                             [-1])
            else:
                pred, normal_pred, laplacian_pred = self.sess.run(
                    [self.tensors[3], self.tensors[4], self.tensors[6]], feed_dict=feed_dict)
                pred = pred[:, :, 0] - pred[:, :, 1]
                pred = np.squeeze(-pred)
                feed_threshold = np.reshape(feed_threshold, newshape=[batch_size, probe_num])
                pred_value = 1 * ((pred) < feed_threshold)
                pred_value = np.squeeze(pred_value)
                pred_value = np.reshape(pred_value, [-1])
                normal_pred = np.reshape(normal_pred[0], [-1, 3])
                laplacian_pred = np.reshape(laplacian_pred, [-1])
                if i == num_of_runs:
                    pred_all[(i * batch_points):] = pred_value[:(num_of_querry - (i * batch_points))]
                    pred_all_normal[(i * batch_points):, :] = normal_pred[:(num_of_querry - (i * batch_points)), :]
                    pred_all_laplacian[(i * batch_points):] = laplacian_pred[:(num_of_querry - (i * batch_points))]
                    pred_all_value[(i * batch_points):] = np.reshape(pred - feed_threshold, [-1])[
                                                          :(num_of_querry - (i * batch_points))]
                else:
                    pred_all[(i * batch_points):((i + 1) * batch_points)] = pred_value
                    pred_all_normal[(i * batch_points):((i + 1) * batch_points), :] = normal_pred
                    pred_all_laplacian[(i * batch_points):((i + 1) * batch_points)] = laplacian_pred
                    pred_all_value[(i * batch_points):((i + 1) * batch_points)] = np.reshape(pred - feed_threshold,
                                                                                             [-1])

        if not normal:
            return pred_all, pred_all_value, pred_all_normal
        else:
            return pred_all, pred_all_value, pred_all_normal, pred_all_laplacian


if __name__=='__main__':
    is_training = False
    train_generator = pv.provide_data(is_training, batch_size, point_num, probe_num)

    o_function = OccFunction('train_results/2020_06_12_23_43_21/trained_models/model.ckpt')

    for data in train_generator:
        o_function.update_PC(data[0][0])
        while True:
            r = o_function.eval(data[2][0], normal=True)
            print('done')

