import open3d
import argparse

import tensorflow as tf
import numpy as np
from datetime import datetime

import os
import sys
from pyhocon import ConfigFactory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
from provider_for_segmentation import ProbeProvider
from probe_network_segnet import ProbeNetwork
import datetime
import h5py
from mayavi import mlab
from open3d import PointCloud, Vector3dVector, draw_geometries_with_editing, draw_geometries

import json

# import pointnet_part_seg as model

pv = ProbeProvider()
# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=5, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=200, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=1024, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_results',
                    help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--config', type=str, default='pointconv_probe', help='Config file name')
parser.add_argument('--expname', type=str, default='probe', help='Config file name')
parser.add_argument('--hypothesis', type=str, default='', help='Config file name')

FLAGS = parser.parse_args()

conf = ConfigFactory.parse_file('../confs/{0}.conf'.format(FLAGS.config))

os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(FLAGS.gpu)

hdf5_data_dir = os.path.join(BASE_DIR, 'segmentation_data/hdf5_data')

# MAIN SCRIPT
point_num = FLAGS.point_num
probe_num = int(point_num / 2)
batch_size = FLAGS.batch
timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())

'''if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

output_dir = os.path.join(FLAGS.output_dir, timestamp)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)'''

# color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
# color_map = json.load(open(color_map_file, 'r'))

# all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
# fin = open(all_obj_cats_file, 'r')
# lines = [line.rstrip() for line in fin.readlines()]
# all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
# fin.close()

# all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
NUM_CATEGORIES = 2
# NUM_PART_CATS = len(all_cats)

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
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

random_color = np.random.uniform(size=(1024, 3))

'''MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

os.system("""cp {0} "{1}" """.format('seg_network.py', MODEL_STORAGE_PATH))  # bkp of model def
os.system("""cp {0} "{1}" """.format('test.py', MODEL_STORAGE_PATH))  # bkp of model def
os.system("""cp {0} "{1}" """.format('testing_ply_file_list.txt', MODEL_STORAGE_PATH))  # bkp of model def
os.system("""cp {0} "{1}" """.format('../layers/convlayer_elements.py', MODEL_STORAGE_PATH))  # bkp of model def
os.system("""cp {0} "{1}" """.format('../tf_util.py', MODEL_STORAGE_PATH))  # bkp of model def
os.system("""cp {0} "{1}" """.format('../layers/convolution_layer.py', MODEL_STORAGE_PATH))  # bkp of model def
os.system("""cp {0} "{1}" """.format('../layers/deconvolution_layer.py', MODEL_STORAGE_PATH))  # bkp of model def
os.system("""cp {0} "{1}" """.format('../layers/pooling.py', MODEL_STORAGE_PATH))  # bkp of model def
os.system("""cp {0} "{1}" """.format('../confs/pointconv_probe.conf', MODEL_STORAGE_PATH))  # bkp of model def
os.system("""cp {0} "{1}" """.format('../confs/pointconv.conf', MODEL_STORAGE_PATH))  # bkp of model def

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER = os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)'''


def printout(flog, data):
    print(data)
    flog.write(data + '\n')


def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    probepoints_ph = tf.placeholder(tf.float32, shape=(batch_size, 2 * probe_num, 3))
    # input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, NUM_CATEGORIES))
    groundtruth_ph = tf.placeholder(tf.int32, shape=(batch_size, 2 * probe_num))
    return pointclouds_ph, probepoints_ph, groundtruth_ph


def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot

def generate_probe_from_pc(pc):
    probe = np.random.normal(0, 0.01, pc.shape) + pc
    return probe



def gradients(yy, xx):
    y = tf.reshape(yy, [-1, 1])
    def gradients_one(input):
        return tf.gradients(input[0], xx[input[1]])
    #g = tf.map_fn(gradients_one, elems=(y, tf.range(0, len(xx))))
    all = []
    for i in range(0,4):
        all.append(tf.gradients(y[i,0], xx[i]))

    return tf.expand_dims(tf.concat(all, axis=0), axis=1)

def generate():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(FLAGS.gpu)):
            tf.set_random_seed(1000)
            pointclouds_ph, probepoints_ph_org, gt_ph = placeholder_inputs()
            probepoints_ph_shape = tf.reshape(probepoints_ph_org, [-1, 3])
            probepoints_ph_list = tf.unstack(probepoints_ph_shape)
            probepoints_ph = tf.stack(probepoints_ph_list)
            probepoints_ph = tf.reshape(probepoints_ph_list, [batch_size, point_num, 3])
            is_training_ph = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                BASE_LEARNING_RATE,  # base learning rate
                batch * batch_size,  # global_var indicating the number of steps
                DECAY_STEP,  # step size
                DECAY_RATE,  # decay rate
                staircase=True  # Stair-case or continuous decreasing
            )
            learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)

            bn_momentum = tf.train.exponential_decay(
                BN_INIT_DECAY,
                batch * batch_size,
                BN_DECAY_DECAY_STEP,
                BN_DECAY_DECAY_RATE,
                staircase=True)
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

            lr_op = tf.summary.scalar('learning_rate', learning_rate)
            batch_op = tf.summary.scalar('batch_number', batch)
            bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)

            network = ProbeNetwork(conf.get_config('network'))
            seg_pred, feature = network.get_network_model(pointclouds_ph, probepoints_ph, \
                                                 is_training=is_training_ph, bn_decay=bn_decay, batch_size=batch_size,
                                                 num_point=point_num, weight_decay=FLAGS.wd)

            # model.py defines both classification net and segmentation net, which share the common global feature extractor network.
            # In model.get_loss, we define the total loss to be weighted sum of the classification and segmentation losses.
            # Here, we only train for segmentation network. Thus, we set weight to be 1.0.
            loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res \
                = network.get_loss(seg_pred, gt_ph)

            soft = tf.slice(seg_pred, [0, 0, 0], [batch_size, point_num, 1]) - tf.slice(seg_pred, [0, 0, 1], [batch_size, point_num, 1])
            normal = tf.gradients(soft, probepoints_ph)

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            total_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            seg_training_loss_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            seg_training_acc_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_acc_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_acc_avg_cat_ph = tf.placeholder(tf.float32, shape=())

            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            total_test_loss_sum_op = tf.summary.scalar('total_testing_loss', total_testing_loss_ph)

            seg_train_loss_sum_op = tf.summary.scalar('seg_training_loss', seg_training_loss_ph)
            seg_test_loss_sum_op = tf.summary.scalar('seg_testing_loss', seg_testing_loss_ph)

            seg_train_acc_sum_op = tf.summary.scalar('seg_training_acc', seg_training_acc_ph)
            seg_test_acc_sum_op = tf.summary.scalar('seg_testing_acc', seg_testing_acc_ph)
            seg_test_acc_avg_cat_op = tf.summary.scalar('seg_testing_acc_avg_cat', seg_testing_acc_avg_cat_ph)

            train_variables = tf.trainable_variables()

            trainer = tf.train.AdamOptimizer(learning_rate)
            train_op = trainer.minimize(loss, var_list=train_variables, global_step=batch)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, 'train_results/2019_03_12_02_44_45/trained_models/model.ckpt')

        global total_paramers
        total_paramers = 0
        for varaible in tf.trainable_variables():
            shape = varaible.get_shape()
            var_parameter = 1
            for dim in shape:
                var_parameter *= dim.value
            print("variabile : {0} , {1}".format(varaible.name, var_parameter))
            total_paramers += var_parameter

        '''train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

        fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        # write logs to the disk
        flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')'''

        def generate_one_batch():
            is_training = False
            train_generator = pv.provide_seg_data_test(batch_size, point_num, probe_num)
            total_loss = 0.0
            total_seg_loss = 0.0
            total_seg_acc = 0.0
            num_batch = 0

            for data in train_generator:
                num_batch += 1

                feed_dict = {
                        pointclouds_ph: data[0],
                        probepoints_ph: data[0],
                        gt_ph: data[7],
                        is_training_ph: is_training,
                        }

                # print ("before")
                #
                # np.save('pl',cur_data[begidx: endidx, 0:point_num, :])
                # for k,i in enumerate(debug_tensors):
                #     print (i)
                #     res = sess.run(i, feed_dict=feed_dict)
                #     #print (res[0])
                #     print(res[0].shape)
                #     np.save('dec_{0}'.format(k),[res[0]])
                #
                # print ("-----------------------")
                loss_val, seg_loss_val, \
                        per_instance_seg_loss_val, seg_pred_val, pred_seg_res \
                        = sess.run([loss, seg_loss, \
                        per_instance_seg_loss, seg_pred, per_instance_seg_pred_res], \
                        feed_dict=feed_dict)
                plot(data[0], data[7], pred_seg_res)

            return 0

        generate_one_batch()

def plot(pc, gt, pred):
    pcd = PointCloud()
    colors = random_color[pred[0, :]]
    pcd.colors = Vector3dVector(colors)
    pcd.points = Vector3dVector(pc[0, :, :])
    draw_geometries_with_editing([pcd], width=600, height=600, )


if __name__ == '__main__':
    generate()