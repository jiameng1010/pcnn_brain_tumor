import argparse
from argparse import Namespace
import tensorflow as tf
import numpy as np
import yaml
import os
import sys
from pyhocon import ConfigFactory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
from braintumor_data import ProbeProvider
from probe_network_no_grid_sigma_edma import ProbeNetwork
from Disp_sampler import octree_generator
import datetime
from medpy.io import load, save
#from octree_log import octree_generator

scale = np.asarray([160, 200, 179])
Affine_uniform2mindboggle = np.asarray([[160.0, 0.0, 0.0, 48.0],
                                        [0.0, 200.0, 0.0, 28.0],
                                        [0.0, 0.0, 179.0, 1.0],
                                        [0.0, 0.0, 0.0, 1.0]])
def affine(affine_m, points):
    point_aug = np.concatenate([points, np.ones(shape=(points.shape[0], 1), dtype=points.dtype)], axis=1)
    point_transformed = affine_m.dot(point_aug.T).T[:, :3]
    return point_transformed

#import pointnet_part_seg as model
with open(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/confs/training.yaml', 'r') as f:
    cfg = yaml.load(f)

if __name__ == '__main__':
    pv = ProbeProvider(cfg)

if __name__ == '__main__':
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
else:
    from argparse import Namespace
    FLAGS = Namespace(gpu=0, batch=None, epoch=200, point_num=150,
                      output_dir='train_results', wd=0,
                      config='pointconv_probe', expname='probe', hypothesis='')


CONF_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/confs'
conf = ConfigFactory.parse_file(CONF_DIR+'/{0}.conf'.format(FLAGS.config))

os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(FLAGS.gpu)

#hdf5_data_dir = os.path.join(BASE_DIR, 'segmentation_data/hdf5_data')

# MAIN SCRIPT
point_num = cfg['training']['num_points1'] + cfg['training']['num_points2']
probe_num = cfg['training']['num_sample_points']
if FLAGS.batch == None:
    batch_size = int(cfg['training']['batch_size'] / 2)

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
    groundtruth_ph = tf.placeholder(tf.float32, shape=(batch_size, probe_num, 3))
    return pointclouds_ph, probepoints_ph, groundtruth_ph

def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot

class DispFunction():
    def __init__(self, saved_model_dir, input_PC=None):
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(FLAGS.gpu)):
                tf.set_random_seed(1000)
                pointclouds_ph, probepoints_ph, gt_ph = placeholder_inputs()
                pointlabel_ph = tf.placeholder(tf.float32, shape=(batch_size, probe_num))
                extend_ph = tf.placeholder(tf.float32, shape=(batch_size, 3))
                # normal_gt = tf.placeholder(tf.float32, shape=(batch_size, probe_num, 3))
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
                disp_pred, feature = network.get_network_model(pointclouds_ph, probepoints_ph, pointlabel_ph, extend_ph, extend_factor=3.0,\
                                                               is_training=is_training_ph, bn_decay=bn_decay,
                                                               batch_size=batch_size, num_point=point_num,
                                                               weight_decay=FLAGS.wd)

                # model.py defines both classification net and segmentation net, which share the common global feature extractor network.
                # In model.get_loss, we define the total loss to be weighted sum of the classification and segmentation losses.
                # Here, we only train for segmentation network. Thus, we set weight to be 1.0.
                # loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res  \
                #    = network.get_loss(seg_pred, gt_ph)
                #per_point_loss, per_instance_loss, total_loss_total = network.get_loss_displacement(disp_pred, gt_ph)

                # soft = tf.nn.softmax(seg_pred)
                disp_x = tf.slice(disp_pred, [0, 0, 0], [batch_size, point_num, 1])
                disp_y = tf.slice(disp_pred, [0, 0, 1], [batch_size, point_num, 1])
                disp_z = tf.slice(disp_pred, [0, 0, 2], [batch_size, point_num, 1])
                normal_pred_x = -tf.gradients(disp_x, probepoints_ph)[0]
                normal_pred_y = -tf.gradients(disp_y, probepoints_ph)[0]
                normal_pred_z = -tf.gradients(disp_z, probepoints_ph)[0]

                # trainer2 = tf.train.AdamOptimizer(learning_rate)
                # train_normal_op = trainer2.minimize(normal_loss, var_list=train_variables, global_step=batch)
                # surface_op = trainer2.minimize(surface_loss, var_list=train_variables, global_step=batch)

            saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.Session(config=config)

            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess,saved_model_dir)

        self.sess = sess
        self.tensors=[pointclouds_ph, probepoints_ph, is_training_ph, disp_pred, feature, pointlabel_ph, extend_ph]
        if input_PC != None:
            self.pointcloud = np.tile(np.expand_dims(input_PC, axis=0), [batch_size, 1, 1])

    def get_data_provider(self):
        pv = ProbeProvider(cfg)
        return pv

    def update_PC(self, point_cloud, extends):
        self.pointcloud = np.tile(np.expand_dims(point_cloud, axis=0), [batch_size, 1, 1])
        self.extends = np.tile(np.expand_dims(extends, axis=0), [batch_size, 1])

    def get_reconstructied_mesh(self):
        self.mesh_generator = octree_generator(self.pointcloud, self.pointlabel, self.sess, self.tensors)
        return self.mesh_generator.get_mesh()

    def get_displacement_image(self, affine_to_talairach=None, shape=[256, 256, 181]):

        out_displacement_image = np.zeros(shape=shape, dtype=np.float32)
        index = np.where(out_displacement_image == 0)
        voxels = np.concatenate([np.expand_dims(index[0], axis=1),
                                 np.expand_dims(index[1], axis=1),
                                 np.expand_dims(index[2], axis=1)], axis=1)
        querry_points = affine(affine_to_talairach, voxels)
        disp = self.eval(querry_points)
        #disp = np.linalg.inv(affine_to_talairach[:3, :3]).dot(disp.T).T

        shape.append(3)
        out_displacement_image = np.zeros(shape=shape, dtype=np.float32)
        out_displacement_image[index] = disp

        return out_displacement_image

    def eval(self, points, threshold=None, laplacian=False):
        if not isinstance(threshold, np.ndarray):
            threshold = np.zeros(shape=points.shape[0])
        batch_points = (batch_size * probe_num)
        num_of_querry = points.shape[0]
        num_of_runs = int(num_of_querry / batch_points)
        pred_all = np.zeros(shape=[num_of_querry, 3])
        pred_all_value = np.zeros(shape=[num_of_querry, 3])
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
                self.tensors[5]: np.concatenate([1*np.ones(shape=(batch_size, 750), dtype=np.float32), 2*np.ones(shape=(batch_size, 750), dtype=np.float32)], axis=1),
                self.tensors[6]: self.extends
            }

            pred = self.sess.run(self.tensors[3], feed_dict=feed_dict)
            feed_threshold = np.reshape(feed_threshold, newshape=[batch_size, probe_num])
            pred_value = pred
            pred_value = np.reshape(pred_value, [-1, 3])
            if i == num_of_runs:
                pred_all[(i * batch_points):, :] = pred_value[:(num_of_querry - (i * batch_points)), :]
            else:
                pred_all[(i * batch_points):((i + 1) * batch_points), :] = pred_value
        return pred_all

trained_model_dir = "2021_10_18_18_38_55"
if __name__=='__main__1':
    is_training = False
    train_generator = pv.provide_data(is_training, batch_size, point_num, probe_num, shuffle=False)

    epoch_list = []
    for entry in os.scandir('../train_results/'+trained_model_dir+'/trained_models'):
        output_dir = entry.path
        print(output_dir)
        if output_dir.split('/')[-1].split('_')[0] == 'epoch':
            epoch_list.append(int(output_dir.split('/')[-1].split('_')[1]))

    epoch_list.sort(reverse=True)
    for data in train_generator:
        n = 0
        for item in epoch_list:
            n += 1
            if n > 10:
                break
            o_function = DispFunction('../train_results/'+trained_model_dir+'/trained_models/epoch_'+str(item)+'/model.ckpt')
            o_function.update_PC(data[0][0], data[5][0])
            disp = o_function.get_displacement_image()
            print(datetime.datetime.now())
            save(disp, '../train_results/'+trained_model_dir+'/trained_models/epoch_' + str(item) + '_' + str(data[6][0]).zfill(5) + '.mha')
        break

if __name__=='__main__':
    is_training = False
    train_generator = pv.provide_data(is_training, batch_size, point_num, probe_num, shuffle=False, get_affine=True, is_debug=True)

    o_function = DispFunction('../train_results/'+trained_model_dir+'/trained_models/model.ckpt')

    n = 0
    for data in train_generator:
        n += 1
        if n == 1:
            continue
        if n > 10:
            break
        o_function.update_PC(data[0][0], data[5][0])
        print(datetime.datetime.now())
        disp = o_function.get_displacement_image(data[7][0])
        save(disp, '../train_results/'+trained_model_dir+'/trained_models/' + str(data[6][0]).zfill(5)+ '.mha')
        print(datetime.datetime.now())
        print('done')
