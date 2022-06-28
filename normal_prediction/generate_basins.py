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
from probe_provider import ProbeProvider
from probe_network_no_grid import ProbeNetwork
import datetime
import h5py
from mayavi import mlab
from open3d import PointCloud, Vector3dVector, draw_geometries_with_editing, draw_geometries

import json

fig = mlab.figure(size=(1200, 1000))
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

            soft = tf.nn.softmax(seg_pred)
            soft = tf.slice(soft, [0, 0, 0], [batch_size, point_num, 1])
            soft2 = tf.slice(seg_pred, [0, 0, 0], [batch_size, point_num, 1]) - tf.slice(seg_pred, [0, 0, 1],
                                                                                        [batch_size, point_num, 1])

            normal = tf.gradients(soft2, probepoints_ph)
            #normal = tf.gradients(seg_pred, probepoints_ph)

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
        #saver.restore(sess,'train_results/2019_01_30_15_38_06/trained_models/model.ckpt')#sofa
        #saver.restore(sess, 'train_results/useful_network2/trained_models/model.ckpt')
        saver.restore(sess, 'train_results/network_no_grid_all/trained_models/model.ckpt')

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
            train_generator = pv.provide_modelnet_data(is_training, batch_size, point_num, probe_num)
            #printout(flog, 'test_one_epoch*****************************************')

            for data in train_generator:
                probe_stack = []
                pred_stack = []
                fea_stack = []
                normal_stack = []

                generator = Basin_flow_generator(pointcloud=data[0],
                                                 probe_point=generate_probe_from_pc(data[0]),
                                                 init_step_length = 0.03,
                                                 step_length_decay=0.97,
                                                 merge_threshold = 0.05,
                                                 threshold_increase_rate = 1.0,
                                                 tensors = [pointclouds_ph, probepoints_ph, is_training_ph, seg_pred, normal])

                for n in range(40):
                    surface, normal_pred = generator.update_surface(sess)

                    probe_stack.append(np.expand_dims(surface, axis=0))
                    normal_stack.append(np.expand_dims(normal_pred, axis=0))
                    #if n%10 == 0:
                generator.merge()
                generator.plot_segmentation()
                data_file = h5py.File('show_results/file'+'{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()) + 'surface.hdf5', 'w')
                data_file.create_dataset('point_cloud', data=data[0])
                data_file.create_dataset('surface_points', data=np.concatenate(probe_stack, axis=0))
                data_file.create_dataset('surface_normal', data=np.concatenate(normal_stack, axis=0))
                data_file.close()
                '''probe_stack, normal_stack = generator.generate_batch(sess)
                data_file = h5py.File('show_results/file'+'{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()) + 'surface.hdf5', 'w')
                data_file.create_dataset('point_cloud', data=data[0])
                data_file.create_dataset('surface_points', data=probe_stack)
                data_file.create_dataset('surface_normal', data=normal_stack)
                data_file.close()'''
                #break

            return 0

        generate_one_batch()

class Basin_flow_generator:
    def __init__(self, pointcloud, probe_point,\
                 init_step_length, step_length_decay, \
                 merge_threshold, threshold_increase_rate,\
                 tensors):
        self.pointcloud = pointcloud
        self.tensors = tensors
        self.step_length = init_step_length
        self.step_decay = step_length_decay
        self.probe_point = probe_point
        self.merge_threshold = merge_threshold
        self.threshold_increase_rate = threshold_increase_rate

        self.active = np.zeros_like(probe_point)
        self.updated_times = np.zeros((5, 1024, 1))
        self.surface_last = np.zeros_like(probe_point)
        self.update = np.zeros_like(probe_point)

        self.segment = []
        for i in range(5):
            self.segment.append(Segment(1024))


        self.step = 0


    def update_step_length(self):
        self.step_length = self.step_decay * self.step_length
        self.merge_threshold = self.merge_threshold * self.threshold_increase_rate

    def update_surface(self, sess):
        self.surface_last = self.active

        if self.step == 0:
            feed_dict = {
                self.tensors[0]: self.pointcloud,
                self.tensors[1]: self.probe_point,
                self.tensors[2]: False,
            }
            pred, normal_pred = sess.run([self.tensors[3], self.tensors[4]], feed_dict=feed_dict)
            pred = pred[:, :, 0] - pred[:, :, 1]
            pred_value = 2 * ((pred) > 0) - 1
            pred_value = np.squeeze(pred_value)  # pred_value [5, 512] (-1, 1)
            norm = np.sqrt(np.sum(normal_pred[0] * normal_pred[0], axis=-1))
            normal_pred_norm = -normal_pred[0] / (np.expand_dims(norm, axis=2) + 0.001)
            self.update = -self.step_length * normal_pred_norm
            self.active = self.probe_point + self.update
            self.pred_value_last = pred_value

        else:
            #self.plot_segmentation()
            #self.active = self.active + self.update
            #self.merge()

            feed_dict = {
                self.tensors[0]: self.pointcloud,
                self.tensors[1]: self.active,
                self.tensors[2]: False,
            }
            pred, normal_pred = sess.run([self.tensors[3], self.tensors[4]], feed_dict=feed_dict)
            pred = pred[:, :, 0] - pred[:, :, 1]
            pred_value = 2 * ((pred) > 0) - 1
            pred_value = np.squeeze(pred_value)  # pred_value [5, 512] (-1, 1)
            norm = np.sqrt(np.sum(normal_pred[0] * normal_pred[0], axis=-1))
            normal_pred_norm = -normal_pred[0] / (np.expand_dims(norm, axis=2) + 0.001)
            self.update = -self.step_length * normal_pred_norm
            self.update_points(pred_value)
            self.pred_value_last = pred_value

            self.plot_mornal(0, normal_pred_norm)
            #self.plot_segmentation()

            '''self.updated_times += not_cross_points
            self.updated_times = not_cross_points * self.updated_times
            bad_points = 1*self.updated_times>8
            not_bad_points = 1*self.updated_times<=8
            self.updated_times = not_bad_points * self.updated_times

            self.add_cross_zero_points(cross_points)
            self.surface = not_cross_points * (self.surface + self.update) + cross_points * generate_probe_from_pc(generate_probe_from_pc(self.pointcloud))
            self.surface = not_bad_points * self.surface + bad_points * generate_probe_from_pc(generate_probe_from_pc(self.pointcloud))
            self.pred_value_last = np.squeeze(not_cross_points) * pred_value
            self.pred_value_last = np.squeeze(not_bad_points) * self.pred_value_last
            if self.step > 1:
                self.plot_current_cross_zero(2)'''


        self.step += 1
        self.update_step_length()
        return self.active, normal_pred_norm

    def update_points(self, pred_value):
        self.active = self.active + self.update
        out_points = np.expand_dims((pred_value < 0), axis=2)
        self.updated_times += out_points
        self.updated_times = out_points * self.updated_times




    def merge(self):
        #self.active[0, :512, :] = 1*np.ones_like(self.active[0, :512, :]) + np.random.normal(scale= 0.1, size=(512, 3))
        #self.active[0, 512:, :] = 2*np.ones_like(self.active[0, 512:, :]) + np.random.normal(scale= 0.1, size=(512, 3))
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        distance5 = np.expand_dims(self.active,axis=1) - np.expand_dims(self.active,axis=2)
        distance5 = np.sqrt(np.sum(distance5*distance5, axis=3))
        for i in range(5):
            distance = distance5[i, :, :]
            should_be_merge1 = distance<self.merge_threshold
            should_be_merge2 = distance>0.0
            should_be_merge = np.where((should_be_merge2 & should_be_merge1)==True)
            print(should_be_merge[0].shape[0])
            for j in range(should_be_merge[0].shape[0]):
                pair = [should_be_merge[0][j], should_be_merge[1][j]]
                self.segment[i].merge(pair)
                #for k in self.segment[i].merged_index:
                #    points_should_be_average = np.where(self.segment[i].segment == k)
                #    merge_to = np.average(self.active[i, points_should_be_average, :], axis=1)
                #    self.active[i, points_should_be_average, :] = merge_to






    def add_cross_zero_points(self, cross_points):
        cross_points = np.squeeze(cross_points)
        middle = (self.surface + self.surface_last) / 2
        x, y  = np.nonzero(cross_points)

        num = [0]
        for i in range(5):
            sum = 0
            for j in range(i+1):
                sum  += np.sum(1*(x==j))
            num.append(sum)

        for i in range(5):
            p = middle[i, y[num[i]:num[i+1]], :]
            self.cross_zero_points[i].append(p)

    def generate_batch(self, sess, iterations=60):
        points = []
        normals = []
        for i in range(iterations):
            probe_points = generate_probe_from_pc(self.probe_point)
            feed_dict = {
                self.tensors[0]: self.pointcloud,
                self.tensors[1]: probe_points,
                self.tensors[2]: False,
            }
            pred, normal_pred = sess.run([self.tensors[3], self.tensors[4]], feed_dict=feed_dict)
            points.append(probe_points*(1.0*(pred<0)))
            normals.append(normal_pred*(1.0*(pred<0)))
        return np.concatenate(points, axis=-2), np.concatenate(normals, axis=-2)

    def plot_mornal(self,i, normals):
        #self.plot_test()
        mlab.points3d(self.active[i,:,0], self.active[i,:,1], self.active[i,:,2], scale_factor=0.05)
        mlab.quiver3d(self.active[i,:,0], self.active[i,:,1], self.active[i,:,2],
                      normals[i,:, 0], normals[i,:, 1], normals[i,:, 2], scale_factor=0.08)

        mlab.draw(fig)
        mlab.savefig('./screenshot_basin/' + str(self.step) + '.jpg')
        mlab.clf(fig)

    def plot_test(self):
        mlab.points3d([0.0, 0.0], [0.0, 0.0], [0.0 ,0.05], scale_factor=0.05)

        mlab.draw(fig)
        mlab.savefig('./screenshot_basin/' + str(self.step) + '.jpg')
        mlab.clf(fig)


    def plot_current_cross_zero(self, i):
        point_to_plot = np.concatenate(self.cross_zero_points[i], axis=0)
        mlab.points3d(point_to_plot[:,0], point_to_plot[:,1], point_to_plot[:,2], scale_factor=0.05)
        mlab.draw(fig)
        mlab.savefig('./screenshot_basin/' + str(self.step) + '.jpg')
        mlab.clf(fig)

    def plot_segmentation(self):
        self.pcd = PointCloud()
        colors = random_color[self.segment[0].segment, :]
        self.pcd.colors = Vector3dVector(colors)
        self.pcd.points = Vector3dVector(self.pointcloud[0, :, :])
        draw_geometries_with_editing([self.pcd], width=600, height=600, )



class Segment:
    def __init__(self, num_of_points):
        self.num_of_segments = 0
        self.merged_index = []
        self.segment = np.arange(start=0, step=1, stop=num_of_points)
        self.hierarchy = np.arange(start=0, step=1, stop=num_of_points)

    def merge(self, pair):
        #self.segment[pair[0]] = self.segment[pair[1]]
        #self.segment[pair[1]] = self.segment[pair[0]]
        if self.segment[pair[1]] == self.segment[pair[0]]:
            return
        p0 = np.where(self.segment == self.segment[pair[0]])
        if self.segment[pair[1]] in self.merged_index:
            self.merged_index.remove(self.segment[pair[1]])
        if self.segment[pair[0]] in self.merged_index:
            self.merged_index.remove(self.segment[pair[0]])
        self.merged_index.append(self.segment[pair[1]])
        self.segment[p0] = self.segment[pair[1]]

if __name__ == '__main__':
    generate()