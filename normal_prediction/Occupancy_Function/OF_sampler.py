#import open3d
import argparse

import tensorflow as tf
import numpy as np
from numpy import *
from datetime import datetime
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eigh, eig
from scipy import ndimage
from sklearn.manifold import spectral_embedding
from sklearn.cluster import spectral_clustering
import pymesh
import os
import sys
from pyhocon import ConfigFactory
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import Voronoi, Delaunay
from threading import Thread
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
from retrieve_dependent_tensors import Tensor_Retrieval
import datetime
import time
import scipy
import skimage
import copy
import h5py
#from mayavi import mlab
#import plot_functions
#from open3d import PointCloud, Vector3dVector, draw_geometries_with_editing, draw_geometries

import json

def PC_to_off(points, filename):
    f = open(filename, 'w')
    f.write('OFF' + '\n')
    f.write(str(points.shape[0]) + ' ' + '0 ' + '0' + '\n')
    for i in range(points.shape[0]):
        f.write(str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2]) + '\n')
    f.close()
    
def resave_mesh(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    f = open(file_name, 'w')
    f.write('OFF\n')
    for i in range(2, len(lines)):
        f.write(lines[i])
    f.close()

#fig = mlab.figure(size=(1200, 1000))
# import pointnet_part_seg as model

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=4, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=200, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=1024, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_results',
                    help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--config', type=str, default='pointconv_probe', help='Config file name')
parser.add_argument('--expname', type=str, default='probe', help='Config file name')
parser.add_argument('--hypothesis', type=str, default='', help='Config file name')
parser.add_argument('--resume', type=bool, default=False, help='if resume experiment')
parser.add_argument('--num_sample_points', type=int, default=2100, help='size of the initial sample points')

FLAGS = parser.parse_args()

CONF_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/confs'
conf = ConfigFactory.parse_file(CONF_DIR+'/{0}.conf'.format(FLAGS.config))

os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(FLAGS.gpu)

hdf5_data_dir = os.path.join(BASE_DIR, 'segmentation_data/hdf5_data')

#import pointnet_part_seg as model
with open(os.path.dirname(os.path.abspath(__file__))+'/eval.yaml', 'r') as f:
    cfg = yaml.load(f)

# MAIN SCRIPT
point_num = cfg['training']['num_points1'] + cfg['training']['num_points2']
probe_num = cfg['training']['num_sample_points']
batch_size = cfg['training']['batch_size']
timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
scale = 0.5


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

SAVED_MODELS_DIR = '/home/mjia/Documents/jointSegmentation/pcnn/normal_prediction/train_results'

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

SEVEN_OFFSET = np.asarray([[0, 0, 1],
                           [0, 1, 0],
                           [1, 0, 0],
                           [1, 0, 1],
                           [0, 1, 1],
                           [1, 1, 0],
                           [1, 1, 1]])

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

def square_distance(a1, a2):
    distance = np.expand_dims(a1, axis=1) - np.expand_dims(a2, axis=0)
    distance = np.sum(distance*distance, axis=-1)
    return distance

def gradients(yy, xx):
    y = tf.reshape(yy, [-1, 1])
    def gradients_one(input):
        return tf.gradients(input[0], xx[input[1]])
    #g = tf.map_fn(gradients_one, elems=(y, tf.range(0, len(xx))))
    all = []
    for i in range(0,4):
        all.append(tf.gradients(y[i,0], xx[i]))

    return tf.expand_dims(tf.concat(all, axis=0), axis=1)


def eliminate_bbox(vertices, minp, maxp):
    nwithinBB = np.sum(1*(vertices<1.2*minp), axis=1) + np.sum(1*(vertices>1.2*maxp), axis=1)
    return vertices[np.where(nwithinBB==0)[0], :]

def size_tetra(v1, v2, v3):
    xa = v1[:, 0]
    ya = v1[:, 1]
    za = v1[:, 2]
    xb = v2[:, 0]
    yb = v2[:, 1]
    zb = v2[:, 2]
    xc = v3[:, 0]
    yc = v3[:, 1]
    zc = v3[:, 2]

    result = xa*(yb*zc - zb*yc) - ya*(xb*zc - zb*xc) + za*(xb*yc - yb*xc)
    return np.abs(result)



def tetra_interpolation(feed_points, v_points, v_value, delaunay_tri):
    result = np.ones(shape=(feed_points.shape[0]))
    simplex = delaunay_tri.find_simplex(feed_points)
    points_inside_trihull = np.where(simplex != -1)[0]
    corresponding_simplices = simplex[points_inside_trihull]
    corresponding_feed_points = feed_points[points_inside_trihull, :]
    #result[points_inside_trihull] = 0

    # form tetrahedron
    vertices_1 = delaunay_tri.simplices[corresponding_simplices, 0]
    vertices_2 = delaunay_tri.simplices[corresponding_simplices, 1]
    vertices_3 = delaunay_tri.simplices[corresponding_simplices, 2]
    vertices_4 = delaunay_tri.simplices[corresponding_simplices, 3]
    '''p2p_distance = square_distance(feed_points, v_points)
    vertices_1 = np.argmin(p2p_distance, axis=1)
    p2p_distance[np.arange(p2p_distance.shape[0]), vertices_1] = np.Inf
    vertices_2 = np.argmin(p2p_distance, axis=1)
    p2p_distance[np.arange(p2p_distance.shape[0]), vertices_2] = np.Inf
    vertices_3 = np.argmin(p2p_distance, axis=1)
    p2p_distance[np.arange(p2p_distance.shape[0]), vertices_3] = np.Inf
    vertices_4 = np.argmin(p2p_distance, axis=1)'''

    v1 = v_points[vertices_1, :] - corresponding_feed_points
    v2 = v_points[vertices_2, :] - corresponding_feed_points
    v3 = v_points[vertices_3, :] - corresponding_feed_points
    v4 = v_points[vertices_4, :] - corresponding_feed_points

    #volume of sub-tetra
    Sp123 = size_tetra(v1, v2, v3)
    Sp234 = size_tetra(v2, v3, v4)
    Sp124 = size_tetra(v1, v2, v4)
    Sp134 = size_tetra(v1, v3, v4)

    interpolated = Sp123 * v_value[vertices_4] + Sp234 * v_value[vertices_1] + Sp124 * v_value[vertices_3] + Sp134 * v_value[vertices_2]
    interpolated = interpolated / (Sp123 + Sp234 + Sp124 + Sp134)
    result[points_inside_trihull] = interpolated

    return result



class octree_generator:
    def __init__(self, pointcloud, pointlabel, sess, tensors, num_sample_points=3000):
        self.pointcloud = pointcloud
        self.pointlabel = pointlabel
        self.duplicate()
        self.tensors = tensors
        self.sess = sess
        self.num_tile = int(num_sample_points/300)

        #self.create_featurelist([self.tensors[3], self.tensors[4]])
        #self.init_feature()
        self.octree = []
        self.should_split = []
        self.occupy_matrix = [np.zeros((64, 64, 64), dtype=np.int8), np.zeros((128, 128, 128), dtype=np.int8), np.zeros((256, 256, 256), dtype=np.int8)]
        self.value_matrix = [np.ones((64, 64, 64), dtype=np.float32), np.ones((128, 128, 128), dtype=np.float32), np.ones((256, 256, 256), dtype=np.float32)]
        #self.off_set_matrix = [np.ones((128, 128, 128), dtype=np.float32), np.ones((256, 256, 256), dtype=np.float32)]
        self.gradient_matrix = [np.ones((64, 64, 64, 3), dtype=np.float32), np.ones((128, 128, 128, 3), dtype=np.float32), np.ones((256, 256, 256, 3), dtype=np.float32)]
        self.laplacian_matrix = [np.ones((64, 64, 64), dtype=np.float32), np.ones((128, 128, 128), dtype=np.float32), np.ones((256, 256, 256), dtype=np.float32)]

        time1 = time.time()
        _, self.self_activation = self.querry(pointcloud[0,:,:], 0)
        self.now_have_level = 0

        time2 = time.time()
        #self.create_voronoi_sample(random_surface_points)
        time3 = time.time()

        self.construct_level1()
        #self.plot()
        #self.plot_occupy(0)
        #self.plot_field(0)
        #self.construct_mesh(0)
        self.construct_level2()
        #self.plot()
        #self.plot_occupy(1)
        #self.plot_field(1)
        #self.construct_mesh(1)
        self.construct_level3()
        #self.plot()
        #self.plot_occupy(2)
        #self.plot_field(2)
        self.construct_mesh(2)
        time4 = time.time()
        #self.save_octree()
        self.contruction_time1 = time3 - time1
        self.contruction_time2 = (time4 - time3) + (time2 - time1)

        #self.plot_slice()

    def get_reconstruction_time(self):
        return self.contruction_time1, self.contruction_time2

    def get_num_Vvertices(self):
        return self.num_Vvertices


    def create_voronoi_sample(self, random_surface_points):

        #point_cloud = random_surface_points
        point_cloud = self.pointcloud[0,:,:]
        point_cloud = np.tile(point_cloud, (self.num_tile, 1))
        point_cloud = point_cloud + np.random.normal(scale=0.05, size=(point_cloud.shape[0], 3))
        #mlab.points3d(point_cloud[:, 0], point_cloud[:, 2], point_cloud[:, 1], scale_factor=0.02)
        #mlab.show()
        #point_cloud = np.concatenate([point_cloud, point_cloud, point_cloud], axis=0) + np.random.normal(scale=0.02, size=(3*point_cloud.shape[0], 3))
        vor1 = Voronoi(point_cloud)
        self.num_Vvertices = vor1.vertices.shape[0]
        #return 0
        voronoi_vertices1 = eliminate_bbox(vor1.vertices, np.min(point_cloud, axis=0), np.max(point_cloud, axis=0))
        #vor2 = Voronoi(point_cloud + np.random.normal(scale=0.02, size=(point_cloud.shape[0], 3)))
        #voronoi_vertices2 = eliminate_bbox(vor2.vertices, np.min(point_cloud, axis=0), np.max(point_cloud, axis=0))
        self.voronoi_sample_points = np.concatenate([point_cloud, voronoi_vertices1], axis=0)
        _, self.voronoi_sample_value = self.querry(self.voronoi_sample_points, 0)
        #self.voronoi_sample_value = -(2*self.voronoi_sample_value - 1)
        self.eigen_crust()
        return 0

        points = vor.points
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.05, color=(0.8, 0, 0))

        points = eliminate_bbox(vor.vertices, np.min(point_cloud, axis=0), np.max(point_cloud, axis=0))
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.05, color=(0, 0.8, 0))

        points = point_cloud_rand
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.05, color=(0, 0, 0.8))
        #points = vor.points
        #mlab.savefig('tmp.png')
        mlab.show()

        points = self.voronoi_sample_points[np.where(self.voronoi_sample_value>0)[0], :]
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.05, color=(0, 0, 0.8))
        #points = vor.points
        #mlab.savefig('tmp.png')
        mlab.show()

    def eigen_crust(self):
        self.vor_top = Voronoi(self.voronoi_sample_points)
        self.tri_top = Delaunay(self.voronoi_sample_points)
        #edge_weight =  np.exp(0.1*((self.voronoi_sample_value[vor_top.ridge_points[:,0]] * self.voronoi_sample_value[vor_top.ridge_points[:,1]])))
        edge_weight = -self.voronoi_sample_value[self.vor_top.ridge_points[:, 0]] * self.voronoi_sample_value[self.vor_top.ridge_points[:,1]]

        '''edge_matrix = np.zeros(shape=(self.voronoi_sample_points.shape[0], self.voronoi_sample_points.shape[0]))
        edge_matrix[vor_top.ridge_points[:,0], vor_top.ridge_points[:,1]] = edge_weight
        edge_matrix[vor_top.ridge_points[:, 1], vor_top.ridge_points[:, 0]] = edge_weight
        Diagnal = np.diag(np.sum(np.abs(edge_matrix), axis=0))
        Laplacian1 = edge_matrix + Diagnal
        eigenvalues, eigenvectors = eig(Laplacian1, Diagnal)'''

        '''edge_matrix = csr_matrix((self.voronoi_sample_points.shape[0], self.voronoi_sample_points.shape[0]))
        edge_matrix[self.vor_top.ridge_points[:,0], self.vor_top.ridge_points[:,1]] = edge_weight
        edge_matrix[self.vor_top.ridge_points[:, 1], self.vor_top.ridge_points[:, 0]] = edge_weight
        Diagnal = scipy.sparse.diags(np.squeeze(np.asarray(np.sum(np.abs(edge_matrix), axis=0))), offsets=0, format="csr")
        Laplacian2 = edge_matrix + Diagnal
        eigenvaluesS, eigenvectorsS = eigs(Laplacian2, k=4, M=Diagnal, which="SM")
        eigenvaluesS = np.real(eigenvaluesS)
        eigenvectorsS = np.real(eigenvectorsS)[:, 1]'''

        #self.visualize_eigenvector(1*(self.voronoi_sample_value>0))
        #self.visualize_eigenvector(eigenvectorsS)
        #plt.hist(eigenvectorsS, bins=200)

        #self.eigen_crust_mesh(eigenvectorsS)

        vertices1, faces1 = self.eigen_crust_mesh(self.voronoi_sample_value)
        vertices1 = vertices1 / 2
        #vertices1 = vertices1 - np.asarray([[128, 128, 128]])
        #vertices1 = 1.1 * vertices1 / (128 * 2)
        #self.voronoi_sample_value = np.abs(Laplacian2).dot(self.voronoi_sample_value)
        #vertices2, faces2 = self.eigen_crust_mesh(self.voronoi_sample_value)
        #self.voronoi_sample_value = np.abs(Laplacian2).dot(self.voronoi_sample_value)
        #vertices3, faces3 = self.eigen_crust_mesh(self.voronoi_sample_value)
        self.mesh_tri = pymesh.form_mesh(vertices1, faces1)

        '''mlab.triangular_mesh([vert[0] for vert in vertices1],
                             [vert[2] for vert in vertices1],
                             [vert[1] for vert in vertices1],
                             faces1,
                             opacity=0.7,
                             color=(0,0,1))

        #mlab.savefig('tmp.png')
        #mlab.show()'''
        #print(' ')

    def eigen_crust_mesh(self, point_feature):
        point_indicator = 1*(point_feature>0)
        non_uniform_edge = point_indicator[self.vor_top.ridge_points[:, 0]] - point_indicator[self.vor_top.ridge_points[:, 1]]
        non_uniform_edge = np.where(non_uniform_edge != 0)[0]
        feature1 = np.abs(point_feature[self.vor_top.ridge_points[non_uniform_edge, 0]])
        feature2 = np.abs(point_feature[self.vor_top.ridge_points[non_uniform_edge, 1]])
        vertices = np.expand_dims(feature2/(feature1+feature2), axis=1) * self.voronoi_sample_points[self.vor_top.ridge_points[non_uniform_edge, 0],:]\
                   + np.expand_dims(feature1/(feature1+feature2), axis=1) * self.voronoi_sample_points[self.vor_top.ridge_points[non_uniform_edge, 1],:]
        #vertices = 0.5 * vertices
        vertices_id_matrix = csr_matrix((self.voronoi_sample_points.shape[0], self.voronoi_sample_points.shape[0]))
        vertices_id_matrix[self.vor_top.ridge_points[non_uniform_edge, 0], self.vor_top.ridge_points[non_uniform_edge, 1]] = np.arange(vertices.shape[0]) + 1
        vertices_id_matrix[self.vor_top.ridge_points[non_uniform_edge, 1], self.vor_top.ridge_points[non_uniform_edge, 0]] = np.arange(vertices.shape[0]) + 1
        vertices_id_matrix = vertices_id_matrix.astype(int)

        simplex_indicator = point_indicator[np.resize(self.tri_top.simplices, [-1])]
        simplex_indicator = np.resize(simplex_indicator, [self.tri_top.simplices.shape[0], 4])
        simplex_case = np.sum(simplex_indicator, axis=1)
        single_triangle_tetra1 = np.where(simplex_case==1)[0]
        single_triangle_tetra2 = np.where(simplex_case==3)[0]
        double_triangle_tetra = np.where(simplex_case==2)[0]

        # tettrahedrons who has the surface interact with it by one triangle, i.e. cut 3 edges
        f1_auxiliary = simplex_indicator[single_triangle_tetra1, :]
        f1_mono = np.where(f1_auxiliary==1)
        f1_non_mono = np.where(f1_auxiliary!=1)
        f1_non_mono_sep = np.resize(f1_non_mono[1], [int(f1_non_mono[1].shape[0]/3), 3])
        v1 = (single_triangle_tetra1, f1_mono[1])
        v21 = (single_triangle_tetra1, f1_non_mono_sep[:,0])
        v22 = (single_triangle_tetra1, f1_non_mono_sep[:,1])
        v23 = (single_triangle_tetra1, f1_non_mono_sep[:,2])
        face1 = np.concatenate([vertices_id_matrix[self.tri_top.simplices[v1], self.tri_top.simplices[v21]],
                                vertices_id_matrix[self.tri_top.simplices[v1], self.tri_top.simplices[v22]],
                                vertices_id_matrix[self.tri_top.simplices[v1], self.tri_top.simplices[v23]]], axis=0)
        face1 = np.asarray(face1.astype(int)).T

        f2_auxiliary = simplex_indicator[single_triangle_tetra2, :]
        f2_mono = np.where(f2_auxiliary==0)
        f2_non_mono = np.where(f2_auxiliary!=0)
        f2_non_mono_sep = np.resize(f2_non_mono[1], [int(f2_non_mono[1].shape[0]/3), 3])
        v1 = (single_triangle_tetra2, f2_mono[1])
        v21 = (single_triangle_tetra2, f2_non_mono_sep[:,0])
        v22 = (single_triangle_tetra2, f2_non_mono_sep[:,1])
        v23 = (single_triangle_tetra2, f2_non_mono_sep[:,2])
        face2 = np.concatenate([vertices_id_matrix[self.tri_top.simplices[v1], self.tri_top.simplices[v21]],
                                vertices_id_matrix[self.tri_top.simplices[v1], self.tri_top.simplices[v22]],
                                vertices_id_matrix[self.tri_top.simplices[v1], self.tri_top.simplices[v23]]], axis=0)
        face2 = np.asarray(face2.astype(int)).T

        # tettrahedrons who has the surface interact with it by two triangle, i.e. cut 4 edges
        f3_auxiliary = simplex_indicator[double_triangle_tetra, :]
        f3_b1 = np.where(f3_auxiliary == 0)
        f3_b1_sep = np.resize(f3_b1[1], [int(f3_b1[1].shape[0] / 2), 2])
        f3_b2 = np.where(f3_auxiliary != 0)
        f3_b2_sep = np.resize(f3_b2[1], [int(f3_b2[1].shape[0] / 2), 2])
        v11 = (double_triangle_tetra, f3_b1_sep[:, 0])
        v12 = (double_triangle_tetra, f3_b1_sep[:, 1])
        v21 = (double_triangle_tetra, f3_b2_sep[:, 0])
        v22 = (double_triangle_tetra, f3_b2_sep[:, 1])
        face31 = np.concatenate([vertices_id_matrix[self.tri_top.simplices[v11], self.tri_top.simplices[v21]],
                                 vertices_id_matrix[self.tri_top.simplices[v11], self.tri_top.simplices[v22]],
                                 vertices_id_matrix[self.tri_top.simplices[v12], self.tri_top.simplices[v21]]], axis=0)
        face31 = np.asarray(face31.astype(int)).T
        face32 = np.concatenate([vertices_id_matrix[self.tri_top.simplices[v12], self.tri_top.simplices[v21]],
                                 vertices_id_matrix[self.tri_top.simplices[v12], self.tri_top.simplices[v22]],
                                 vertices_id_matrix[self.tri_top.simplices[v11], self.tri_top.simplices[v22]]], axis=0)
        face32 = np.asarray(face32.astype(int)).T
        #check if need to split long edges
        edge1 = vertices[vertices_id_matrix[self.tri_top.simplices[v11], self.tri_top.simplices[v22]]-1, :] - vertices[vertices_id_matrix[self.tri_top.simplices[v12], self.tri_top.simplices[v21]]-1, :]
        edge1 = np.squeeze(edge1)
        edge1 = np.sum(edge1 * edge1, axis=1)
        edge2 = vertices[vertices_id_matrix[self.tri_top.simplices[v11], self.tri_top.simplices[v21]]-1, :] - vertices[vertices_id_matrix[self.tri_top.simplices[v12], self.tri_top.simplices[v22]]-1, :]
        edge2 = np.squeeze(edge2)
        edge2 = np.sum(edge2 * edge2, axis=1)
        should_change = np.where(edge1 > edge2)[0]
        face31[should_change, 2] = vertices_id_matrix[self.tri_top.simplices[v12], self.tri_top.simplices[v22]][0, should_change]
        face32[should_change, 2] = vertices_id_matrix[self.tri_top.simplices[v11], self.tri_top.simplices[v21]][0, should_change]

        faces = np.concatenate([face1, face2, face31, face32], axis=0) - 1
        invalid = np.where(faces == -1)[0]
        invalid = np.unique(invalid)
        faces = np.delete(faces, invalid, axis=0)
        return vertices, faces



    def eigen_crust_sparse(self):
        vor_top = Voronoi(self.voronoi_sample_points)
        # edge_weight =  np.exp(0.1*((self.voronoi_sample_value[vor_top.ridge_points[:,0]] * self.voronoi_sample_value[vor_top.ridge_points[:,1]])))
        edge_weight = -self.voronoi_sample_value[vor_top.ridge_points[:, 0]] * self.voronoi_sample_value[vor_top.ridge_points[:, 1]]

        edge_matrix = np.zeros(shape=(self.voronoi_sample_points.shape[0], self.voronoi_sample_points.shape[0]))

    def visualize_eigenvector(self, eigenvector):
        points = self.voronoi_sample_points
        normal_eigenvector1 = (eigenvector - np.min(eigenvector)) / (np.max(eigenvector)- np.min(eigenvector))
        normal_eigenvector2 = -(eigenvector - np.max(eigenvector)) / (np.max(eigenvector) - np.min(eigenvector))
        color_map = np.concatenate([np.expand_dims(normal_eigenvector1, axis=1),
                                np.zeros(shape=(points.shape[0], 1)),
                                np.expand_dims(normal_eigenvector2, axis=1)], axis=1)
        plot = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.05)
        plot.glyph.scale_mode = 'scale_by_vector'
        plot.mlab_source.dataset.point_data.scalars = color_map

        # points = vor.points
        # mlab.savefig('tmp.png')
        mlab.show()



    def construct_mesh(self, i):
        with h5py.File('value_matrix.h5', 'w') as f:
            f.create_dataset('value_matrix', data=self.value_matrix[2])

        try:
            verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(self.value_matrix[2], level=0.0)
        except:
            verts, faces, normals, values = [0,0,0,0]
            self.mesh = 0
            return
        #verts, faces = skimage.measure.marching_cubes_classic(self.occupy_matrix[i])

        #rescale
        verts = verts - np.asarray([[128, 128, 128]])
        verts = 1.1*verts / (128*2) - np.asarray([[0.0, 0.0, 0.2]])
        #verts = verts / (128*2)

        '''mlab.figure(bgcolor=(1, 1, 1))
        mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[2] for vert in verts],
                             [vert[1] for vert in verts],
                             faces,
                             opacity=0.7,
                             color=(0,1,0))

        mlab.savefig('tmp.png')
        mlab.show()'''

        '''plane_vertices = np.asarray([[1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0]])
        plane_faces = np.asarray([[0, 1, 2], [1, 2, 3]])
        m2 = pymesh.form_mesh(plane_vertices, plane_faces)
        mesa = mlab.triangular_mesh(m2.vertices[:, 0],
                                    m2.vertices[:, 2],
                                    m2.vertices[:, 1],
                                    m2.faces,
                                    # representation='points',
                                    opacity=0.4,)'''


        self.mesh = pymesh.form_mesh(verts, faces)
        #pymesh.save_mesh('upsampled_surface.obj', self.mesh)
        #pymesh.save_mesh('upsampled_surface.off', self.mesh)

    def save_mesh(self, save_name):
        pymesh.save_mesh(save_name, self.mesh)
        #resave_mesh(save_name)
        print('mesh saved to :' + save_name)

    def get_mesh(self):
        return self.mesh

    def save_mesh_tri(self, save_name):
        pymesh.save_mesh(save_name, self.mesh_tri)
        #resave_mesh(save_name)
        print('mesh saved to :' + save_name)

    def construct_level1(self):
        grid_width = 1/32.0
        candidate = 1.1*np.mgrid[-32:32,-32:32, -32:32]/32.0
        candidate = np.transpose(candidate, (1, 2, 3, 0))
        candidate = np.reshape(candidate, [-1, 3])
        #threshold = self.get_shreshold(candidate)
        threshold = 0
        candidate = candidate + np.asarray([[0.0, 0.0, 0.2]])
        occupy, value = self.querry(candidate, threshold)
        #occupy, value = self.querry_voronoi(candidate, threshold)
        self.octree.append(candidate[np.where(occupy==1)])
        self.occupy_matrix[0] = np.reshape(occupy, [64, 64, 64])
        self.value_matrix[0] = np.reshape(value, [64, 64, 64])
        #self.laplacian_matrix[0] = np.reshape(laplacian, [64, 64, 64])
        #self.should_split.append(self.get_split_points_for_level1p(candidate, occupy, grid_weight))
        self.now_have_level += 1
        print('level 1 has been constructed!')
        #self.plot_slice()

    def construct_level2(self):
        subtruct = np.asarray([[64, 64, 64]])
        grid_width = 1/64.0
        current_level = 1
        pervious_level = 0

        candidate = []
        base_occupy = self.occupy_matrix[pervious_level][:-1, :-1, :-1]
        index_of_occupy = np.where(self.occupy_matrix[pervious_level]==1)
        index_of_occupy2 = (2*index_of_occupy[0], 2*index_of_occupy[1], 2*index_of_occupy[2])
        index_of_all = np.where(base_occupy >= -1)
        index_of_all2 = (2 * index_of_all[0], 2 * index_of_all[1], 2 * index_of_all[2])
        self.occupy_matrix[current_level][index_of_occupy2] = 1
        self.value_matrix[current_level][index_of_all2] = self.value_matrix[pervious_level][index_of_all]

        for i in range(7):
            index_of_occupy = np.where(base_occupy == 1)
            index_of_occupy2 = (2 * index_of_occupy[0]+SEVEN_OFFSET[i][0], 2 * index_of_occupy[1]+SEVEN_OFFSET[i][1], 2 * index_of_occupy[2]+SEVEN_OFFSET[i][2])
            index_of_all = np.where(base_occupy >= -1)
            index_of_all2 = (2 * index_of_all[0] + SEVEN_OFFSET[i][0], 2 * index_of_all[1] + SEVEN_OFFSET[i][1], 2 * index_of_all[2] + SEVEN_OFFSET[i][2])
            self.occupy_matrix[current_level][index_of_occupy2] = 1
            self.value_matrix[current_level][index_of_all2] = self.value_matrix[pervious_level][index_of_all]

            a = SEVEN_OFFSET[i][0]
            b = SEVEN_OFFSET[i][1]
            c = SEVEN_OFFSET[i][2]
            shifted_occupy = self.occupy_matrix[pervious_level][a:a+63, b:b+63, c:c+63]
            candid = base_occupy != shifted_occupy
            if i == 0:
                candidate = candid
            else:
                candidate = candidate | candid
            #candid = np.where(candid == True)
            #candid = np.asarray([2 * candid[0]+SEVEN_OFFSET[i][0], 2 * candid[1]+SEVEN_OFFSET[i][1], 2 * candid[2]+SEVEN_OFFSET[i][2]])
            #candidate.append(candid)

        candidates = []
        for i in range(7):
            candid = np.where(candidate == True)
            candid = np.asarray([2 * candid[0] + SEVEN_OFFSET[i][0], 2 * candid[1] + SEVEN_OFFSET[i][1], 2 * candid[2] + SEVEN_OFFSET[i][2]])
            candidates.append(candid)

        candidate = np.transpose(np.concatenate(candidates, axis=-1))
        real_candidate = 1.1*(candidate - subtruct) / subtruct[0,0]
        #threshold = self.get_shreshold(real_candidate)
        threshold = 0
        real_candidate = real_candidate + np.asarray([[0.0, 0.0, 0.2]])
        occupy, value = self.querry(real_candidate, threshold)
        #occupy, value = self.querry_voronoi(real_candidate, threshold)
        self.octree.append(real_candidate[np.where(occupy==1)])
        index_of_occupy = candidate[np.where(occupy==1)]
        index_of_occupy = tuple(np.transpose(index_of_occupy))
        index_of_not_occupy = candidate[np.where(occupy==0)]
        index_of_not_occupy = tuple(np.transpose(index_of_not_occupy))
        self.occupy_matrix[current_level][index_of_occupy] = 1
        self.occupy_matrix[current_level][index_of_not_occupy] = 0
        self.value_matrix[current_level][tuple(np.transpose(candidate))] = value
        #self.should_split.append(self.get_split_points_for_level1p(candidate, occupy, grid_weight))
        self.now_have_level += 1
        print('level 2 has been constructed!')

    def construct_level3(self):
        subtruct = np.asarray([[128, 128, 128]])
        grid_width = 1/128.0
        current_level = 2
        pervious_level = 1

        candidate = []
        base_occupy = self.occupy_matrix[pervious_level][:-1, :-1, :-1]
        index_of_occupy = np.where(self.occupy_matrix[pervious_level]==1)
        index_of_occupy2 = (2*index_of_occupy[0], 2*index_of_occupy[1], 2*index_of_occupy[2])
        index_of_all = np.where(base_occupy >= -1)
        index_of_all2 = (2 * index_of_all[0], 2 * index_of_all[1], 2 * index_of_all[2])
        self.occupy_matrix[current_level][index_of_occupy2] = 1
        self.value_matrix[current_level][index_of_all2] = self.value_matrix[pervious_level][index_of_all]
        for i in range(7):
            index_of_occupy = np.where(base_occupy == 1)
            index_of_occupy2 = (2 * index_of_occupy[0]+SEVEN_OFFSET[i][0], 2 * index_of_occupy[1]+SEVEN_OFFSET[i][1], 2 * index_of_occupy[2]+SEVEN_OFFSET[i][2])
            index_of_all = np.where(base_occupy >= -1)
            index_of_all2 = (2 * index_of_all[0] + SEVEN_OFFSET[i][0], 2 * index_of_all[1] + SEVEN_OFFSET[i][1], 2 * index_of_all[2] + SEVEN_OFFSET[i][2])
            self.occupy_matrix[current_level][index_of_occupy2] = 1
            self.value_matrix[current_level][index_of_all2] = self.value_matrix[pervious_level][index_of_all]

            a = SEVEN_OFFSET[i][0]
            b = SEVEN_OFFSET[i][1]
            c = SEVEN_OFFSET[i][2]
            shifted_occupy = self.occupy_matrix[pervious_level][a:a+127, b:b+127, c:c+127]
            candid = base_occupy != shifted_occupy
            if i == 0:
                candidate = candid
            else:
                candidate = candidate | candid
            #candid = np.where(candid == True)
            #candid = np.asarray([2 * candid[0]+SEVEN_OFFSET[i][0], 2 * candid[1]+SEVEN_OFFSET[i][1], 2 * candid[2]+SEVEN_OFFSET[i][2]])
            #candidate.append(candid)

        candidates = []
        for i in range(7):
            candid = np.where(candidate == True)
            candid = np.asarray([2 * candid[0] + SEVEN_OFFSET[i][0], 2 * candid[1] + SEVEN_OFFSET[i][1], 2 * candid[2] + SEVEN_OFFSET[i][2]])
            candidates.append(candid)

        candidate = np.transpose(np.concatenate(candidates, axis=-1))
        real_candidate = 1.1*(candidate - subtruct) / subtruct[0,0]
        #threshold = self.get_shreshold(real_candidate)
        threshold = 0
        real_candidate = real_candidate + np.asarray([[0.0, 0.0, 0.2]])
        occupy, value = self.querry(real_candidate, threshold)
        #occupy, value = self.querry_voronoi(real_candidate, threshold)
        self.octree.append(real_candidate[np.where(occupy==1)])
        index_of_occupy = candidate[np.where(occupy==1)]
        index_of_occupy = tuple(np.transpose(index_of_occupy))
        index_of_not_occupy = candidate[np.where(occupy==0)]
        index_of_not_occupy = tuple(np.transpose(index_of_not_occupy))
        self.occupy_matrix[current_level][index_of_occupy] = 1
        self.occupy_matrix[current_level][index_of_not_occupy] = 0
        self.value_matrix[current_level][tuple(np.transpose(candidate))] = value
        #self.should_split.append(self.get_split_points_for_level1p(candidate, occupy, grid_weight))
        self.now_have_level += 1
        print('level 3 has been constructed!')

    def get_matrix(self):
        return self.occupy_matrix[2], self.value_matrix[2]

    def get_shreshold(self, candidate):
        distancel2 = np.expand_dims(candidate, axis=1) - self.pointcloud[0, :, :]
        distancel2 = np.sum(distancel2 * distancel2, axis=-1)
        weight = np.exp(-4.0*distancel2)
        weight = weight / np.expand_dims(np.sum(weight, axis=1), axis=1)
        threshold = np.matmul(weight, self.self_activation)
        return threshold

    def plot_slice(self):
        candidate = 0.2*np.mgrid[-128:128, -128:128, 0:1] / 128.0
        candidate = np.transpose(candidate, (1, 2, 3, 0))
        candidate = np.reshape(candidate, [-1, 3])
        occupy, value, gradient, laplacian, pred, softmax = self.querry(candidate, 0, normal=True)
        gradient = np.reshape(gradient, [256, 256, 3])
        softmax = np.reshape(softmax, [256, 256, 2])
        occupy = np.reshape(value, [256, 256])
        value = np.reshape(value, [256, 256])
        laplacian = np.reshape(laplacian, [256, 256])
        #
        #gaussian = np.exp(-value**2/2.5)
        #gradient = np.tile(np.expand_dims(gaussian, axis=2), [1,1,3]) * gradient
        M = np.sqrt(gradient[:,:,0]*gradient[:,:,0] + gradient[:,:,1]*gradient[:,:,1])
        m = np.expand_dims(np.log(M), axis=2)
        #gradient[np.where(m<(0.0001*np.max(m))), :] = 0
        gradient = gradient * m
        M = np.sqrt(M)
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[2, 3])
        ax0 = fig.add_subplot(gs[0, 0])
        vmin = np.min(value)
        vmax = np.max(value)
        if np.abs(vmax) > np.abs(vmin):
            ax0.imshow(value, origin='lower', cmap=plt.get_cmap('jet'), vmin=-vmax, vmax=vmax)
        else:
            ax0.imshow(value, origin='lower', cmap=plt.get_cmap('jet'), vmin=vmin, vmax=-vmin)
        ax0.set_title('Predicted Occupancy Function')
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.imshow(1.0 * (value > 0), origin='lower', alpha=0.3)
        ax1.quiver(np.linspace(0, 256, 256), np.linspace(0, 256, 256), gradient[:,:,1], gradient[:,:,0], linewidth=0.1)
        ax1.set_title('Gradient')
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(1.0 * (value > 0), origin='lower', alpha=0.3)
        ax2.streamplot(np.linspace(0, 256, 256), np.linspace(0, 256, 256), gradient[:,:,1], gradient[:,:,0],
                       #density=[1, 5])
                       color=M[:,:], linewidth=0.5, cmap='autumn', density = 5, arrowstyle='->')
        ax2.set_title('Gradient Flow')

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(laplacian, origin='lower')
        ax3.set_title('Laplacian')

        ax4 = fig.add_subplot(gs[0, 2])
        gradient_soft = np.expand_dims((softmax[:, :, 0] - softmax[:, :, 1]), axis=2) * gradient
        ax4.quiver(np.linspace(0, 256, 256), np.linspace(0, 256, 256), gradient_soft[:,:,1], gradient_soft[:,:,0], linewidth=0.1)
        ax4.set_title('gradient_*_softmax')

        ax5 = fig.add_subplot(gs[1, 2])
        gradient_laplacian = np.expand_dims(laplacian, axis=2) * gradient
        ax5.quiver(np.linspace(0, 256, 256), np.linspace(0, 256, 256), gradient_laplacian[:,:,1], gradient_laplacian[:,:,0], linewidth=0.1)
        ax5.set_title('laplacian_*_softmax')
        plt.savefig('/tmp/pycharm_pcnn_liver/normal_prediction/train_results/stream.png', dpi=2000)

    def construct_next_level(self):
        grid_weight = 1.0 / ((64.0 * 2**self.now_have_level))
        candidate_parent = self.should_split[-1]
        candidate_parent = np.tile(np.expand_dims(candidate_parent, axis=0), [8, 1, 1])
        off_set = [[grid_weight, grid_weight, grid_weight],
                   [-grid_weight, grid_weight, grid_weight],
                   [grid_weight, -grid_weight, grid_weight],
                   [grid_weight, grid_weight, -grid_weight],
                   [grid_weight, -grid_weight, -grid_weight],
                   [-grid_weight, -grid_weight, grid_weight],
                   [-grid_weight, grid_weight, -grid_weight],
                   [-grid_weight, -grid_weight, -grid_weight],]
        candidate = candidate_parent - np.expand_dims(off_set, axis=1)
        candidate = np.reshape(candidate, [-1, 3])
        print(candidate.shape[0])

        occupy, _ = self.querry(candidate, 0)
        points_in_this_level = candidate[np.where(occupy==1)]
        print(points_in_this_level.shape[0])
        self.octree.append(points_in_this_level)
        if self.now_have_level<2:
            self.should_split.append(self.get_split_points_for_level2(candidate, occupy, grid_weight))
        self.now_have_level += 1
        print('level ' + str(self.now_have_level) + ' has been constructed!')

    def get_split_points_for_level2(self, candidate, occupy, grid):
        num_of_grids = int(32 * 2**self.now_have_level)
        grid_weight = 4 * grid
        point_in = np.squeeze(candidate[np.where(occupy == 1), :])
        threeD_index = np.round(point_in/grid_weight).astype(np.int32) + np.asarray([[num_of_grids+1, num_of_grids+1, num_of_grids+1]])
        occupy_matrix = np.zeros(shape=(2*num_of_grids, 2*num_of_grids, 2*num_of_grids), dtype=np.int8)
        occupy_matrix[threeD_index[:, 0], threeD_index[:, 1], threeD_index[:, 2]] = 1
        for i in range(6):
            if i == 0:
                difference = occupy_matrix != self.shift(occupy_matrix, i)
            else:
                difference = difference | (occupy_matrix != self.shift(occupy_matrix, i))

        difference = np.transpose(np.asarray(np.where(difference==True)), [1, 0])
        difference =  difference - np.asarray([[num_of_grids+1, num_of_grids+1, num_of_grids+1]])
        return difference * grid_weight


    def get_split_points_for_level1(self, candidate, occupy, grid_weight):
        point_in = candidate[np.where(occupy==1), :]
        point_out = candidate[np.where(occupy==0), :]
        output = []
        batch_points = int((20480 * 2048 * 3)/point_in.shape[1])
        num_of_runs = int(point_out.shape[1] / batch_points)
        print(point_in.shape[1])
        print(point_out.shape[1])
        for i in range(num_of_runs+1):
            if i == num_of_runs:
                feed_points = point_out[:, (i * batch_points):, :]

            else:
                feed_points = point_out[:, (i*batch_points):((i+1)*batch_points), :]

            distance = point_in - np.transpose(feed_points, [1, 0, 2])
            distance = np.linalg.norm(distance*distance, ord=2, axis=2)
            connect_map = distance < grid_weight
            connect_map1 = np.sum(connect_map, axis=0)
            connect_map2 = np.sum(connect_map, axis=1)
            output.append(np.concatenate([np.squeeze(point_in[0, np.where(connect_map1>0),:]), np.squeeze(feed_points[0, np.where(connect_map2>0),:])], axis=0))
        return np.concatenate(output, axis=0)

    def shift(self, mat, direction):
        mat2 = copy.deepcopy(mat)
        if direction == 0:
            return np.roll(mat2, 1, axis=0)
        elif direction == 1:
            return np.roll(mat2, -1, axis=0)
        elif direction == 2:
            return np.roll(mat2, 1, axis=1)
        elif direction == 3:
            return np.roll(mat2, -1, axis=1)
        elif direction == 4:
            return np.roll(mat2, 1, axis=2)
        elif direction == 5:
            return np.roll(mat2, -1, axis=2)



    def get_split_points_for_level1p(self, candidate, occupy, grid_weight):
        occupy_matrix = np.reshape(occupy, [64, 64, 64])
        self.occupy_matrix.append(occupy_matrix)
        for i in range(6):
            if i == 0:
                difference = occupy_matrix != self.shift(occupy_matrix, i)
            else:
                difference = difference | (occupy_matrix != self.shift(occupy_matrix, i))

        difference = np.reshape(difference, [-1])
        #self.plot_debug(candidate[np.where(difference>0), :])
        return np.squeeze(candidate[np.where(difference>0), :])


    def get_split_points(self, candidate, occupy, grid_weight):
        point_in = candidate[np.where(occupy==1), :]
        point_out = candidate[np.where(occupy==0), :]

        distance = point_in - np.transpose(point_out, [1, 0, 2])
        distance = np.linalg.norm(distance*distance, ord=2, axis=2)
        connect_map = distance < grid_weight
        connect_map1 = np.sum(connect_map, axis=0)
        connect_map2 = np.sum(connect_map, axis=1)
        return np.concatenate([np.squeeze(point_in[0, np.where(connect_map1>0),:]), np.squeeze(point_out[0, np.where(connect_map2>0),:])], axis=0)

    def querry_voronoi(self, points, threshold):
        if not isinstance(threshold, np.ndarray):
            threshold = np.zeros(shape=points.shape[0])
        batch_points = int(2 * self.voronoi_sample_points.shape[0])
        num_of_querry = points.shape[0]
        num_of_runs = int(num_of_querry / batch_points)
        #pred_all = np.zeros(shape=[num_of_querry])
        pred_all_value = np.zeros(shape=[num_of_querry])
        #tri = Delaunay(self.voronoi_sample_points)
        for i in range(num_of_runs+1):
            if i == num_of_runs:
                feed_points = points[(i * batch_points):, :]
            else:
                feed_points = points[(i * batch_points):((i + 1) * batch_points), :]

            pred = tetra_interpolation(feed_points, self.voronoi_sample_points, self.voronoi_sample_value, self.tri_top)

            if i == 0:
                pred_all_value = pred
            else:
                pred_all_value = np.concatenate([pred_all_value, pred])
        pred_all = 1*(pred_all_value>threshold)
        return pred_all, pred_all_value

    def querry(self, points_org, threshold, normal=False):
        points = scale*points_org
        if not isinstance(threshold, np.ndarray):
            threshold = np.zeros(shape=points.shape[0])
        batch_points = (batch_size*probe_num)
        num_of_querry = points.shape[0]
        num_of_runs = int(num_of_querry / batch_points)
        pred_all = np.zeros(shape=[num_of_querry])
        pred_all_value = np.zeros(shape=[num_of_querry])
        pred_all_normal = np.zeros(shape=[num_of_querry, 3])
        pred_all_laplacian = np.zeros(shape=[num_of_querry])
        pred_true_all = np.zeros(shape=[num_of_querry, 2])
        for i in range(num_of_runs+1):
            if i == num_of_runs:
                feed_points = np.zeros(shape=(batch_points, 3))
                feed_threshold = np.zeros(shape=(batch_points))
                feed_points[:(num_of_querry-(i * batch_points)), :] = points[(i * batch_points):, :]
                feed_threshold[:(num_of_querry-(i * batch_points))] = threshold[(i * batch_points):]
            else:
                feed_points = points[(i*batch_points):((i+1)*batch_points), :]
                feed_threshold = threshold[(i*batch_points):((i+1)*batch_points)]
            feed_points = np.reshape(feed_points, [batch_size, probe_num, 3])
            feed_dict = {
                self.tensors[0]: self.pointcloud,
                self.tensors[8]: self.pointlabel,
                self.tensors[1]: feed_points,
                self.tensors[2]: False,
            }

            if not normal:
                pred, normal_pred = self.sess.run([self.tensors[3], self.tensors[4]], feed_dict=feed_dict)
                pred = pred[:, :, 1] - pred[:, :, 0]
                pred = np.squeeze(-pred)
                feed_threshold = np.reshape(feed_threshold,newshape=[batch_size, probe_num])
                pred_value = 1 * ((pred) < feed_threshold)
                pred_value = np.squeeze(pred_value)
                pred_value = np.reshape(pred_value, [-1])
                if i == num_of_runs:
                    pred_all[(i * batch_points):] = pred_value[:(num_of_querry - (i * batch_points))]
                    pred_all_value[(i * batch_points):] = np.reshape(pred-feed_threshold, [-1])[:(num_of_querry - (i * batch_points))]
                else:
                    pred_all[(i * batch_points):((i + 1) * batch_points)] = pred_value
                    pred_all_value[(i * batch_points):((i + 1) * batch_points)] = np.reshape(pred-feed_threshold, [-1])
            else:
                pred, normal_pred, laplacian_pred = self.sess.run([self.tensors[3], self.tensors[4], self.tensors[6]], feed_dict=feed_dict)
                soft = np.reshape(pred, [-1, 2])
                pred = pred[:, :, 0] - pred[:, :, 1]
                pred = np.squeeze(-pred)
                feed_threshold = np.reshape(feed_threshold,newshape=[batch_size, probe_num])
                pred_value = 1 * ((pred) < feed_threshold)
                pred_value = np.squeeze(pred_value)
                pred_value = np.reshape(pred_value, [-1])
                normal_pred = np.reshape(normal_pred, [-1, 3])
                laplacian_pred = np.reshape(laplacian_pred, [-1])
                if i == num_of_runs:
                    pred_all[(i * batch_points):] = pred_value[:(num_of_querry - (i * batch_points))]
                    pred_all_normal[(i * batch_points):, :] = normal_pred[:(num_of_querry - (i * batch_points)), :]
                    pred_all_laplacian[(i * batch_points):] = laplacian_pred[:(num_of_querry - (i * batch_points))]
                    pred_all_value[(i * batch_points):] = np.reshape(pred-feed_threshold, [-1])[:(num_of_querry - (i * batch_points))]
                    pred_true_all[(i * batch_points):, :] = soft[:(num_of_querry - (i * batch_points)), :]
                else:
                    pred_all[(i * batch_points):((i + 1) * batch_points)] = pred_value
                    pred_all_normal[(i * batch_points):((i + 1) * batch_points), :] = normal_pred
                    pred_all_laplacian[(i * batch_points):((i + 1) * batch_points)] = laplacian_pred
                    pred_all_value[(i * batch_points):((i + 1) * batch_points)] = np.reshape(pred-feed_threshold, [-1])
                    pred_true_all[(i * batch_points):((i + 1) * batch_points), :] = soft
        if not normal:
            return pred_all, pred_all_value
        else:
            softmax_all = scipy.special.softmax(pred_true_all, axis=1)
            return pred_all, pred_all_value, pred_all_normal, pred_all_laplacian, pred_true_all, softmax_all

    def querry_gradient(self, points):
        batch_points = (batch_size*1024)
        num_of_querry = points.shape[0]
        num_of_runs = int(num_of_querry / batch_points)
        pred_all = np.zeros(shape=[num_of_querry])
        pred_all_value = np.zeros(shape=[num_of_querry])
        for i in range(num_of_runs+1):
            if i == num_of_runs:
                feed_points = np.zeros(shape=(batch_points, 3))
                feed_points[:(num_of_querry-(i * batch_points)), :] = points[(i * batch_points):, :]
            else:
                feed_points = points[(i*batch_points):((i+1)*batch_points), :]
            feed_points = np.reshape(feed_points, [batch_size, 1024, 3])
            feed_dict = {
                self.tensors[0]: self.pointcloud,
                #self.tensors[5]: self.feature,
                self.tensors[1]: feed_points,
                self.tensors[2]: False,
            }
            pred, normal_pred = self.sess.run([self.tensors[3], self.tensors[4]], feed_dict=feed_dict)
            pred = pred[:, :, 0] - pred[:, :, 1]
            feed_threshold = np.reshape(feed_threshold,newshape=[batch_size, 1024])
            pred_value = 1 * ((pred) < feed_threshold)
            pred_value = np.squeeze(pred_value)
            pred_value = np.reshape(pred_value, [-1])
            if i == num_of_runs:
                pred_all[(i * batch_points):] = pred_value[:(num_of_querry - (i * batch_points))]
                pred_all_value[(i * batch_points):] = np.reshape(pred-feed_threshold, [-1])[:(num_of_querry - (i * batch_points))]
            else:
                pred_all[(i * batch_points):((i + 1) * batch_points)] = pred_value
                pred_all_value[(i * batch_points):((i + 1) * batch_points)] = np.reshape(pred-feed_threshold, [-1])
        return pred_all, pred_all_value


    def duplicate(self):
        for i in range(batch_size):
            self.pointcloud[i, :, :] = self.pointcloud[0, :, :]

    def create_featurelist(self, out_tensor_list):
        featurelist = []
        for out_tensor in out_tensor_list:
            featurelist += Tensor_Retrieval(tf.get_default_graph(), out_tensor, self.tensors[1]).get_dep()

        print(featurelist)

    def init_feature(self):
        self.create_featurelist([self.tensors[3], self.tensors[4]])

        feed_dict = {
            self.tensors[0]: self.pointcloud,
            self.tensors[2]: False,
        }
        self.feature = self.sess.run(self.tensors[5], feed_dict=feed_dict)

    def plot(self):
        if self.now_have_level == 1:
            mlab.points3d(self.pointcloud[0, :, 0], self.pointcloud[0, :, 2], self.pointcloud[0, :, 1], scale_factor=0.05)
            mlab.show()

        '''point_to_plot = self.octree[-1]
        scale = 1.0/(32 * 2**(self.now_have_level-1))
        mlab.points3d(point_to_plot[:, 0], point_to_plot[:, 2], point_to_plot[:, 1],
                      mode='cube', scale_factor=scale, scale_mode='none')
        mlab.show()'''


    def plot_occupy(self, i):
        mlab.figure(bgcolor=(1, 1, 1))
        occupy_to_plot = self.occupy_matrix[i]
        point_to_plot = np.where(occupy_to_plot == 1)
        point_to_plot = np.transpose(np.asarray(point_to_plot))
        mlab.points3d(point_to_plot[:, 0], point_to_plot[:, 2], point_to_plot[:, 1],
                      mode='cube', scale_factor=1.05, scale_mode='none')
        mlab.show()

    def plot_field(self, i):
        field_to_plot = self.occupy_matrix[i]#self.value_matrix[i] * (self.value_matrix[i]>=0.0)
        field_to_plot = np.transpose(field_to_plot, [0, 2, 1]).astype(np.float32)
        field_to_plot = ndimage.gaussian_filter(field_to_plot, sigma=5)
        size = field_to_plot.shape[0]
        mlab.figure(bgcolor=(1,1,1))
        mlab.pipeline.volume(mlab.pipeline.scalar_field(field_to_plot))

        #black = (1, 1, 1)
        #mlab.flow(0, 0, 0, size, size, size, line_width=5.0)
        mlab.show()

    def plot_debug(self, point_to_plot):
        scale = 1.0/(32 * 2**(self.now_have_level-1))
        mlab.points3d(point_to_plot[0, :, 0], point_to_plot[0, :, 1], point_to_plot[0, :, 2],
                      mode='cube', scale_factor=scale, scale_mode='none')
        mlab.show()

    def save_octree(self):
        h5_file_name = "../../saved_results/" + 'octree_example1.h5'
        h5 = h5py.File(h5_file_name, 'w')
        h5.create_dataset('org', data=self.pointcloud)
        h5.create_dataset('octree1', data=self.octree[0])
        h5.create_dataset('octree2', data=self.octree[1])
        h5.create_dataset('octree3', data=self.octree[2])
        h5.close()
        PC_to_off(self.octree[1], '../../saved_results/upsampled_surface.off')


def load_result():
    h5_file_name = "../../saved_results/" + 'octree_example_chair.h5'
    h5 = h5py.File(h5_file_name, 'r')
    org = h5['org'][:]
    octree1 = h5['octree1'][:]
    octree2 = h5['octree2'][:]
    octree3 = h5['octree3'][:]
    h5.close()
    plot_octree(octree1, 1)
    plot_octree(octree2, 2)
    plot_octree(octree3, 3)



def plot_octree(point_to_plot, level):
    scale = 1.0/(32 * 2**(level-1))
    mlab.points3d(point_to_plot[:, 0], point_to_plot[:, 1], point_to_plot[:, 2],
                      mode='cube', scale_factor=scale, scale_mode='none')
    mlab.show()

if __name__ == 'main':
    generator = octree_generator()