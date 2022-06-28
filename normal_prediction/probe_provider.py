import os
import sys
import numpy as np
import h5py
import trimesh
#import pymesh
from trimesh import sample, ray, triangles
from trimesh.ray.ray_triangle import RayMeshIntersector, ray_triangle_id
BASE_DIR = './'
sys.path.append(BASE_DIR)
from provider import Provider
from random import shuffle
import socket
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=5)
#from mayavi import mlab


def plot(point):
    point_to_plot = point.astype(np.float32)
    mlab.points3d(point_to_plot[:, 0], point_to_plot[:, 1], point_to_plot[:, 2], scale_factor=0.05)
    mlab.show()

class ProbeProvider(Provider):
    def __init__(self):
        DATA_DIR = os.path.join(BASE_DIR,'segmentation_data')
        HOSTNAME = socket.gethostname()
        d_dir = open(os.getenv("HOME") + '/Data_dir', 'r')
        self.data_dir = d_dir.read()[:-1]
        self.modelnet_data_dir = '/media/mjia/Data/data_for_occupancy_network/data/ShapeNet.build'
        self.processed_shapenet_data_dir = '/home/mjia/Documents/jointSegmentation/Occupancy-Networks/data_org/ShapeNet'
        #self.modelnet_data_dir = '/media/mjia/Documents/ShapeCompletion/ModelNet40'
        #self.modelnet_data_dir = '/home/mjia/Documents/jointSegmentation/PatchIndicatorNet/data'
        d_dir.close()

        self.train_files = os.path.join(os.path.join(DATA_DIR, 'hdf5_data'),'train_hdf5_file_list.txt')
        self.val_files = os.path.join(os.path.join(DATA_DIR, 'hdf5_data'),'val_hdf5_file_list.txt')

    def getValDataFiles(self):
        return self.getDataFiles(self.val_files)

    def getTrainDataFiles(self):
        return self.getDataFiles(self.train_files)

    def getDataFiles(self,list_filename):
        return [line.rstrip() for line in open(list_filename)]

    def load_h5(self,h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)

    def loadDataFile(self,filename):
        return self.load_h5(os.path.join(os.path.join(BASE_DIR,'hdf5_data')),filename)
        #verts,_ = read_off(filename)
        #return verts

    def read_off(file):
        if 'OFF' != file.readline().strip():
            raise ('Not a valid OFF header')
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = []
        for i_vert in range(n_verts):
            verts.append([float(s) for s in file.readline().strip().split(' ')])
        faces = []
        for i_face in range(n_faces):
            faces.append([int(s) for s in file.readline().strip().split(' ')][1:])
        return verts, faces

    def load_h5_data_label_seg(self,h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:].astype(np.int32)
        seg = f['pid'][:]
        return (data, label, seg)

    def translate_point_cloud3(self,batch_data1, batch_data2, batch_data3,):

        translated_data1 = np.zeros(batch_data1.shape, dtype=np.float32)
        translated_data2 = np.zeros(batch_data2.shape, dtype=np.float32)
        translated_data3 = np.zeros(batch_data3.shape, dtype=np.float32)
        for k in range(batch_data1.shape[0]):
            xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
            xyz2 = np.random.uniform(low=-0.2,high=0.2,size=[3])

            shape_pc1 = batch_data1[k, ...]
            shape_pc2 = batch_data2[k, ...]
            shape_pc3 = batch_data3[k, ...]

            translated_data1[k, ...] = np.add(np.multiply(shape_pc1, xyz1), xyz2)
            translated_data2[k, ...] = np.add(np.multiply(shape_pc2, xyz1), xyz2)
            translated_data3[k, ...] = np.add(np.multiply(shape_pc3, xyz1), xyz2)
        return translated_data1, translated_data2, translated_data3


    def loadDataFile_with_seg(self,filename):
        return self.load_h5_data_label_seg(filename)

    def provide_data(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE):
        if is_train:
            file = open(self.data_dir + '/03001627/03001627train.txt', 'r')
        else:
            file = open(self.data_dir + '/03001627/03001627val.txt', 'r')
        lines = file.read().split('\n')
        shuffle(lines)
        file.close()

        output_pointcloud = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_probepoint = np.zeros(shape=(BATCH_SIZE, 2*NUM_PROBE, 3), dtype=np.float32)
        output_label = np.ones(shape=(BATCH_SIZE, NUM_PROBE), dtype=np.int32)
        #output_weight = np.zeros(shape=(BATCH_SIZE, 1024), dtype=np.float32)
        output_label3 = np.concatenate((0*output_label, output_label), axis=1)
        filled = 0
        for id in lines:
            try:
                h5f = h5py.File(self.data_dir + '/03001627/' + id + "/model1points.h5", 'r')
                #h5f2 = h5py.File(data_dir + '/03001627/' + id + "/model1pointsweight.h5", 'r')
            except:
                continue
            else:
                n = np.sum(h5f['in_out_lable'])
                total = h5f['in_out_lable'][:].shape[0]
                if n<NUM_PROBE or (total-n)<NUM_PROBE:
                    continue
                else:
                    shuffled_points_on = h5f['points_on'][:]
                    np.random.shuffle(shuffled_points_on)
                    output_pointcloud[filled, :, :] = shuffled_points_on[0:NUM_POINT, :]
                    #output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]

                    index = np.argsort(h5f['in_out_lable'][:])
                    index_of_out = index[0:total-n]
                    index_of_in = index[total-n:]
                    np.random.shuffle(index_of_out)
                    np.random.shuffle(index_of_in)
                    output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]#h5f['points_in_out'][:][index_of_in[:NUM_PROBE], :]
                    output_probepoint[filled, NUM_PROBE:2*NUM_PROBE, :] = h5f['points_in_out'][:][index_of_out[:NUM_PROBE], :]
                    #output_probepoint[filled, NUM_PROBE:2 * NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]

                    filled += 1
                    h5f.close()
                    if filled == BATCH_SIZE:
                        filled = 0
                        yield output_pointcloud, output_probepoint, output_label3


    def provide_data2(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE):
        HALF_PROBE = int(NUM_PROBE/2)
        if is_train:
            file = open(self.data_dir + '/04256520/04256520train.txt', 'r')
        else:
            file = open(self.data_dir + '/04256520/04256520val.txt', 'r')
        lines = file.read().split('\n')
        shuffle(lines)
        file.close()

        output_pointcloud = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_probepoint = np.zeros(shape=(BATCH_SIZE, 2*NUM_PROBE, 3), dtype=np.float32)
        output_label = np.ones(shape=(BATCH_SIZE, NUM_PROBE), dtype=np.int32)
        #output_weight = np.zeros(shape=(BATCH_SIZE, 1024), dtype=np.float32)
        output_label3 = np.concatenate((0*output_label, output_label), axis=1)
        filled = 0
        for id in lines:
            try:
                h5f = h5py.File(self.data_dir + '/04256520/' + id + "/model1points.h5", 'r')
                #h5f2 = h5py.File(data_dir + '/03001627/' + id + "/model1pointsweight.h5", 'r')
            except:
                continue
            else:
                n = np.sum(h5f['in_out_lable'])
                total = h5f['in_out_lable'][:].shape[0]
                if n<NUM_PROBE or (total-n)<NUM_PROBE:
                    continue
                else:
                    shuffled_points_on = h5f['points_on'][:]
                    np.random.shuffle(shuffled_points_on)
                    output_pointcloud[filled, :, :] = shuffled_points_on[0:NUM_POINT, :]
                    #output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]

                    index = np.argsort(h5f['in_out_lable'][:])
                    index_of_out = index[0:total-n]
                    index_of_in = index[total-n:]
                    np.random.shuffle(index_of_out)
                    np.random.shuffle(index_of_in)
                    output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]
                    output_probepoint[filled, NUM_PROBE:(NUM_PROBE+HALF_PROBE), :] = h5f['points_in_out'][:][index_of_out[:HALF_PROBE], :]
                    output_probepoint[filled, (NUM_PROBE+HALF_PROBE):(2*NUM_PROBE), :] = h5f['points_in_out'][:][index_of_in[:HALF_PROBE], :]

                    filled += 1
                    h5f.close()
                    if filled == BATCH_SIZE:
                        filled = 0
                        yield output_pointcloud, output_probepoint, output_label3



    def provide_data3(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE):
        if is_train:
            file = open(self.data_dir + '/04256520/04256520train.txt', 'r')
        else:
            file = open(self.data_dir + '/04256520/04256520val.txt', 'r')
        lines = file.read().split('\n')
        shuffle(lines)
        file.close()

        output_pointcloud = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_probepoint = np.zeros(shape=(BATCH_SIZE, 2*NUM_PROBE, 3), dtype=np.float32)
        output_label = np.ones(shape=(BATCH_SIZE, NUM_PROBE), dtype=np.int32)
        #output_weight = np.zeros(shape=(BATCH_SIZE, 1024), dtype=np.float32)
        output_label3 = np.concatenate((0*output_label, output_label), axis=1)
        filled = 0
        for id in lines:
            try:
                h5f = h5py.File(self.data_dir + '/04256520/' + id + "/model1points.h5", 'r')
                #h5f2 = h5py.File(data_dir + '/03001627/' + id + "/model1pointsweight.h5", 'r')
            except:
                continue
            else:
                n = np.sum(h5f['in_out_lable'])
                total = h5f['in_out_lable'][:].shape[0]
                if n<NUM_PROBE or (total-n)<NUM_PROBE:
                    continue
                else:
                    shuffled_points_on = h5f['points_on'][:]
                    shuffled_points_on = partial_sample(shuffled_points_on)
                    if shuffled_points_on.shape[0] < NUM_POINT:
                        continue
                    np.random.shuffle(shuffled_points_on)
                    output_pointcloud[filled, :, :] = shuffled_points_on[0:NUM_POINT, :]
                    #output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]

                    index = np.argsort(h5f['in_out_lable'][:])
                    index_of_out = index[0:total-n]
                    index_of_in = index[total-n:]
                    np.random.shuffle(index_of_out)
                    np.random.shuffle(index_of_in)
                    output_probepoint[filled, 0:NUM_PROBE, :] = h5f['points_in_out'][:][index_of_in[:NUM_PROBE], :]
                    output_probepoint[filled, NUM_PROBE:2*NUM_PROBE, :] = h5f['points_in_out'][:][index_of_out[:NUM_PROBE], :]

                    filled += 1
                    h5f.close()
                    if filled == BATCH_SIZE:
                        filled = 0
                        yield output_pointcloud, output_probepoint, output_label3


    def provide_data4(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE):
        if is_train:
            file = open(self.data_dir + '/04256520/04256520train.txt', 'r')
        else:
            file = open(self.data_dir + '/04256520/04256520val.txt', 'r')
        lines = file.read().split('\n')
        shuffle(lines)
        file.close()

        output_pointcloud = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_probepoint = np.zeros(shape=(BATCH_SIZE, 2*NUM_PROBE, 3), dtype=np.float32)
        output_label = np.ones(shape=(BATCH_SIZE, NUM_PROBE), dtype=np.int32)
        #output_weight = np.zeros(shape=(BATCH_SIZE, 1024), dtype=np.float32)
        output_label3 = np.concatenate((0*output_label, output_label), axis=1)
        filled = 0
        for id in lines:
            try:
                h5f = h5py.File(self.data_dir + '/04256520/' + id + "/model1points.h5", 'r')
                h5f2 = h5py.File(self.data_dir + '/04256520/' + id + "/model1partialview.h5", 'r')
                #h5f2 = h5py.File(data_dir + '/03001627/' + id + "/model1pointsweight.h5", 'r')
            except:
                continue
            else:
                n = np.sum(h5f['in_out_lable'])
                total = h5f['in_out_lable'][:].shape[0]
                if n<NUM_PROBE or (total-n)<NUM_PROBE:
                    continue
                else:
                    shuffled_points_on = h5f2['partialview'][:]
                    np.random.shuffle(shuffled_points_on)
                    output_pointcloud[filled, :, :] = shuffled_points_on
                    #output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]

                    index = np.argsort(h5f['in_out_lable'][:])
                    index_of_out = index[0:total-n]
                    index_of_in = index[total-n:]
                    np.random.shuffle(index_of_out)
                    np.random.shuffle(index_of_in)
                    output_probepoint[filled, 0:NUM_PROBE, :] = h5f['points_in_out'][:][index_of_in[:NUM_PROBE], :]
                    output_probepoint[filled, NUM_PROBE:2*NUM_PROBE, :] = h5f['points_in_out'][:][index_of_out[:NUM_PROBE], :]
                    #output_probepoint[filled, 0:2*NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+2*NUM_PROBE), :]

                    filled += 1
                    h5f.close()
                    if filled == BATCH_SIZE:
                        filled = 0
                        yield output_pointcloud, output_probepoint, output_label3

    def provide_modelnet_data(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE):
        if is_train:
            file = open(self.modelnet_data_dir + '/train.txt', 'r')
        else:
            file = open(self.modelnet_data_dir + '/test.txt', 'r')
        lines = file.read().split('\n')
        shuffle(lines)
        file.close()

        output_pointcloud = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_surfacenormal = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_surfacepoint = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_probepoint = np.zeros(shape=(BATCH_SIZE, 2*NUM_PROBE, 3), dtype=np.float32)
        output_label = np.ones(shape=(BATCH_SIZE, NUM_PROBE), dtype=np.int32)
        #output_weight = np.zeros(shape=(BATCH_SIZE, 1024), dtype=np.float32)
        output_label3 = np.concatenate((0*output_label, output_label), axis=1)
        filled = 0
        nn = 0
        for id in lines:
            nn += 1
            if nn > 20000:
                break
            #if len(id.split('/')) != 9:
            #    continue
            #if id.split('/')[-3] != 'chair':
            #    continue
            try:
                h5f = h5py.File(id, 'r')
                #h5f2 = h5py.File(data_dir + '/03001627/' + id + "/model1pointsweight.h5", 'r')
            except:
                continue
            else:
                n = np.sum(h5f['in_out_lable'])
                total = h5f['in_out_lable'][:].shape[0]
                if n<NUM_PROBE or (total-n)<NUM_PROBE:
                    continue
                else:
                    pointcloud = h5f['points_on'][:]
                    surfacenoemal = h5f['points_normal'][:]
                    shuffled_points_on = np.concatenate((pointcloud, surfacenoemal), axis=1)
                    np.random.shuffle(shuffled_points_on)
                    output_pointcloud[filled, :, :] = shuffled_points_on[0:NUM_POINT, :3]
                    output_surfacenormal[filled, :, :] = shuffled_points_on[NUM_POINT:(2*NUM_POINT), 3:]
                    output_surfacepoint[filled, :, :] = shuffled_points_on[NUM_POINT:(2*NUM_POINT), :3]
                    #output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]

                    index = np.argsort(h5f['in_out_lable'][:])
                    index_of_out = index[0:total-n]
                    index_of_in = index[total-n:]
                    np.random.shuffle(index_of_out)
                    np.random.shuffle(index_of_in)
                    output_probepoint[filled, 0:NUM_PROBE, :] = h5f['points_in_out'][:][index_of_in[:NUM_PROBE], :]
                    output_probepoint[filled, NUM_PROBE:2*NUM_PROBE, :] = h5f['points_in_out'][:][index_of_out[:NUM_PROBE], :]
                    #output_probepoint[filled, 0:2*NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+2*NUM_PROBE), :3]

                    filled += 1
                    h5f.close()
                    if filled == BATCH_SIZE:
                        filled = 0
                        distance_probe = np.expand_dims(output_pointcloud[:, :3], axis=1) - np.expand_dims(output_probepoint, axis=2)
                        distance_probe = np.sum(distance_probe*distance_probe, axis=3)
                        distance_probe = np.min(distance_probe, axis=2)
                        distance_probe = 2 * distance_probe * (output_label3 - 0.5*np.ones_like(output_label3))
                        #output_pointcloud, output_probepoint, output_surfacepoint = self.translate_point_cloud3(output_pointcloud, output_probepoint, output_surfacepoint)
                        yield (output_pointcloud+np.random.normal(scale=0.005, size=(BATCH_SIZE, NUM_POINT, 3))),\
                              output_probepoint, output_label3, output_surfacenormal, output_surfacepoint, distance_probe

    def provide_modelnet_data_nothalf_half(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE):
        if is_train:
            file = open(self.modelnet_data_dir + '/train_distance.txt', 'r')
        else:
            file = open(self.modelnet_data_dir + '/test_distance.txt', 'r')
        lines = file.read().split('\n')
        shuffle(lines)
        file.close()

        output_pointcloud = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_surfacenormal = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_surfacepoint = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_probepoint = np.zeros(shape=(BATCH_SIZE, 2*NUM_PROBE, 3), dtype=np.float32)
        #output_label = np.ones(shape=(BATCH_SIZE, NUM_PROBE), dtype=np.int32)
        #output_weight = np.zeros(shape=(BATCH_SIZE, 1024), dtype=np.float32)
        output_label3 = np.ones(shape=(BATCH_SIZE, 2*NUM_PROBE), dtype=np.int32)
        filled = 0
        nn = 0
        for id in lines:
            nn += 1
            if nn > 20000:
                break
            #if len(id.split('/')) != 9:
            #    continue
            #if id.split('/')[-3] != 'chair':
            #    continue
            try:
                h5f = h5py.File(id, 'r')
                #h5f2 = h5py.File(data_dir + '/03001627/' + id + "/model1pointsweight.h5", 'r')
            except:
                continue
            else:
                n = np.sum(h5f['in_out_lable'])
                total = h5f['in_out_lable'][:].shape[0]
                if False:#n<NUM_PROBE or (total-n)<NUM_PROBE:
                    continue
                else:
                    pointcloud = h5f['points_on'][:]
                    surfacenoemal = h5f['points_normal'][:]
                    shuffled_points_on = np.concatenate((pointcloud, surfacenoemal), axis=1)
                    np.random.shuffle(shuffled_points_on)
                    output_pointcloud[filled, :, :] = shuffled_points_on[0:NUM_POINT, :3]
                    output_surfacenormal[filled, :, :] = shuffled_points_on[NUM_POINT:(2*NUM_POINT), 3:]
                    output_surfacepoint[filled, :, :] = shuffled_points_on[NUM_POINT:(2*NUM_POINT), :3]
                    #output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]

                    index = np.asarray(range(len(h5f['in_out_lable'][:])))
                    np.random.shuffle(index)
                    #np.argsort(h5f['in_out_lable'][:])
                    #index_of_out = index[0:total-n]
                    #index_of_in = index[total-n:]
                    #np.random.shuffle(index_of_out)
                    #np.random.shuffle(index_of_in)
                    output_probepoint[filled, :, :] = h5f['points_in_out'][:][index[:2*NUM_PROBE], :]
                    output_label3[filled, :] = h5f['in_out_lable'][:][index[:2*NUM_PROBE]]
                    #output_probepoint[filled, NUM_PROBE:2*NUM_PROBE, :] = h5f['points_in_out'][:][index_of_out[:NUM_PROBE], :]
                    #output_probepoint[filled, 0:2*NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+2*NUM_PROBE), :3]

                    filled += 1
                    h5f.close()
                    if filled == BATCH_SIZE:
                        filled = 0
                        distance_probe = np.expand_dims(output_pointcloud[:, :3], axis=1) - np.expand_dims(output_probepoint, axis=2)
                        distance_probe = np.sum(distance_probe*distance_probe, axis=3)
                        distance_probe = np.min(distance_probe, axis=2)
                        distance_probe = 2 * distance_probe * (output_label3 - 0.5*np.ones_like(output_label3))
                        #output_pointcloud, output_probepoint, output_surfacepoint = self.translate_point_cloud3(output_pointcloud, output_probepoint, output_surfacepoint)
                        yield (output_pointcloud+np.random.normal(scale=0.005, size=(BATCH_SIZE, NUM_POINT, 3))),\
                              output_probepoint, output_label3, output_surfacenormal, output_surfacepoint, distance_probe                        
                        
                        
    def provide_processed_shapenet_data(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE):
        if is_train:
            file = open(self.processed_shapenet_data_dir + '/train.txt', 'r')
        else:
            file = open(self.processed_shapenet_data_dir + '/test.txt', 'r')
        lines = file.read().split('\n')
        shuffle(lines)
        file.close()

        output_pointcloud = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        output_probepoint = np.zeros(shape=(BATCH_SIZE, 2*NUM_PROBE, 3), dtype=np.float32)
        output_label = np.ones(shape=(BATCH_SIZE, 2*NUM_PROBE), dtype=np.int32)
        filled = 0
        nn = 0
        for id in lines:
            nn += 1
            if nn > 20000:
                break
            #if len(id.split('/')) != 9:
            #    continue
            #if id.split('/')[-3] != 'chair':
            #    continue
            try:
                pointcloud, probe_points, label = load_processed_shapenet(id, id[:-9]+'s.npz', NUM_POINT, NUM_PROBE)
            except:
                continue
            else:
                output_pointcloud[filled, :, :] = pointcloud
                output_probepoint[filled, :, :] = probe_points
                output_label[filled, :] = label

                filled += 1
                if filled == BATCH_SIZE:
                    filled = 0
                    yield output_pointcloud, output_probepoint, output_label
                    


def load_processed_shapenet(pointcloud_f, points_f, NUM_POINT, NUM_PROBE):
    pointcloud_data = np.load(pointcloud_f)
    points_data = np.load(points_f)

    if pointcloud_data['scale'] == points_data['scale']:
        scale = pointcloud_data['scale']
    else:
        raise Exception('scale not equal')

    if any(pointcloud_data['loc'] != np.asarray([0,0,0])) or any(points_data['loc'] != np.asarray([0,0,0])):
        raise Exception('loc not zero')

    pointcloud = pointcloud_data['points']
    probe_points = points_data['points']
    np.random.shuffle(pointcloud)
    label = np.unpackbits(points_data['occupancies'], axis=0)
    index_in = np.where(label == 1)[0]
    index_out = np.where(label ==0)[0]
    if len(index_in) < NUM_PROBE or len(index_out) < NUM_PROBE:
        raise Exception('too less points')
    np.random.shuffle(index_in)
    np.random.shuffle(index_out)

    output_pointcloud = pointcloud[:NUM_POINT, :] + np.random.normal(scale=0.05, size=(NUM_POINT, 3))# / scale
    output_points1 = probe_points[index_in[:NUM_PROBE], :]# /scale
    output_points2 = probe_points[index_out[:NUM_PROBE], :]
    output_points = np.concatenate([output_points1, output_points2], axis=0)
    output_label = np.concatenate([np.ones(shape=NUM_PROBE), np.zeros(shape=NUM_PROBE)], axis=0)
    #plot(output_pointcloud)
    #plot(output_points[np.where(output_label==1)[0], :])
    return 2*output_pointcloud, 2*output_points, output_label

def random_view_point():
    direction = np.empty((3))
    direction[2] = 0.5 * np.random.rand()
    direction[1] = np.random.rand() - 0.5
    direction[0] = np.random.rand() - 0.5
    rangm = 10 * (1 + np.random.rand())

    # random view point
    viewpoint = rangm * direction / np.linalg.norm(direction)
    return viewpoint


def view_partial_sample(id, n):
    viewpoint = random_view_point()
    # this method retures a n*3 array that indicates the pionts sampled from the visible surface
    mesh = trimesh.load(id)
    location_v = np.empty((n, 3))
    number_of_visibles = 0
    r = RayMeshIntersector(mesh)


    init_sample = sample.sample_surface(mesh, int(n/4))[0]

    ray_origins = np.tile(np.expand_dims(viewpoint, axis=0), [n, 1])
    ray_directions = np.concatenate([init_sample]*4, axis=0) - ray_origins

    [index_tri, index_ray, location] = r.intersects_id(ray_origins, ray_directions, return_locations=True,
                                                       multiple_hits=False)
    scale = np.max(mesh.bounding_box.vertices[:, :])

    return (0.5 / scale)*location
    # delete unseen
    # visible_index = []
    '''for i in range(index_ray.shape[0]):
        if not (abs(location[i, 0] - init_sample[index_ray[i], 0]) + abs(
                location[i, 1] - init_sample[index_ray[i], 1]) + abs(
            location[i, 2] - init_sample[index_ray[i], 2])) > 0.0001:
            location_v[number_of_visibles, :] = location[i, :]
            number_of_visibles += 1
            if number_of_visibles == n:
                return location_v'''

def partial_sample(points_on):
    direction = np.empty((3))
    direction[2] = 0.5 * np.random.rand()
    direction[1] = np.random.rand() - 0.5
    direction[0] = np.random.rand() - 0.5
    norm = np.linalg.norm(direction)
    direction = norm * direction
    rotated_points = np.matmul(points_on, direction)
    points_out = np.squeeze(points_on[np.argwhere(rotated_points > 0), :])

    return points_out


#load_processed_shapenet('/home/mjia/Documents/jointSegmentation/Occupancy-Networks/data_org/ShapeNet/02691156/d2412f19d33572dc4c3a35cee92bb95b/pointcloud.npz',\
#                        '/home/mjia/Documents/jointSegmentation/Occupancy-Networks/data_org/ShapeNet/02691156/d2412f19d33572dc4c3a35cee92bb95b/points.npz', 300, 400)

#pv = ProbeProvider()
#train_generator = pv.provide_processed_shapenet_data(True, 5, 300, 400)
#for a in train_generator:
#    print(' ')
