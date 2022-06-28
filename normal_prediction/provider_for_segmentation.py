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

class ProbeProvider(Provider):
    def __init__(self):
        DATA_DIR = os.path.join(BASE_DIR,'segmentation_data')
        HOSTNAME = socket.gethostname()
        d_dir = open(os.getenv("HOME") + '/Data_dir', 'r')
        self.data_dir = d_dir.read()[:-1]
        self.modelnet_data_dir = '/media/mjia/Data/ShapeNetSegmentation/labeled_meshes/SHAPENET_MESHES'
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

    def provide_seg_data(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE):
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
        output_seg_label = np.zeros(shape=(BATCH_SIZE, 2*NUM_PROBE), dtype=np.int32)
        output_seg_label_on = np.zeros(shape=(BATCH_SIZE, NUM_POINT), dtype=np.int32)
        filled = 0
        num = 0
        for id in lines:
            num += 1
            #if num>100:
            #    break
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
                    seg_label_on = np.expand_dims(h5f['seg_lable_on'][:], axis=1).astype(np.float32)
                    shuffled_points_on = np.concatenate((pointcloud, surfacenoemal, seg_label_on), axis=1)
                    np.random.shuffle(shuffled_points_on)
                    output_pointcloud[filled, :, :] = shuffled_points_on[0:NUM_POINT, :3]
                    output_surfacenormal[filled, :, :] = shuffled_points_on[NUM_POINT:(2*NUM_POINT), 3:6]
                    output_surfacepoint[filled, :, :] = shuffled_points_on[NUM_POINT:(2*NUM_POINT), :3]
                    output_seg_label_on[filled, :] = shuffled_points_on[0:NUM_POINT, 6].astype(np.int32)
                    #output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]

                    index = np.argsort(h5f['in_out_lable'][:])
                    index_of_out = index[0:total-n]
                    index_of_in = index[total-n:]
                    np.random.shuffle(index_of_out)
                    np.random.shuffle(index_of_in)
                    output_probepoint[filled, 0:NUM_PROBE, :] = h5f['points_in_out'][:][index_of_in[:NUM_PROBE], :]
                    output_probepoint[filled, NUM_PROBE:2*NUM_PROBE, :] = h5f['points_in_out'][:][index_of_out[:NUM_PROBE], :]
                    #output_probepoint[filled, 0:2*NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+2*NUM_PROBE), :3]

                    output_seg_label[filled, 0:NUM_PROBE] = h5f['seg_lable_inout'][:][index_of_in[:NUM_PROBE]]
                    output_seg_label[filled, NUM_PROBE:2*NUM_PROBE] = h5f['seg_lable_inout'][:][index_of_out[:NUM_PROBE]]


                    filled += 1
                    h5f.close()
                    if filled == BATCH_SIZE:
                        filled = 0
                        distance_probe = np.expand_dims(output_pointcloud[:, :3], axis=1) - np.expand_dims(output_probepoint, axis=2)
                        distance_probe = np.sum(distance_probe*distance_probe, axis=3)
                        distance_probe = np.min(distance_probe, axis=2)
                        distance_probe = 2 * distance_probe * (output_label3 - 0.5*np.ones_like(output_label3))
                        yield output_pointcloud, output_probepoint, output_label3, output_surfacenormal, output_surfacepoint, distance_probe, output_seg_label, output_seg_label_on

    def provide_seg_data_test(self, BATCH_SIZE, NUM_POINT, NUM_PROBE):
        file = open('/media/mjia/Data/ShapeNetSegmentation/shapenetcore_partanno_segmentation_benchmark_v0' + '/test.txt', 'r')
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
        output_seg_label = np.zeros(shape=(BATCH_SIZE, 2*NUM_PROBE), dtype=np.int32)
        output_seg_label_on = np.zeros(shape=(BATCH_SIZE, NUM_POINT), dtype=np.int32)
        filled = 0
        num = 0
        for id in lines:
            num += 1
            if num>5000:
                break
            try:
                h5f = h5py.File(id, 'r')
                #h5f2 = h5py.File(data_dir + '/03001627/' + id + "/model1pointsweight.h5", 'r')
            except:
                continue
            else:
                pointcloud = h5f['points_on'][:]
                if pointcloud.shape[0] < NUM_POINT:
                    continue
                seg_label_on = np.expand_dims(h5f['seg_lable_on'][:], axis=1).astype(np.float32)
                shuffled_points_on = np.concatenate((pointcloud, seg_label_on), axis=1)
                np.random.shuffle(shuffled_points_on)
                output_pointcloud[filled, :, :] = shuffled_points_on[0:NUM_POINT, :3]
                #output_surfacepoint[filled, :, :] = shuffled_points_on[NUM_POINT:(2*NUM_POINT), :3]
                output_seg_label_on[filled, :] = shuffled_points_on[0:NUM_POINT, 3].astype(np.int32)
                #output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]


                filled += 1
                h5f.close()
                if filled == BATCH_SIZE:
                    filled = 0
                    distance_probe = np.expand_dims(output_pointcloud[:, :3], axis=1) - np.expand_dims(output_probepoint, axis=2)
                    distance_probe = np.sum(distance_probe*distance_probe, axis=3)
                    distance_probe = np.min(distance_probe, axis=2)
                    distance_probe = 2 * distance_probe * (output_label3 - 0.5*np.ones_like(output_label3))
                    yield output_pointcloud, output_probepoint, output_label3, output_surfacenormal, output_surfacepoint, distance_probe, output_seg_label, output_seg_label_on

    def provide_seg_data_train(self, BATCH_SIZE, NUM_POINT, NUM_PROBE):
        file = open('/media/mjia/Data/ShapeNetSegmentation/shapenetcore_partanno_segmentation_benchmark_v0' + '/train.txt', 'r')
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
        output_seg_label = np.zeros(shape=(BATCH_SIZE, 2*NUM_PROBE), dtype=np.int32)
        output_seg_label_on = np.zeros(shape=(BATCH_SIZE, NUM_POINT), dtype=np.int32)
        filled = 0
        num = 0
        for id in lines:
            num += 1
            #if num>100:
            #    break
            try:
                h5f = h5py.File(id, 'r')
                #h5f2 = h5py.File(data_dir + '/03001627/' + id + "/model1pointsweight.h5", 'r')
            except:
                continue
            else:
                pointcloud = h5f['points_on'][:]
                if pointcloud.shape[0] < NUM_POINT:
                    continue
                seg_label_on = np.expand_dims(h5f['seg_lable_on'][:], axis=1).astype(np.float32)
                shuffled_points_on = np.concatenate((pointcloud, seg_label_on), axis=1)
                np.random.shuffle(shuffled_points_on)
                output_pointcloud[filled, :, :] = shuffled_points_on[0:NUM_POINT, :3]
                #output_surfacepoint[filled, :, :] = shuffled_points_on[NUM_POINT:(2*NUM_POINT), :3]
                output_seg_label_on[filled, :] = shuffled_points_on[0:NUM_POINT, 3].astype(np.int32)
                #output_probepoint[filled, 0:NUM_PROBE, :] = shuffled_points_on[NUM_POINT:(NUM_POINT+NUM_PROBE), :]

                index = np.asarray(range(h5f['points_in_out'][:].shape[0]))
                np.random.shuffle(index)
                output_probepoint[filled, 0:NUM_POINT, :] = h5f['points_in_out'][:][index[:NUM_POINT], :]

                output_seg_label[filled, 0:NUM_POINT] = h5f['seg_lable_inout'][:][index[:NUM_POINT]]

                filled += 1
                h5f.close()
                if filled == BATCH_SIZE:
                    filled = 0
                    distance_probe = np.expand_dims(output_pointcloud[:, :3], axis=1) - np.expand_dims(output_probepoint, axis=2)
                    distance_probe = np.sum(distance_probe*distance_probe, axis=3)
                    distance_probe = np.min(distance_probe, axis=2)
                    distance_probe = 2 * distance_probe * (output_label3 - 0.5*np.ones_like(output_label3))
                    yield output_pointcloud, output_probepoint, output_label3, output_surfacenormal, output_surfacepoint, distance_probe, output_seg_label, output_seg_label_on

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



#pv = ProbeProvider()
#train_generator = pv.provide_seg_data(True, 5, 1024, 512)
#for a in train_generator:
#    print(' ')