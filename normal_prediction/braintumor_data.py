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
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from provider import Provider
from random import shuffle
import socket
import string
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=5)
#from mayavi import mlab

def handle_punctuation(input):
    output = input.replace('.', '我')
    output = output.translate(str.maketrans('', '', string.punctuation))
    output = output.replace('我', '.')
    output = output.replace(' ', '_')
    return output


class ProbeProvider(Provider):
    def __init__(self, cfg):
        self.cfg = cfg
        self.training_data_dir = cfg['basics']['dataset_dir']
        self.training_data_file = self.training_data_dir + '/' + cfg['basics']['traindata_file']
        self.num_train_files = cfg['basics']['total_training']
        self.num_val_files = cfg['basics']['total_validation']

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

    def provide_data(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE, shuffle=True, get_affine=False, is_debug=False):
        hf = h5py.File(self.training_data_file, 'r')
        tumor_PC = hf['tumor_PC'][:]
        tumor_PC_displacement = hf['tumor_PC_displacement'][:]
        querry_PC = hf['querry_PC'][:]
        querry_PC_displacement = hf['querry_PC_displacement'][:]
        affine_ALL = hf['affine_matrix'][:]
        hf.close()
        if is_debug:
            num =100
            tumor_PC = tumor_PC
            tumor_PC_displacement = tumor_PC_displacement
            querry_PC = querry_PC
            querry_PC_displacement = querry_PC_displacement
            affine_ALL = affine_ALL
            indices = np.arange(100) + 331
        else:
            if is_train:
                num = self.num_train_files
                tumor_PC = tumor_PC[:num]
                tumor_PC_displacement = tumor_PC_displacement[:num]
                querry_PC = querry_PC[:num]
                querry_PC_displacement = querry_PC_displacement[:num]
                affine_ALL = affine_ALL[:num]
                if shuffle:
                    indices = np.random.permutation(num)
                else:
                    indices = np.arange(num)
            else:
                num = self.num_val_files
                tumor_PC = tumor_PC[-num:]
                tumor_PC_displacement = tumor_PC_displacement[-num:]
                querry_PC = querry_PC[-num:]
                querry_PC_displacement = querry_PC_displacement[-num:]
                affine_ALL = affine_ALL[-num:]
                if shuffle:
                    indices = np.random.permutation(num)
                else:
                    indices = np.arange(num)

        output_pointcloud = np.zeros(shape=(self.cfg['training']['batch_size'], self.cfg['training']['num_points'], 3), dtype=np.float32)
        output_point_disp = np.zeros(shape=(self.cfg['training']['batch_size'], self.cfg['training']['num_points'], 3), dtype=np.float32)
        output_extend = np.zeros(shape=(self.cfg['training']['batch_size'], 3), dtype=np.float32)
        output_querrypoint = np.zeros(shape=(self.cfg['training']['batch_size'], self.cfg['training']['num_sample_points'], 3), dtype=np.float32)
        output_querry_disp = np.zeros(shape=(self.cfg['training']['batch_size'], self.cfg['training']['num_sample_points'], 3), dtype=np.float32)
        output_affine = np.zeros(shape=(self.cfg['training']['batch_size'], 4, 4), dtype=np.float32)
        filled = 0
        for i in range(num):
            pointcloud = tumor_PC[indices[i]]
            point_disp = tumor_PC_displacement[indices[i]]
            querrypoint = querry_PC[indices[i]]
            #querrypoint = np.concatenate([querrypoint[:1000, :], querrypoint[:1000, :]],axis=0)
            querry_disp = querry_PC_displacement[indices[i]]
            #querry_disp = np.concatenate([querry_disp[:1000, :], querry_disp[:1000, :]], axis=0)

            p1 = np.random.permutation(pointcloud.shape[0])[:self.cfg['training']['num_points']]
            output_pointcloud[filled, :, :] = pointcloud[p1, :]
            output_point_disp[filled, :, :] = point_disp[p1]
            output_extend[filled, :] = np.asarray([np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]),
                                                   np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
                                                   np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
            p2 = np.random.permutation(querrypoint.shape[0])[:self.cfg['training']['num_sample_points']]
            output_querrypoint[filled, :, :] = querrypoint[p2, :]
            output_querry_disp[filled, :, :] = querry_disp[p2, :]

            output_affine[filled, :, :] = affine_ALL[indices[i]]

            filled += 1
            if filled == BATCH_SIZE:
                filled = 0
                #with h5py.File('/tmp/Liver_Phantom_data/src/sparse_dataset/debug.h5', 'w') as f:
                #    f.create_dataset('output_pointcloud', data=output_pointcloud)
                #if is_train:
                #    output_pointcloud += np.random.normal(0.0, 2e-3, size=output_pointcloud.shape)#1e-3, 2e-3
                if get_affine:
                    yield output_pointcloud, output_point_disp, output_querrypoint, output_querry_disp, \
                          np.ones(shape=(self.cfg['training']['batch_size'], self.cfg['training']['num_points']), dtype=np.float32), \
                          output_extend, \
                          indices[i-BATCH_SIZE+1 : i+1], output_affine
                else:
                    yield output_pointcloud, output_point_disp, output_querrypoint, output_querry_disp, \
                          np.ones(shape=(self.cfg['training']['batch_size'], self.cfg['training']['num_points']), dtype=np.float32), \
                          output_extend, \
                          indices[i-BATCH_SIZE+1 : i+1]

    def provide_data_edma(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE, shuffle=True, get_affine=False, is_debug=False):
        hf = h5py.File(self.training_data_file, 'r')
        tumor_PC = hf['tumor_PC'][:]
        tumor_PC_displacement = hf['tumor_PC_displacement'][:]
        querry_PC = hf['querry_PC'][:]
        edma_PC = hf['brain_PC'][:]
        querry_PC_displacement = hf['querry_PC_displacement'][:]
        affine_ALL = hf['affine_matrix'][:]
        hf.close()
        if is_debug:
            num =100
            tumor_PC = tumor_PC
            tumor_PC_displacement = tumor_PC_displacement
            querry_PC = querry_PC
            querry_PC_displacement = querry_PC_displacement
            affine_ALL = affine_ALL
            indices = np.arange(100) + 331
        else:
            if is_train:
                num = self.num_train_files
                tumor_PC = tumor_PC[-num:]
                edma_PC = edma_PC[-num:]
                tumor_PC_displacement = tumor_PC_displacement[-num:]
                querry_PC = querry_PC[-num:]
                querry_PC_displacement = querry_PC_displacement[-num:]
                affine_ALL = affine_ALL[-num:]
                if shuffle:
                    indices = np.random.permutation(num)
                else:
                    indices = np.arange(num)
            else:
                num = self.num_val_files
                tumor_PC = tumor_PC[:num]
                edma_PC = edma_PC[:num]
                tumor_PC_displacement = tumor_PC_displacement[:num]
                querry_PC = querry_PC[:num]
                querry_PC_displacement = querry_PC_displacement[:num]
                affine_ALL = affine_ALL[:num]
                if shuffle:
                    indices = np.random.permutation(num)
                else:
                    indices = np.arange(num)

        output_pointcloud = np.zeros(shape=(self.cfg['training']['batch_size'], self.cfg['training']['num_points'], 3), dtype=np.float32)
        output_point_disp = np.zeros(shape=(self.cfg['training']['batch_size'], self.cfg['training']['num_points'], 3), dtype=np.float32)
        output_extend = np.zeros(shape=(self.cfg['training']['batch_size'], 3), dtype=np.float32)
        output_querrypoint = np.zeros(shape=(self.cfg['training']['batch_size'], self.cfg['training']['num_sample_points'], 3), dtype=np.float32)
        output_querry_disp = np.zeros(shape=(self.cfg['training']['batch_size'], self.cfg['training']['num_sample_points'], 3), dtype=np.float32)
        output_affine = np.zeros(shape=(self.cfg['training']['batch_size'], 4, 4), dtype=np.float32)
        filled = 0
        for i in range(num):
            pointcloud = tumor_PC[indices[i]]
            edma_pointcloud = edma_PC[indices[i]]
            point_disp = tumor_PC_displacement[indices[i]]
            querrypoint = querry_PC[indices[i]]
            #querrypoint = np.concatenate([querrypoint[:1000, :], querrypoint[:1000, :]],axis=0)
            querry_disp = querry_PC_displacement[indices[i]]
            #querry_disp = np.concatenate([querry_disp[:1000, :], querry_disp[:1000, :]], axis=0)

            p1 = np.random.permutation(pointcloud.shape[0])[:self.cfg['training']['num_points']]
            output_pointcloud[filled, :int(self.cfg['training']['num_points']/2), :] = pointcloud[p1[:int(self.cfg['training']['num_points']/2)], :]
            output_pointcloud[filled, int(self.cfg['training']['num_points'] / 2):, :] = edma_pointcloud[p1[int(
                        self.cfg['training']['num_points'] / 2):], :]
            output_point_disp[filled, :, :] = point_disp[p1]
            output_extend[filled, :] = np.asarray([np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]),
                                                   np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
                                                   np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
            p2 = np.random.permutation(querrypoint.shape[0])[:self.cfg['training']['num_sample_points']]
            output_querrypoint[filled, :, :] = querrypoint[p2, :]
            output_querry_disp[filled, :, :] = querry_disp[p2, :]

            output_affine[filled, :, :] = affine_ALL[indices[i]]

            filled += 1
            if filled == BATCH_SIZE:
                filled = 0
                #with h5py.File('/tmp/Liver_Phantom_data/src/sparse_dataset/debug.h5', 'w') as f:
                #    f.create_dataset('output_pointcloud', data=output_pointcloud)
                #if is_train:
                #    output_pointcloud += np.random.normal(0.0, 2e-3, size=output_pointcloud.shape)#1e-3, 2e-3
                if get_affine:
                    yield output_pointcloud, output_point_disp, output_querrypoint, output_querry_disp, \
                          np.concatenate([1 * np.ones(shape=(self.cfg['training']['batch_size'], int(self.cfg['training']['num_points'] / 2)), dtype=np.float32),
                                          2 * np.ones(shape=(self.cfg['training']['batch_size'], int(self.cfg['training']['num_points'] / 2)), dtype=np.float32)], axis=1).astype(np.uint8), \
                          output_extend, \
                          indices[i-BATCH_SIZE+1 : i+1], output_affine
                else:
                    yield output_pointcloud, output_point_disp, output_querrypoint, output_querry_disp, \
                          np.concatenate([1 * np.ones(shape=(self.cfg['training']['batch_size'], int(self.cfg['training']['num_points'] / 2)), dtype=np.float32),
                                          2 * np.ones(shape=(self.cfg['training']['batch_size'], int(self.cfg['training']['num_points'] / 2)), dtype=np.float32)], axis=1).astype(np.uint8), \
                          output_extend, \
                          indices[i-BATCH_SIZE+1 : i+1]

def parse_points(points_on, points_part_label):
    index_front = np.where(points_part_label == 1)
    points_on_front = points_on[index_front[0], :]
    index_back = np.where(points_part_label == 0)
    points_on_back = points_on[index_back[0], :]
    index_FF = np.where(points_part_label == 2)
    points_on_FF = points_on[index_FF[0], :]
    index_LR = np.where(points_part_label == 3)
    points_on_LR = points_on[index_LR[0], :]
    index_RR = np.where(points_part_label == 4)
    points_on_RR = points_on[index_RR[0], :]
    return points_on_front, points_on_back, points_on_FF, points_on_LR, points_on_RR


def angle2rotmatrix(angle):
    r1 = np.asarray([[np.cos(angle[2]), -np.sin(angle[2]), 0],
                     [np.sin(angle[2]), np.cos(angle[2]), 0],
                     [0, 0, 1]])
    r2 = np.asarray([[np.cos(angle[1]), 0, np.sin(angle[1])],
                     [0.0, 1.0, 0.0],
                     [-np.sin(angle[1]), 0, np.cos(angle[1])]])
    r3 = np.asarray([[1, 0, 0],
                     [0, np.cos(angle[0]), -np.sin(angle[0])],
                     [0, np.sin(angle[0]), np.cos(angle[0])]])
    return r1, r2, r3, np.matmul(r1, np.matmul(r2, r3))

def scale_rotation(input_on, input_all, size, provide_transformation=False):
    # shift and scale
    max = np.max(input_on, axis=0)
    min = np.min(input_on, axis=0)
    center = (max + min)/2
    # rotation
    r1, r2, r3, rotation_matrix = angle2rotmatrix(np.random.uniform(0, (5/360)*2*np.pi, size=3))
    output_on = np.matmul(rotation_matrix, (input_on-center).T).T
    output_all = np.matmul(rotation_matrix, (input_all-center).T).T

    max = np.max(output_on, axis=0)
    min = np.min(output_on, axis=0)
    center2 = (max + min)/2
    scale = 4.0#size / np.max(max - center2)
    output_on = scale * (output_on)
    output_all = scale * (output_all)

    if provide_transformation:
        return output_on, output_all, ((scale)*rotation_matrix, center)
    else:
        return output_on, output_all


def load_data(h5f):
    points_on = h5f['points_on'][:]
    p1 = np.random.permutation(points_on.shape[0])
    points_on = points_on[p1,:]
    points_part_label = h5f['points_part_label'][:]
    points_part_label = points_part_label[p1]

    points_around = h5f['points_around'][:]
    p2 = np.random.permutation(points_around.shape[0])
    points_around = points_around[p2, :]
    around_inout = h5f['around_inout'][:]
    around_inout = around_inout[p2]

    points_uniform = h5f['points_uniform'][:]
    p3 = np.random.permutation(points_around.shape[0])
    points_uniform = points_uniform[p3, :]
    uniform_inout = h5f['uniform_inout'][:]
    uniform_inout = uniform_inout[p3]
    return points_on, points_part_label, points_around, around_inout, points_uniform, uniform_inout

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
if __name__ == '__main__':
    import yaml
    from mayavi import mlab
    with open('../confs/debug.yaml', 'r') as f:
        cfg = yaml.load(f)

    pv = ProbeProvider(cfg)
    train_generator = pv.provide_data(False, 4, 1500, 1500, shuffle=False)
    for data in train_generator:
        querry_PC = [data[2], data[3]]
        mlab.points3d(querry_PC[0][0, :, 0], querry_PC[0][0, :, 1], querry_PC[0][0, :, 2], scale_factor=0.002)
        mlab.quiver3d(querry_PC[0][0, :, 0], querry_PC[0][0, :, 1], querry_PC[0][0, :, 2],
                      querry_PC[1][0, :, 0], querry_PC[1][0, :, 1], querry_PC[1][0, :, 2], scale_factor=0.1)
        mlab.show()

        print('done')