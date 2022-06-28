import numpy as np
import os
import h5py
import trimesh
from mayavi import mlab
#import pymesh
import copy
from trimesh import sample, ray, triangles
from trimesh.ray.ray_triangle import RayMeshIntersector, ray_triangle_id
import pandas as pd
import gc
import subprocess
import open3d

label_name = ['Airplane_labelA', 'Airplane_labelB', 'Airplane_labelC', 'Airplane_labelD',\
              'Bag_labelA', 'Bag_labelB',\
              'Cap_labelA', 'Cap_labelB',\
              'Car_labelA', 'Car_labelB', 'Car_labelC', 'Car_labelD',\
              'Chair_labelA', 'Chair_labelB', 'Chair_labelC', 'Chair_labelD',\
              'Earphone_labelA', 'Earphone_labelB', 'Earphone_labelC',\
              'Guitar_labelA', 'Guitar_labelB', 'Guitar_labelC',\
              'Knife_labelA', 'Knife_labelB',\
              'Lamp_labelA', 'Lamp_labelB', 'Lamp_labelC', 'Lamp_labelD',\
              'Laptop_labelA', 'Laptop_labelB',\
              'Motorbike_labelA', 'Motorbike_labelB', 'Motorbike_labelC', 'Motorbike_labelD', 'Motorbike_labelE', 'Motorbike_labelF',\
              'Mug_labelA', 'Mug_labelB',\
              'Pistol_labelA', 'Pistol_labelB', 'Pistol_labelC',\
              'Rocket_labelA', 'Rocket_labelB', 'Rocket_labelC',\
              'Skateboard_labelA', 'Skateboard_labelB', 'Skateboard_labelC',\
              'Table_labelA', 'Table_labelB', 'Table_labelC']

random_color = np.random.uniform(size=(50, 3))
SHAPENET_DIR = '/media/mjia/Data/ShapeNetSegmentation/labeled_meshes/SHAPENET_MESHES'
#SHAPENET_DIR = '/media/mjia/Data/Data/labeled_meshes/SHAPENET_MESHES/Mug'

def getlabel_on(mesh, label_txt, faces_id, faces_id_in):
    f = open(label_txt)
    steps = 0

    mesh_belong = np.zeros(mesh.faces.shape[0])
    for i in f.readlines():
        if steps%3 == 0:
            label = i[:-1]
            if label in label_name:
                label = label_name.index(label)
            else:
                raise label + 'is not in'
        elif steps%3 == 1:
            faces = i.split(' ')
            faces = np.asarray(faces)
            faces = faces.astype(np.int) - 1
            mesh_belong[faces] = int(label)
        else:
            steps += 1
            continue
        steps += 1

    mesh_belong = mesh_belong.astype(np.int)

    labels_on = mesh_belong[faces_id]
    labels_in = mesh_belong[faces_id_in]
    return labels_on, labels_in

def generate(mesh, save_filename, label_txt):
    points_on = np.empty((8192, 3))
    complete_sample, faces_id = sample.sample_surface(mesh, 8192)

    points_on = copy.deepcopy(complete_sample)
    points_normal = mesh.face_normals[faces_id]

    complete_sample[0:6144, :] = complete_sample[0:6144, :] + np.random.normal(scale=(0.1 * mesh.extents),size=(6144, 3))
    complete_sample[6144:, :] = complete_sample[6144:, :] + np.random.normal(scale=(0.02 * mesh.extents), size=(2048, 3))
    contained = mesh.contains(complete_sample)

    _, distance, faces_id_in = trimesh.proximity.closest_point(mesh, complete_sample)
    points_on_label, points_in_label = getlabel_on(mesh, label_txt, faces_id, faces_id_in)

    points_on, complete_sample = re_scale(points_on, complete_sample, mesh.bounding_box.vertices)
    #
    print(np.sum(contained))
    check(mesh, points_on, points_normal, complete_sample, contained, points_on_label, points_in_label)
    #
    '''h5f = h5py.File(save_filename, 'w')
    h5f.create_dataset('points_on', data=points_on)
    h5f.create_dataset('points_normal', data=points_normal)
    h5f.create_dataset('points_in_out', data=complete_sample)
    h5f.create_dataset('in_out_lable', data=contained)
    h5f.create_dataset('seg_lable_on', data=points_on_label)
    h5f.create_dataset('seg_lable_inout', data=points_in_label)
    h5f.close()'''
    print('d')
    return np.sum(contained)<256, np.sum(contained)<512, np.sum(contained)<1024


def re_scale(points_on, complete_sample, bounding_box):
    mean = np.expand_dims(np.mean(bounding_box, axis=0), axis=0)
    max = np.max(bounding_box - mean)
    points_on = (points_on - mean) / max
    complete_sample = (complete_sample - mean) / max
    return points_on, complete_sample


def check(mesh, points_on, points_normal, complete_sample, contained, points_on_label, points_in_label):
    mesh.show()
    normal_indicate_labels = random_color[points_on_label]
    plot_normal(points_on, normal_indicate_labels)

    n = np.sum(contained)
    print(n)
    total = contained.shape[0]
    index = np.argsort(contained)
    index_of_out = index[0:total - n]
    index_of_in = index[total - n:]
    normal_indicate_labels = random_color[points_in_label[index_of_in]]
    plot_normal(complete_sample[index_of_in, :], normal_indicate_labels)

def plot_normal(probe_points, normals):
    mlab.points3d(probe_points[:512,0], probe_points[:512,1], probe_points[:512,2], scale_factor=0.05)
    mlab.quiver3d(probe_points[:512,0], probe_points[:512,1], probe_points[:512,2],
                  normals[:512, 0], normals[:512, 1], normals[:512, 2], scale_factor=0.2)
    mlab.show()

def plot_pc(pc):
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], scale_factor=0.05)
    mlab.show()

def main():
    num = 0
    L256 = 0
    L512 = 0
    L1024 = 0
    for root, dirs, files in os.walk(SHAPENET_DIR):
        for i in files:
            if i[-3:] == 'off':
                if root.split('/')[-1] != 'Motorbike':
                    continue
                off_file_name = root + '/' + i
                label_txt = off_file_name[:-4] + '_labels.txt'
                save_file_name = off_file_name[:-3] + 'h5'
                mesh = trimesh.load(off_file_name)
                l256, l512, l1024 = generate(mesh, save_file_name, label_txt)
                if l256:
                    L256 += 1
                if l512:
                    L512 += 1
                if l1024:
                    L1024 += 1
                num += 1
                print(off_file_name)

    print(L256)
    print(L512)
    print(L1024)


def get_test_list():
    f = open('testing_ply_file_list.txt')
    ids = []
    for fname in f.readlines():
        id = fname.split(' ')[0].split('/')[-1][:-4]
        ids.append(id)
    return ids



def generate_txt():
    ids = get_test_list()
    f1 = open(SHAPENET_DIR + '/train.txt', 'w')
    f2 = open(SHAPENET_DIR + '/test.txt', 'w')
    for root, dirs, files in os.walk(SHAPENET_DIR):
        for i in files:
            if i[-2:] == 'h5':
                h5_filename = root + '/' + i
                id = h5_filename.split('/')[-1][:-3]
                #if h5_filename.split('/')[-2] == 'train':
                #    f1.write(h5_filename + '\n')
                #if h5_filename.split('/')[-2] == 'test':
                #    f2.write(h5_filename + '\n')
                if id not in ids:
                #if np.random.uniform() < 0.5:
                    f1.write(h5_filename + '\n')
                else:
                    f2.write(h5_filename + '\n')
    f1.close()
    f2.close()

main()
#generate_txt()