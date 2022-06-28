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


MODEL_NET_DIR = '/media/mjia/Documents/ShapeCompletion/ModelNet40'

def generate(mesh, save_filename):
    points_on = np.empty((8192, 3))
    complete_sample, faces_id = sample.sample_surface(mesh, 8192)

    points_on = copy.deepcopy(complete_sample)
    points_normal = mesh.face_normals[faces_id]

    complete_sample[0:6144, :] = complete_sample[0:6144, :] + np.random.normal(scale=(0.1 * mesh.extents),size=(6144, 3))
    complete_sample[6144:, :] = complete_sample[6144:, :] + np.random.normal(scale=(0.02 * mesh.extents), size=(2048, 3))
    contained = mesh.contains(complete_sample)

    points_on, complete_sample = re_scale(points_on, complete_sample, mesh.bounding_box.vertices)
    #check(mesh, points_on, points_normal, complete_sample, contained)
    #
    h5f = h5py.File(save_filename, 'w')
    h5f.create_dataset('points_on', data=points_on)
    h5f.create_dataset('points_normal', data=points_normal)
    h5f.create_dataset('points_in_out', data=complete_sample)
    h5f.create_dataset('in_out_lable', data=contained)
    h5f.close()
    print('d')
    return np.sum(contained)<256, np.sum(contained)<512, np.sum(contained)<1024


def re_scale(points_on, complete_sample, bounding_box):
    mean = np.expand_dims(np.mean(bounding_box, axis=0), axis=0)
    max = np.max(bounding_box - mean)
    points_on = (points_on - mean) / max
    complete_sample = (complete_sample - mean) / max
    return points_on, complete_sample


def check(mesh, points_on, points_normal, complete_sample, contained):
    mesh.show()
    plot_normal(points_on, points_normal)

    n = np.sum(contained)
    total = contained.shape[0]
    index = np.argsort(contained)
    index_of_out = index[0:total - n]
    index_of_in = index[total - n:]
    plot_pc(complete_sample[index_of_in, :])

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
    for root, dirs, files in os.walk(MODEL_NET_DIR):
        for i in files:
            if i[-3:] == 'off':
                off_file_name = root + '/' + i
                save_file_name = off_file_name[:-3] + 'h5'
                mesh = trimesh.load(off_file_name)
                l256, l512, l1024 = generate(mesh, save_file_name)
                if l256:
                    L256 += 1
                if l512:
                    L512 += 1
                if l1024:
                    L1024 += 1
                num += 1
                if num > 200:
                    break
                print(off_file_name)

    print(L256)
    print(L512)
    print(L1024)

def generate_txt():
    f1 = open(MODEL_NET_DIR+'/train.txt', 'w')
    f2 = open(MODEL_NET_DIR+'/test.txt', 'w')
    for root, dirs, files in os.walk(MODEL_NET_DIR):
        for i in files:
            if i[-2:] == 'h5':
                h5_filename = root + '/' + i
                if h5_filename.split('/')[-2] == 'train':
                    f1.write(h5_filename+'\n')
                if h5_filename.split('/')[-2] == 'test':
                    f2.write(h5_filename+'\n')
    f1.close()
    f2.close()



main()
#generate_txt()