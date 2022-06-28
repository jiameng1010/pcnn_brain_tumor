
import os
import sys
import numpy as np
import h5py
from mayavi import mlab
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.interactive(True)

class Extention_Visulator():
    def __init__(self, filename, index):
        f = h5py.File(filename)
        self.data = f['data'][:][index, 300:(300+256), :]
        self.normal = f['normal'][:][index, 300:(300+256), :]
        #self.label = f['label'][:][index, :, :]
        self.shape = 70
        self.shapeh = int(self.shape/2)

        self.sigma = 0.06

        self.extention = np.empty(shape=(self.shape, self.shape, self.shape), dtype=np.float32)

    def generate_extention(self, type):
        if type == 1:
            coordinate = np.empty(shape=(self.shape, self.shape, self.shape, 3), dtype=np.float32)
            #coordinate[:, 0] = range(-1.5, 1.5, 0.03)
            #coordinate[:, 1] = range(-1.5, 1.5, 0.03)
            #coordinate[:, 2] = range(-1.5, 1.5, 0.03)
            for i in range(-self.shapeh, self.shapeh):
                for j in range(-self.shapeh, self.shapeh):
                    for k in range(-self.shapeh, self.shapeh):
                        coordinate[i+self.shapeh, j+self.shapeh, k+self.shapeh, :]  = np.asarray([i*2.0/self.shapeh, j*2.0/self.shapeh, k*2.0/self.shapeh])
            coordinate = np.reshape(coordinate, newshape=(-1, 3))
            #self.data = np.asarray([[0.0, 0.0, 0.0]])
            #self.normal = np.asarray([[1.0, 0.0, 0.0]])
            difference = np.expand_dims(coordinate, axis=1) - np.expand_dims(self.data, axis=0)
            dp = np.sum(difference * np.expand_dims(self.normal, axis=0), axis = 2)
            dc2 = np.sum(difference * difference, axis=2) - dp*dp
            #extention = np.exp(-(dc2 + dp * dp) / (self.sigma ** 2))
            extention = np.exp(-(0.3*dc2 + 1.7*dp * dp) / (self.sigma ** 2))

            #extention = (np.clip(dp, 0, 100)/self.sigma)*np.exp(-(dc2 + dp*dp)/ (self.sigma**2))

            #extention = np.exp(-(dc2 + dp * dp) / (self.sigma ** 2))
            #extention = extention - 3 * np.exp(-(dc2 + dp * dp) / (0.5*self.sigma ** 2))

            #extention = np.exp(-(dc2 + dp*dp) / (self.sigma**2)) - 1.0*(dp*dp/self.sigma**2) * np.exp(-(dc2 + dp*dp) / (1.0*self.sigma**2))

            #extention = np.abs(extention)
            #extention = np.clip(extention, 0, 100)
            extention = np.sin(0.06*dp) * extention
            extention = np.sum(extention, axis=1)
            #\extention = np.clip(extention, 0, 100)
            self.extention = np.reshape(extention, newshape=(self.shape, self.shape, self.shape))
            print('d')


    def plot(self):
        mlab.pipeline.volume(mlab.pipeline.scalar_field(self.extention))
        mlab.show()
        mlab.contour3d(self.extention)
        mlab.show()

    def plot_pc(self):
        mlab.points3d(self.data[:,0], self.data[:,1], self.data[:,2])
        mlab.show()



ev = Extention_Visulator('/home/mjia/Documents/jointSegmentation/pcnn/data/modelnet40_ply_hdf5_2048/ply_data_train1.h5', 2)
ev.generate_extention(type=1)
ev.plot_pc()
ev.plot()
print ('d')