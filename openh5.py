import h5py
filename = '/home/mjia/Documents/jointSegmentation/pcnn/data/modelnet40_ply_hdf5_2048/ply_data_train0.h5'
f = h5py.File(filename, 'r')
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])