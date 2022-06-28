import json
import os
import h5py
import numpy as np

with open('/media/mjia/Data/ShapeNetSegmentation/shapenetcore_partanno_segmentation_benchmark_v0/train_test_split/shuffled_test_file_list.json') as f:
    testfiles = json.load(f)
    ids = []
    for i in testfiles:
        ids.append(i.split('/')[-1])

DATA_DIR = '/media/mjia/Data/ShapeNetSegmentation/shapenetcore_partanno_segmentation_benchmark_v0'

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

classname = {'02691156': [0, 1, 2, 3],
'02773838':[4, 5],
'02954340':[6, 7],
'02958343':[8, 9, 10, 11],
'03001627':[12, 13, 14, 15],
'03261776':[16, 17, 18],
'03467517':[19, 20, 21],
'03624134':[22, 23],
'03636649':[24, 25, 26, 27],
'03642806':[28, 29],
'03790512':[30, 31, 32, 33, 34, 35],
'03797390':[36, 37],
'03948459':[38, 39, 40],
'04099429':[41, 42, 43],
'04225987':[44, 45, 46],
'04379243':[47, 48, 49]}

def re_scale(points_on, bounding_box):
    mean = np.expand_dims(np.mean(bounding_box, axis=0), axis=0)
    max = np.max(bounding_box - mean)
    points_on = (points_on - mean) / max
    return points_on

def getlabel(point_cloud, label_list):
    points_inout = np.concatenate([point_cloud, point_cloud, point_cloud, point_cloud])\
                + np.random.normal(scale = 0.05, size=[4*point_cloud.shape[0], 3])

    distance = np.expand_dims(point_cloud, axis=1)\
               - np.expand_dims(points_inout, axis=0)
    distance = np.sqrt(np.sum(distance * distance, axis=2))
    closest_id = np.argmin(distance, axis=0)
    label_list4 = np.asarray(label_list)
    labels_on = label_list4[closest_id]
    return points_inout, labels_on


def generate_h5(labelf, pointf, savef):
    class_id = labelf.split('/')[-3]
    map_dict = classname[class_id]

    label_text = open(labelf)
    point_text = open(pointf)
    point_list = []
    label_list = []
    for point in point_text.readlines():
        label = label_text.readline()
        point_list.append([float(point.split(' ')[0]), float(point.split(' ')[1]), float(point.split(' ')[2])])
        label_list.append(map_dict[int(label)-1])

    point_cloud = np.asarray(point_list)
    maxs = np.max(point_cloud, axis=0)
    mins = np.min(point_cloud, axis=0)
    bounding_box = [[maxs[0], maxs[1], maxs[2]],
                    [maxs[0], maxs[1], mins[2]],
                    [maxs[0], mins[1], maxs[2]],
                    [mins[0], maxs[1], maxs[2]],
                    [mins[0], maxs[1], mins[2]],
                    [mins[0], mins[1], maxs[2]],
                    [maxs[0], mins[1], mins[2]],
                    [mins[0], mins[1], mins[2]]]
    point_cloud = re_scale(point_cloud, bounding_box)

    complete_sample, points_in_label = getlabel(point_cloud, label_list)

    h5f = h5py.File(savef, 'w')
    h5f.create_dataset('points_on', data=point_cloud)
    h5f.create_dataset('points_in_out', data=complete_sample)
    h5f.create_dataset('seg_lable_on', data=np.asarray(label_list))
    h5f.create_dataset('seg_lable_inout', data=points_in_label)
    h5f.close()
    return

def main():
    for root, dirs, files in os.walk(DATA_DIR):
        for i in files:
            if i[-3:] == 'seg':
                id = i.split('/')[-1][:-4]

                if id in ids:
                    seg_file_name = root + '/' + i
                    point_fname = root[:-6] + '/' + id + '.pts'
                    h5_fname = point_fname[:-3] + 'h5'
                    #if os.path.isfile(h5_fname):
                    #    continue
                    generate_h5(seg_file_name, point_fname, h5_fname)
                    print(seg_file_name)
                    print('done')

def generate_txt():
    #f1 = open(DATA_DIR + '/train.txt', 'w')
    f2 = open(DATA_DIR + '/test.txt', 'w')
    for root, dirs, files in os.walk(DATA_DIR):
        for i in files:
            if i[-2:] == 'h5':
                h5_filename = root + '/' + i
                id = h5_filename.split('/')[-1][:-3]
                #if h5_filename.split('/')[-2] == 'train':
                #    f1.write(h5_filename + '\n')
                #if h5_filename.split('/')[-2] == 'test':
                #    f2.write(h5_filename + '\n')
                if id in ids:
                    f2.write(h5_filename + '\n')
    #f1.close()
    f2.close()

#main()
#generate_txt()