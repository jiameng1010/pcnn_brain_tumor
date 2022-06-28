
import tensorflow as tf
import numpy as np
import os
import sys
UPPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(UPPER_DIR + '/layers')
from convlayer_elements import ConvElements, DeConvElements
import tf_util
from pooling import PoolingLayer
from convolution_layer import ConvLayer
from deconvolution_layer import DeconvLayer
from pyhocon import ConfigFactory


def get_probepoints(batch_size, num):
    grid = tf.linspace(0.0, 1.0, tf.constant(num))

    return tf.tile(tf.reshape(tf.transpose(tf.reshape(tf.stack(tf.meshgrid(grid, grid, grid)), [3, -1])),
                              [1, num**3, 3]), [batch_size, 1, 1])

class ProbeNetwork:
    def __init__(self, conf):
        self.conf = conf

    def get_network_model(self, pointclouds_pl,probepoints, pointlabel_ph,\
		batch_size, num_point, weight_decay,is_training, bn_decay=None):
        with_bn = self.conf.get_bool('with_bn')
        batch_size = pointclouds_pl.get_shape()[0].value
        num_point = pointclouds_pl.get_shape()[1].value
        num_probe = probepoints.get_shape()[1].value
        #ps_function_pl = tf.concat([pointclouds_pl, tf.expand_dims(pointlabel_ph, axis=2), tf.ones(shape=[batch_size, num_point, 1], dtype=tf.float32)],
        #                           axis=2)
        ps_function_pl = tf.expand_dims(pointlabel_ph, axis=2)

        pool_sizes_sigma = self.conf.get_list('pool_sizes_sigma')  # [256,64,4]
        spacing = np.float32(self.conf.get_float('kernel_spacing'))

        # first scale
        network = ps_function_pl

        input_channel = 1

        blocks = self.conf.get_list('blocks_out_channels')
        pointclouds_pl_down = []
        probe = []

        for block_index, block in enumerate(blocks):
            block_elm = ConvElements(probepoints, pointclouds_pl, np.float32(0.3)/np.sqrt(pointclouds_pl.get_shape()[1].value,dtype=np.float32), spacing,
                                          np.float32(self.conf.get_float('kernel_sigma_factor')))
            for out_index, out_channel in enumerate(block):
                convlayer = ConvLayer(input_channel, block_elm, out_channel,
                                     '{0}_block_{1}'.format(block_index, out_index), is_training)

                network, network_probe_real = convlayer.get_layer(
                    probepoints, network, with_bn, bn_decay,False,True)
                probe.append(network_probe_real)

                input_channel = out_channel


            pointclouds_pl_down.append([pointclouds_pl, network])
            #pointclouds_pl_down.append([pointclouds_pl, network])
            #probe.append(network_probe)
            pointclouds_pl, network = PoolingLayer(block_elm, out_channel, out_channel,
                                                   int(pool_sizes_sigma[block_index + 1][0])).get_layer(network,use_fps = tf.constant(False),is_subsampling = self.conf.get_bool("is_subsampling"))


        #pointclouds_pl_down.append([pointclouds_pl,None])
        pointclouds_pl_down.append([pointclouds_pl, None])


        #one_hot_label_expand = tf.tile(tf.expand_dims(input_label, axis=1),[1,pointclouds_pl_down[-1][0].get_shape()[1].value,1])
        #network = tf.concat(axis=2, values=[network, one_hot_label_expand])

        for i in range(len(pointclouds_pl_down) -1 ,0,-1):
            if i == 3:
                deconvnet = DeconvLayer(pointclouds_pl_down[i][0],tf.concat([network], axis=1),pointclouds_pl_down[i-1][0]).get_layer()
            else:
                deconvnet = DeconvLayer(pointclouds_pl_down[i][0],network,pointclouds_pl_down[i-1][0]).get_layer()

            block_elm = ConvElements(probepoints, pointclouds_pl_down[i - 1][0],
                              0.3*tf.reciprocal(tf.sqrt(1.0 * pointclouds_pl_down[i - 1][0].get_shape()[1].value)), spacing,
                              self.conf.get_float('kernel_sigma_factor'))

            convnet, network_probe = ConvLayer(deconvnet.get_shape()[-1].value, block_elm, pointclouds_pl_down[i-1][1].get_shape()[-1].value, "{0}_deconv".format(i),
                           is_training).get_layer(probepoints,deconvnet, with_bn , bn_decay, False,True)


            network = convnet
            probe.append(network_probe)

            if (self.conf.get_bool('is_dropout') and i  <= 2):
                network = tf_util.dropout(network,is_training,"dropout_{0}".format(i),keep_prob = self.conf.get_float('dropout.keep_prob'))

        #block_elm = ConvElements(probepoints, pointclouds_pl_down[0][0],
        #                         tf.reciprocal(tf.sqrt(1.0 * pointclouds_pl_down[0][0].get_shape()[1].value)),
        #                              spacing,
        #                              self.conf.get_float('kernel_sigma_factor')) \

        #_, network_probe = ConvLayer(network_probe.get_shape()[-1].value, block_elm,
        #                         2, "{0}_deconv".format('last'),
        #                   is_training).get_layer(probepoints,network_probe, False, bn_decay,False ,False)
        network_probe_P = tf.concat(probe, axis=2)
        '''network_probe_P = tf_util.fully_connected(tf.reshape(network_probe_P, [batch_size*num_probe, -1]),
                                                 128,
                                                 'output_layer',
                                                 bn=False,
                                                 is_training=is_training)
        network_probe_P = tf.reshape(network_probe_P, [batch_size, num_probe, 128])'''
        #network_probe_P = tf.layers.conv1d(network_probe_P, 128, 1)
        network_probe = tf.layers.conv1d(network_probe_P, 3, 1, use_bias=False)

        return network_probe, network_probe_P

    def get_loss_displacement(self, disp_pred, disp):
        error = disp_pred - disp
        sq_error = error * error
        per_point_loss = tf.reduce_sum(sq_error, axis=2)
        per_instance_loss = tf.reduce_mean(per_point_loss, axis=1)
        total_loss = tf.reduce_mean(per_instance_loss)
        return per_point_loss, per_instance_loss, total_loss

    def get_loss_displacement_abs(self, disp_pred, disp):
        loss = tf.keras.losses.MeanSquaredError()
        return loss(disp_pred, disp)
        '''error = disp_pred - disp
        abs_error = tf.abs(error)
        per_point_loss = tf.reduce_sum(abs_error, axis=2)
        per_instance_loss = tf.reduce_mean(per_point_loss, axis=1)
        total_loss = tf.reduce_mean(per_instance_loss)
        return per_point_loss, per_instance_loss, total_loss'''

    def get_loss(self, seg_pred, seg):

        # size of seg_pred is batch_size x point_num x part_cat_num
        # size of seg is batch_size x point_num
        per_instance_seg_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
        seg_loss = tf.reduce_hgmean(per_instance_seg_loss)

        per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

        total_loss =  seg_loss

        return total_loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res

    def get_distance_loss(self, seg_pred, seg):
        return tf.losses.mean_squared_error(tf.expand_dims(seg, axis=-1), seg_pred)

    def get_normal_loss(self, normal_pred, normal):
        # a.b/|a|.|b|
        #ab = tf.reduce_sum(normal * normal_pred, axis=-1)
        #aabb = tf.sqrt(tf.reduce_sum(normal_pred * normal_pred, axis=-1)) * tf.sqrt(tf.reduce_sum(normal * normal, axis=-1))
        #return -tf.reduce_mean(ab/aabb)
        norm = tf.sqrt(tf.reduce_sum(normal_pred * normal_pred, axis=-1))
        normal_pred_norm = normal_pred / tf.expand_dims(norm, axis=2)
        return tf.losses.cosine_distance(normal, normal_pred_norm, axis=-1), tf.losses.cosine_distance(normal, normal_pred_norm, axis=-1)

