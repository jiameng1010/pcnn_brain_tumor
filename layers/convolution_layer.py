import tensorflow as tf
import sys
sys.path.append('../')
import tf_util

class ConvLayer:
    def __init__(self, input_channels, block_elemnts, out_channels,scope,is_training,use_xavier=True,l2_regularizer = 1.0e-3):
        self.input_channels = input_channels
        self.block_elemnts = block_elemnts
        self.out_channels = out_channels
        self.scope = scope
        self.use_xavier = use_xavier
        self.is_training = is_training
        self.weight_decay = 0.0
        self.l2_regularizer = l2_regularizer

        with tf.variable_scope(self.scope, tf.AUTO_REUSE) as sc:

            self.k_tensor = tf_util._variable_with_weight_decay('weights',
                                                           shape=[self.out_channels, self.input_channels, self.block_elemnts.num_of_translations],
                                                           use_xavier=self.use_xavier,
                                                           stddev=0.1, wd=self.weight_decay)

    def get_convlution_operator(self, probe_points, functions_pl,interpolation,dtype=tf.float32):
        translations = self.block_elemnts.kernel_translations

        distances = self.block_elemnts.get_distance_matrix()
        distances_probe = tf.expand_dims(probe_points, axis=2) - tf.expand_dims(self.block_elemnts.points_pl, axis=1)
        distances_probe = tf.reduce_sum(tf.multiply(distances_probe, distances_probe), axis=3)
        points_translations_dot = tf.matmul(self.block_elemnts.points_pl, tf.transpose(translations, [0, 2, 1]))
        points_translations_dot_probe = tf.matmul(probe_points, tf.transpose(translations, [0, 2, 1]))
        translations_square = tf.reduce_sum(translations * translations, axis=2)

        # Find weights w of the extension operator. result dimensions are BATCH_SIZE x NUM_OF_POINTS x  POINT_CLOUD_FUNCTION_DIM
        if (interpolation):
            w_tensor = tf.matrix_solve_ls(self.block_elemnts.get_interpolation_matrix(),
                                          functions_pl,l2_regularizer=self.l2_regularizer)
        else:
            w_tensor = tf.multiply(
                tf.expand_dims(tf.reciprocal(tf.reduce_sum(self.block_elemnts.get_interpolation_matrix(), axis=2)),
                               axis=2), functions_pl)

        # Calculate the product of w_tensor and the kernel weights. Result dimensions are TRANSLATIONS x BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        b_tensor = tf.transpose(tf.matmul(tf.tile(tf.expand_dims(w_tensor, axis=1), [1, self.out_channels, 1, 1]),
                                          tf.tile(tf.expand_dims(self.k_tensor, axis=0),
                                                  [self.block_elemnts.batch_size, 1, 1, 1])),
                                [3, 0, 2, 1])

        def convopeator_per_translation(input):
            translation_index = input[1]
            b_per_translation = input[0]

            dot = tf.tile(-2 * tf.slice(points_translations_dot, [0, 0, translation_index],
                                        [self.block_elemnts.batch_size, self.block_elemnts.num_of_points, 1]),
                          [1, 1, self.block_elemnts.num_of_points])

            q_tensor = tf.exp(-(tf.add(tf.add(distances,
                                              tf.add(dot, -tf.transpose(dot, [0, 2, 1]))),
                                       tf.expand_dims(tf.slice(translations_square, [0, translation_index],
                                                               [self.block_elemnts.batch_size, 1]), axis=2)))
                              / (2 * self.block_elemnts.combined_sigma ** 2))

            return tf.matmul(q_tensor, b_per_translation)

        def convopeator_per_translation_probe(input):
            translation_index = input[1]
            b_per_translation = input[0]

            dot = tf.tile(-2 * tf.slice(points_translations_dot, [0, 0, translation_index],
                                        [self.block_elemnts.batch_size, self.block_elemnts.num_of_points, 1]),
                          [1, 1, distances_probe.shape[1]])
            dot2 = tf.tile(-2 * tf.slice(points_translations_dot_probe, [0, 0, translation_index],
                                        [self.block_elemnts.batch_size, distances_probe.shape[1], 1]),
                          [1, 1, self.block_elemnts.num_of_points])

            q_tensor = tf.exp(-(tf.add(tf.add(distances_probe,
                                              tf.add(dot2, -tf.transpose(dot, [0, 2, 1]))),
                                       tf.expand_dims(tf.slice(translations_square, [0, translation_index],
                                                               [self.block_elemnts.batch_size, 1]), axis=2)))
                              / (2 * (self.block_elemnts.combined_sigma) ** 2))

            return tf.matmul(q_tensor, b_per_translation)



        # Calculate the pcnn convolution operator. Result dimensions are BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        return tf.reduce_sum(tf.map_fn(convopeator_per_translation,
                                              elems=(b_tensor, tf.range(0, self.block_elemnts.num_of_translations)),
                                              dtype=dtype), axis=0), \
               tf.reduce_sum(tf.map_fn(convopeator_per_translation_probe,
                                       elems=(b_tensor, tf.range(0, self.block_elemnts.num_of_translations)),
                                       dtype=dtype), axis=0)

    def get_convlution_operator2(self, probe_points, functions_pl, interpolation, dtype=tf.float32):
        translations = self.block_elemnts.kernel_translations

        distances = self.block_elemnts.get_distance_matrix()
        distances_probe = tf.expand_dims(probe_points, axis=2) - tf.expand_dims(self.block_elemnts.points_pl,
                                                                                axis=1)
        distances_probe = tf.reduce_sum(tf.multiply(distances_probe, distances_probe), axis=3)
        points_translations_dot = tf.matmul(self.block_elemnts.points_pl, tf.transpose(translations, [0, 2, 1]))
        points_translations_dot_probe = tf.matmul(probe_points, tf.transpose(translations, [0, 2, 1]))
        translations_square = tf.reduce_sum(translations * translations, axis=2)

        # Find weights w of the extension operator. result dimensions are BATCH_SIZE x NUM_OF_POINTS x  POINT_CLOUD_FUNCTION_DIM
        if (interpolation):
            w_tensor = tf.matrix_solve_ls(self.block_elemnts.get_interpolation_matrix(),
                                          functions_pl, l2_regularizer=self.l2_regularizer)
        else:
            w_tensor = tf.multiply(
                tf.expand_dims(tf.reciprocal(tf.reduce_sum(self.block_elemnts.get_interpolation_matrix(), axis=2)),
                               axis=2), functions_pl)

        # Calculate the product of w_tensor and the kernel weights. Result dimensions are TRANSLATIONS x BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        b_tensor = tf.transpose(tf.matmul(tf.tile(tf.expand_dims(w_tensor, axis=1), [1, self.out_channels, 1, 1]),
                                          tf.tile(tf.expand_dims(self.k_tensor, axis=0),
                                                  [self.block_elemnts.batch_size, 1, 1, 1])),
                                [3, 0, 2, 1])

        dot = tf.tile(tf.expand_dims(-2 * points_translations_dot, axis=2), [1, 1, self.block_elemnts.num_of_points, 1])
        dot = tf.transpose(dot, [3, 0, 1, 2])  # 27*5*512*512'
        c1 = tf.expand_dims(distances, axis=0)  # 1*5*512*512
        c2 = tf.expand_dims(tf.expand_dims(translations_square, axis=2), axis=3) # 5*27*1*1
        c2 = tf.transpose(c2, [1, 0, 2, 3]) # 27*5*1*1
        q1 = tf.exp(-(tf.add(tf.add(c1,
                                    tf.add(dot, -tf.transpose(dot, [0, 1, 3, 2]))),
                             c2))
                    / tf.expand_dims(tf.expand_dims(tf.expand_dims(2 * self.block_elemnts.combined_sigma ** 2, axis=0), axis=-1), axis=-1))

        dot = tf.tile(tf.expand_dims(-2 * points_translations_dot, axis=2), [1, 1, distances_probe.shape[1], 1])
        dot = tf.transpose(dot, [3, 0, 1, 2])
        dot2 = tf.tile(tf.expand_dims(-2 * points_translations_dot_probe, axis=2), [1, 1, self.block_elemnts.num_of_points, 1])
        dot2 = tf.transpose(dot2, [3, 0, 1, 2])  # 27*5*512*512'
        c21 = tf.expand_dims(distances_probe, axis=0)  # 1*5*512*512
        q2 = tf.exp(-(tf.add(tf.add(c21,
                                    tf.add(dot2, -tf.transpose(dot, [0, 1, 3, 2]))),
                             c2))
                    / tf.expand_dims(tf.expand_dims(tf.expand_dims(2 * self.block_elemnts.combined_sigma ** 2, axis=0), axis=-1), axis=-1))

        return tf.reduce_sum(tf.matmul(q1, b_tensor), axis=0), tf.reduce_sum(tf.matmul(q2, b_tensor), axis=0)


    def get_layer(self, probe_points, functions_pl, with_bn, bn_decay,interpolation,with_Relu = True, with_Sin=True):

        #convlution_operation, convlution_operation2 = self.get_convlution_operator(probe_points, functions_pl,interpolation)
        convlution_operation, convlution_operation2 = self.get_convlution_operator2(probe_points, functions_pl, interpolation)

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE) as sc:
            biases = tf_util._variable_on_cpu('biases', [self.out_channels],
                                              tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(convlution_operation, biases)
            outputs2 = tf.nn.bias_add(convlution_operation2, biases)

            if (with_bn):
                outputs = tf_util.batch_norm_template(outputs, self.is_training, 'bn', [0, 1], bn_decay)
                #outputs2 = tf_util.batch_norm_template(outputs2, self.is_training, 'bn_pb', [0, 1], bn_decay)
            if (with_Relu):
                outputs = tf.nn.relu(outputs)
                outputs2 = tf.nn.relu(outputs2)
            if (with_Sin):
                outputs = tf.math.sin(outputs)
                outputs2 = tf.math.sin(outputs2)
            return outputs, outputs2

    def get_convlution_operator_org(self,functions_pl,interpolation,dtype=tf.float32):
        translations = self.block_elemnts.kernel_translations

        distances = self.block_elemnts.get_distance_matrix()
        points_translations_dot = tf.matmul(self.block_elemnts.points_pl, tf.transpose(translations, [0, 2, 1]))
        translations_square = tf.reduce_sum(translations * translations, axis=2)

        # Find weights w of the extension operator. result dimensions are BATCH_SIZE x NUM_OF_POINTS x  POINT_CLOUD_FUNCTION_DIM
        if (interpolation):
            w_tensor = tf.matrix_solve_ls(self.block_elemnts.get_interpolation_matrix(),
                                          functions_pl,l2_regularizer=self.l2_regularizer)
        else:
            w_tensor = tf.multiply(
                tf.expand_dims(tf.reciprocal(tf.reduce_sum(self.block_elemnts.get_interpolation_matrix(), axis=2)),
                               axis=2), functions_pl)

        # Calculate the product of w_tensor and the kernel weights. Result dimensions are TRANSLATIONS x BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        b_tensor = tf.transpose(tf.matmul(tf.tile(tf.expand_dims(w_tensor, axis=1), [1, self.out_channels, 1, 1]),
                                          tf.tile(tf.expand_dims(self.k_tensor, axis=0),
                                                  [self.block_elemnts.batch_size, 1, 1, 1])),
                                [3, 0, 2, 1])

        def convopeator_per_translation(input):
            translation_index = input[1]
            b_per_translation = input[0]

            dot = tf.tile(-2 * tf.slice(points_translations_dot, [0, 0, translation_index],
                                        [self.block_elemnts.batch_size, self.block_elemnts.num_of_points, 1]),
                          [1, 1, self.block_elemnts.num_of_points])

            q_tensor = tf.exp(-(tf.add(tf.add(distances,
                                              tf.add(dot, -tf.transpose(dot, [0, 2, 1]))),
                                       tf.expand_dims(tf.slice(translations_square, [0, translation_index],
                                                               [self.block_elemnts.batch_size, 1]), axis=2)))
                              / (2 * (self.block_elemnts.combined_sigma) ** 2))

            return tf.matmul(q_tensor, b_per_translation)

        # Calculate the pcnn convolution operator. Result dimensions are BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        return tf.reduce_sum(tf.map_fn(convopeator_per_translation,
                                              elems=(b_tensor, tf.range(0, self.block_elemnts.num_of_translations)),
                                              dtype=dtype), axis=0)

    def get_layer_org(self, functions_pl, with_bn, bn_decay,interpolation,with_Relu = True):

        convlution_operation = self.get_convlution_operator_org(functions_pl,interpolation)

        with tf.variable_scope(self.scope) as sc:
            biases = tf_util._variable_on_cpu('biases', [self.out_channels],
                                              tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(convlution_operation, biases)

            if (with_bn):
                outputs = tf_util.batch_norm_template(outputs, self.is_training, 'bn', [0, 1], bn_decay)

            if (with_Relu):
                outputs = tf.nn.relu(outputs)
            return outputs