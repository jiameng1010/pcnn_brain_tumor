import tensorflow as tf
import sys
sys.path.append('../')
import tf_util
import numpy as np

class ConvLayer_pn:
    def __init__(self, input_channels, block_elemnts, out_channels,scope,is_training,use_xavier=True,l2_regularizer = 1.0e-3):
        self.input_channels = input_channels
        self.block_elemnts = block_elemnts
        self.out_channels = int(out_channels/2)
        self.scope = scope
        self.use_xavier = use_xavier
        self.is_training = is_training
        self.weight_decay = 0.0
        self.l2_regularizer = l2_regularizer

        with tf.variable_scope(self.scope) as sc:

            self.k_tensor = tf_util._variable_with_weight_decay('weights',
                                                           shape=[self.out_channels, self.input_channels, self.block_elemnts.num_of_translations],
                                                           use_xavier=self.use_xavier,
                                                           stddev=0.1, wd=self.weight_decay)
            self.k_tensor2 = tf_util._variable_with_weight_decay('weights2',
                                                           shape=[self.out_channels, self.input_channels, self.block_elemnts.num_of_translations],
                                                           use_xavier=self.use_xavier,
                                                           stddev=0.1, wd=self.weight_decay)

    def get_convlution_operator(self,functions_pl,interpolation,dtype=tf.float32):
        alpha = self.block_elemnts.alpha
        beta = self.block_elemnts.beta
        denominator = 0.5/(self.block_elemnts.sigma_f * self.block_elemnts.sigma_f)
        #(16 * 27 * 3)
        translations = self.block_elemnts.kernel_translations

        difference = tf.tile(tf.expand_dims(self.block_elemnts.points_pl, axis=2), [1, 1, self.block_elemnts.num_of_translations, 1])\
        - tf.tile(tf.expand_dims(translations, axis=1), [1, self.block_elemnts.num_of_points, 1, 1])
        difference2 = tf.reduce_sum(tf.multiply(difference, difference), axis=3)

        N_tensor = tf.matmul(tf.expand_dims(self.block_elemnts.normals_pl, axis=3), tf.expand_dims(self.block_elemnts.normals_pl, axis=2))
        A_tensor = 2 * denominator * ((alpha * alpha - beta * beta)*N_tensor\
                    + (1.0 + beta * beta)*tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(3), axis=0), axis=0), [self.block_elemnts.batch_size, self.block_elemnts.num_of_points, 1, 1]))
        A_inv_tensor = tf.linalg.inv(A_tensor)
        A_det_tensor = tf.linalg.det(A_tensor)
        pTN = tf.squeeze(tf.matmul(tf.expand_dims(self.block_elemnts.points_pl, axis=2), N_tensor))
        #n2p = tf.multiply(tf.tile(tf.reduce_sum(tf.multiply(self.block_elemnts.normals_pl, self.block_elemnts.normals_pl), axis=2, keep_dims=True), [1,1,3]), self.block_elemnts.points_pl)
        n2p = self.block_elemnts.points_pl
        B_tensor1 = denominator * (- 2*(alpha * alpha - beta * beta)*pTN - 2*beta * beta*n2p)
        C_tensor1 = denominator * ((alpha * alpha - beta * beta)*tf.reduce_sum(tf.multiply(pTN, self.block_elemnts.points_pl), axis=2)\
                                   + (beta * beta)*tf.reduce_sum(tf.multiply(n2p, self.block_elemnts.points_pl), axis=2))

        #distances = self.block_elemnts.get_distance_matrix()
        points_translations_dot = tf.matmul(self.block_elemnts.points_pl, tf.transpose(translations, [0, 2, 1]))
        translations_square = tf.reduce_sum(translations * translations, axis=2)

        # Find weights w of the extension operator. result dimensions are BATCH_SIZE x NUM_OF_POINTS x  POINT_CLOUD_FUNCTION_DIM
        if (interpolation):
            w_tensor = tf.matrix_solve_ls(self.block_elemnts.get_interpolation_matrix(),
                                          functions_pl,l2_regularizer=self.l2_regularizer)
        else:
            w_tensor = tf.multiply(
                tf.expand_dims(tf.reciprocal(tf.reduce_sum(self.block_elemnts.get_interpolation_matrix_org(), axis=2)),
                               axis=2), functions_pl)
            #w_tensor = functions_pl

        # Calculate the product of w_tensor and the kernel weights. Result dimensions are TRANSLATIONS x BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        b_tensor = tf.transpose(tf.matmul(tf.tile(tf.expand_dims(w_tensor, axis=1), [1, self.out_channels, 1, 1]),
                                          tf.tile(tf.expand_dims(self.k_tensor, axis=0),
                                                  [self.block_elemnts.batch_size, 1, 1, 1])),[3, 0, 2, 1])

        def convopeator_per_translation(input):
            translation_index = input[1]
            b_per_translation = input[0]

            #(16 * 1024 * 1024)
            difference_slice = denominator * tf.tile(tf.slice(difference, [0, 0, translation_index, 0], [self.block_elemnts.batch_size, self.block_elemnts.num_of_points, 1, 3]), [1, 1, self.block_elemnts.num_of_points, 1])
            difference_slice = tf.transpose(difference_slice, [0, 2, 1, 3])
            difference2_slice = denominator * tf.tile(tf.slice(difference2, [0, 0, translation_index], [self.block_elemnts.batch_size, self.block_elemnts.num_of_points, 1]), [1, 1, self.block_elemnts.num_of_points])
            difference2_slice = tf.transpose(difference2_slice, [0, 2, 1])

            #(16 * 1024 * 1024)
            C_tensor = - (tf.tile(tf.expand_dims(C_tensor1, axis=2), [1, 1, self.block_elemnts.num_of_points]) + difference2_slice)

            #(16 * 1024 * 1024 * 3)
            B_tensor = - (tf.tile(tf.expand_dims(B_tensor1, axis=2), [1, 1, self.block_elemnts.num_of_points, 1])\
                          - 2 * difference_slice)

            #(16 * 1024 * 1024 * 3)
            AinvB = tf.squeeze(tf.matmul( tf.tile(tf.expand_dims(A_inv_tensor, axis=2), [1, 1, self.block_elemnts.num_of_points, 1, 1]), \
                               tf.expand_dims(B_tensor, axis=4)))

            #q_tensor = tf.exp(tf.reduce_sum(tf.multiply(B_tensor, AinvB), axis=3))
            #(16 * 1024 * 1024)
            q_tensor = tf.exp(C_tensor + 0.5 * tf.reduce_sum(tf.multiply(B_tensor, AinvB), axis=3))
            q_tensor = tf.multiply(q_tensor, tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(self.block_elemnts.normals_pl, axis=2), [1, 1, self.block_elemnts.num_of_points, 1]),\
                (AinvB - tf.tile(tf.expand_dims(self.block_elemnts.points_pl, axis=2), [1, 1, self.block_elemnts.num_of_points, 1])) )))

            q_tensor = tf.multiply(q_tensor, tf.tile(tf.reciprocal(tf.expand_dims((tf.sqrt(A_det_tensor)), axis=2)), (1, 1, self.block_elemnts.num_of_points)))
            q_tensor = np.sqrt(8*np.pi*np.pi*np.pi) * q_tensor
            q_tensor = tf.transpose(q_tensor, [0, 2, 1])

            return tf.matmul(q_tensor, b_per_translation)

        # Calculate the pcnn convolution operator. Result dimensions are BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        return tf.reduce_sum(tf.map_fn(convopeator_per_translation,
                                              elems=(b_tensor, tf.range(0, self.block_elemnts.num_of_translations)),
                                              dtype=dtype), axis=0)

    def get_layer(self, functions_pl, with_bn, bn_decay,interpolation,with_Relu = True):

        convlution_operation1 = self.get_convlution_operator_org(functions_pl,interpolation)
        convlution_operation2 = self.get_convlution_operator_2(functions_pl,interpolation, type=3)
        convlution_operation = tf.concat([convlution_operation1, convlution_operation2], axis=2)

        with tf.variable_scope(self.scope) as sc:
            #biases = tf_util._variable_on_cpu('biases', [self.out_channels],
            #                                  tf.constant_initializer(0.0))
            biases = tf.get_variable(name='biases', shape=[self.out_channels*2],  initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            outputs = tf.nn.bias_add(convlution_operation, biases)

            if (with_bn):
                outputs = tf_util.batch_norm_template(outputs, self.is_training, 'bn', [0, 1], bn_decay)

            if (with_Relu):
                outputs = tf.nn.relu(outputs)
                #outputs = tf.clip_by_value(outputs, tf.constant([0.0]), tf.reduce_mean(outputs, axis=1, keepdims=True))#######################################################3
            return outputs, tf.reduce_sum(tf.to_float(tf.greater(outputs, tf.zeros_like(outputs))))/tf.reduce_sum(tf.ones_like(outputs))

    def get_convlution_operator_3(self, functions_pl, interpolation, dtype=tf.float32, type=1):
        alpha = self.block_elemnts.alpha
        beta = self.block_elemnts.beta
        gamma = self.block_elemnts.gamma
        # (16 * 27 * 3)
        translations = self.block_elemnts.kernel_translations

        difference = tf.tile(tf.expand_dims(self.block_elemnts.points_pl, axis=2),
                             [1, 1, self.block_elemnts.num_of_translations, 1]) \
                     - tf.tile(tf.expand_dims(translations, axis=1), [1, self.block_elemnts.num_of_points, 1, 1])

        # xyp = tf.tile(tf.expand_dims(difference, axis=2), [1,1,self.block_elemnts.num_of_points,1,1]) - tf.tile(tf.expand_dims(tf.expand_dims(translations, axis=1), axis=1), [1,self.block_elemnts.num_of_points,self.block_elemnts.num_of_points,1,1])

        # Find weights w of the extension operator. result dimensions are BATCH_SIZE x NUM_OF_POINTS x  POINT_CLOUD_FUNCTION_DIM
        if (interpolation):
            w_tensor = tf.matrix_solve_ls(self.block_elemnts.get_interpolation_matrix(),
                                          functions_pl, l2_regularizer=self.l2_regularizer)
        else:
            w_tensor = tf.multiply(
               tf.expand_dims(tf.reciprocal(tf.reduce_sum(self.block_elemnts.get_interpolation_matrix_org(), axis=2)),
                              axis=2), functions_pl)
            #w_tensor = functions_pl

        # Calculate the product of w_tensor and the kernel weights. Result dimensions are TRANSLATIONS x BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        b_tensor = tf.transpose(tf.matmul(tf.tile(tf.expand_dims(w_tensor, axis=1), [1, self.out_channels, 1, 1]),
                                          tf.tile(tf.expand_dims(self.k_tensor, axis=0),
                                                  [self.block_elemnts.batch_size, 1, 1, 1])), [3, 0, 2, 1])

        def convopeator_per_translation(input):
            translation_index = input[1]
            b_per_translation = input[0]

            # (16 * 1024' * 1024*3)
            difference_slice = tf.tile(tf.slice(difference, [0, 0, translation_index, 0],
                                                [self.block_elemnts.batch_size, self.block_elemnts.num_of_points, 1,
                                                 3]), [1, 1, self.block_elemnts.num_of_points, 1])
            # difference_slice = tf.transpose(difference_slice, [0, 2, 1, 3])

            # (16 * 1024' * 1024 * 3)
            xyp = - difference_slice + tf.tile(tf.expand_dims(self.block_elemnts.points_pl, axis=1),
                                               [1, self.block_elemnts.num_of_points, 1, 1])
            # (16 * 1024' * 1024)
            b = tf.reduce_sum(tf.multiply(xyp, tf.tile(tf.expand_dims(self.block_elemnts.normals_pl, axis=1),
                                                       [1, self.block_elemnts.num_of_points, 1, 1])), axis=3)
            b2 = tf.multiply(b, b)
            a2 = tf.reduce_sum(tf.multiply(xyp, xyp), axis=3) - b2
            # a2 = tf.multiply(a, a)

            # (16 * 1024' * 1024)
            q_tensor = - (((gamma ** 2) * (beta ** 2)) / (
                        2 * ((gamma ** 2) + (beta ** 2)) * ((self.block_elemnts.sigma_f) ** 2))) * a2 \
                       - (((gamma ** 2) * (alpha ** 2)) / (
                        2 * ((gamma ** 2) + (alpha ** 2)) * ((self.block_elemnts.sigma_f) ** 2))) * b2
            q_tensor = tf.exp(q_tensor)
            # q_tensor = tf.multiply(b2, q_tensor)
            factor = 2 * np.pi * (np.power(self.block_elemnts.sigma_f, 2)) / (beta ** 2 + gamma ** 2)
            q_tensor = factor * q_tensor

            if type == 1:
                # q_tensor = tf.multiply(b, q_tensor)
                aa = (alpha**2+gamma**2) / (2 * self.block_elemnts.sigma_f**2)
                bb2 = (gamma**2 / self.block_elemnts.sigma_f**2)**2 * b2
                cc = (alpha**2) / (self.block_elemnts.sigma_f**2)

                factor = ((bb2 + 2*aa) * cc) - 2*aa*aa
                q_tensor = factor * q_tensor


            return tf.matmul(q_tensor, b_per_translation)

        # Calculate the pcnn convolution operator. Result dimensions are BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        return tf.reduce_sum(tf.map_fn(convopeator_per_translation,
                                       elems=(b_tensor, tf.range(0, self.block_elemnts.num_of_translations)),
                                       dtype=dtype), axis=0)

    def get_convlution_operator_2(self,functions_pl,interpolation,dtype=tf.float32,type=1):
        alpha = self.block_elemnts.alpha
        beta = self.block_elemnts.beta
        gamma = self.block_elemnts.gamma
        #(16 * 27 * 3)
        translations = self.block_elemnts.kernel_translations

        difference = tf.tile(tf.expand_dims(self.block_elemnts.points_pl, axis=2), [1, 1, self.block_elemnts.num_of_translations, 1])\
        - tf.tile(tf.expand_dims(translations, axis=1), [1, self.block_elemnts.num_of_points, 1, 1])

        #xyp = tf.tile(tf.expand_dims(difference, axis=2), [1,1,self.block_elemnts.num_of_points,1,1]) - tf.tile(tf.expand_dims(tf.expand_dims(translations, axis=1), axis=1), [1,self.block_elemnts.num_of_points,self.block_elemnts.num_of_points,1,1])

        # Find weights w of the extension operator. result dimensions are BATCH_SIZE x NUM_OF_POINTS x  POINT_CLOUD_FUNCTION_DIM
        if (interpolation):
            w_tensor = tf.matrix_solve_ls(self.block_elemnts.get_interpolation_matrix(),
                                          functions_pl,l2_regularizer=self.l2_regularizer)
        else:
            w_tensor = tf.multiply(
                tf.expand_dims(tf.reciprocal(tf.reduce_sum(self.block_elemnts.get_interpolation_matrix_org(), axis=2)),
                               axis=2), functions_pl)
            #w_tensor = functions_pl

        # Calculate the product of w_tensor and the kernel weights. Result dimensions are TRANSLATIONS x BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        b_tensor = tf.transpose(tf.matmul(tf.tile(tf.expand_dims(w_tensor, axis=1), [1, self.out_channels, 1, 1]),
                                          tf.tile(tf.expand_dims(self.k_tensor2, axis=0),
                                                  [self.block_elemnts.batch_size, 1, 1, 1])),[3, 0, 2, 1])

        def convopeator_per_translation(input):
            translation_index = input[1]
            b_per_translation = input[0]

            #(16 * 1024' * 1024*3)
            difference_slice = tf.tile(tf.slice(difference, [0, 0, translation_index, 0], [self.block_elemnts.batch_size, self.block_elemnts.num_of_points, 1, 3]), [1, 1, self.block_elemnts.num_of_points, 1])
            #difference_slice = tf.transpose(difference_slice, [0, 2, 1, 3])

            #(16 * 1024' * 1024 * 3)
            xyp = - difference_slice + tf.tile(tf.expand_dims(self.block_elemnts.points_pl, axis=1), [1,self.block_elemnts.num_of_points,1,1])
            # (16 * 1024' * 1024)
            b = tf.reduce_sum(tf.multiply(xyp, tf.tile(tf.expand_dims(self.block_elemnts.normals_pl, axis=1), [1,self.block_elemnts.num_of_points,1,1])), axis=3)
            b2 = tf.multiply(b, b)
            a2 = tf.reduce_sum(tf.multiply(xyp, xyp), axis=3) - b2
            #a2 = tf.multiply(a, a)

            # (16 * 1024' * 1024)
            q_tensor = - (((gamma**2)*(beta**2)) / (2*((gamma**2)+(beta**2))*((self.block_elemnts.sigma_f)**2)))*a2\
                       - (((gamma**2)*(alpha**2)) / (2*((gamma**2)+(alpha**2))*((self.block_elemnts.sigma_f)**2)))*b2
            q_tensor = tf.exp(q_tensor)
            #q_tensor = tf.multiply(b2, q_tensor)
            factor = 2 * np.pi * (np.power(self.block_elemnts.sigma_f, 2)) / (beta**2+gamma**2)
            q_tensor = factor * q_tensor

            if type == 1:
                #q_tensor = tf.multiply(b, q_tensor)
                factor = self.block_elemnts.sigma_f * (gamma**2) * np.sqrt(2 * np.pi) / (alpha**2+gamma**2)
                q_tensor = factor * q_tensor
            elif type ==2:
                aa = (alpha**2+gamma**2) / (2 * self.block_elemnts.sigma_f**2)
                bb = (gamma**2 / self.block_elemnts.sigma_f**2) * (b/self.block_elemnts.sigma_f)
                b24a = gamma**4 / (2 * self.block_elemnts.sigma_f**2 * (alpha**2+gamma**2))
                b24a = b24a * (b2/self.block_elemnts.sigma_f**2)
                factor = np.sqrt(np.pi)*(tf.igammac(0.5*tf.ones_like(b24a), b24a)-2*tf.ones_like(b24a)) * aa * bb\
                         - 2 * tf.igammac(tf.ones_like(b24a), b24a) * np.power(aa, 1.5)
                factor = - factor / (4 * np.power(aa, 2.5))
                q_tensor = factor * q_tensor
                #q_tensor = tf.abs(q_tensor)
            elif type == 3:
                aa = (alpha**2+gamma**2) / (2 * self.block_elemnts.sigma_f**2)
                bb = (gamma**2 / self.block_elemnts.sigma_f**2) * b / self.block_elemnts.sigma_f
                cc = 0.5 * alpha**2 / (2 * self.block_elemnts.sigma_f**2)
                factor = np.sqrt(np.pi) * tf.exp(-(cc**2)/(4 * aa)) * tf.sin(0.5*cc*bb/(2*aa)) / np.power(aa, 0.5)
                #factor = np.sqrt(np.pi)# * tf.exp(-1 / (4 * aa))# / np.power(aa, 0.5)
                q_tensor = factor * q_tensor
            elif type == 4:
                aa = (alpha**2+gamma**2) / (2 * self.block_elemnts.sigma_f**2)
                bb = (gamma**2 / self.block_elemnts.sigma_f**2) * b
                factor = np.sqrt(np.pi)*(bb*bb + 2*aa) / (4 * np.power(aa, 2.5))
                q_tensor = factor * q_tensor
            elif type == 5:
                aa = (alpha**2+gamma**2) / (2 * self.block_elemnts.sigma_f**2)
                bb = (gamma**2 / self.block_elemnts.sigma_f**2) * b
                factor = np.sqrt(np.pi) * tf.math.erf(bb/(2*np.power(aa, 0.5))) / np.power(aa, 0.5)
                q_tensor = factor * q_tensor



            return tf.matmul(q_tensor, b_per_translation)

        # Calculate the pcnn convolution operator. Result dimensions are BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        return tf.reduce_sum(tf.map_fn(convopeator_per_translation,
                                              elems=(b_tensor, tf.range(0, self.block_elemnts.num_of_translations)),
                                              dtype=dtype), axis=0)


    def get_convlution_operator_org(self,functions_pl,interpolation,dtype=tf.float32):
        translations = self.block_elemnts.kernel_translations

        distances = self.block_elemnts.get_distance_matrix()
        points_translations_dot = tf.matmul(self.block_elemnts.points_pl, tf.transpose(translations, [0, 2, 1]))
        translations_square = tf.reduce_sum(translations * translations, axis=2)

        # Find weights w of the extension operator. result dimensions are BATCH_SIZE x NUM_OF_POINTS x  POINT_CLOUD_FUNCTION_DIM
        if (interpolation):
            w_tensor = tf.matrix_solve_ls(self.block_elemnts.get_interpolation_matrix_org(),
                                          functions_pl,l2_regularizer=self.l2_regularizer)
        else:
            w_tensor = tf.multiply(
                tf.expand_dims(tf.reciprocal(tf.reduce_sum(self.block_elemnts.get_interpolation_matrix_org(), axis=2)),
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

        # Calculate the pcnn convolution operator. Result dimensions are BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        return tf.reduce_sum(tf.map_fn(convopeator_per_translation,
                                              elems=(b_tensor, tf.range(0, self.block_elemnts.num_of_translations)),
                                              dtype=dtype), axis=0)