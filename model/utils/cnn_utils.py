import tensorflow as tf
import numpy as np

def convolution(name_scope, x, d, di, w, l2_reg, reuse):
    with tf.name_scope(name_scope + "-conv"):
        print("--------conv size----", d)
        with tf.variable_scope("conv") as scope:
            print(x.get_shape())
            conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=di,
                        kernel_size=(d, w),
                        stride=1,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=reuse,
                        trainable=True,
                        scope=scope
                    )
                    # Weight: [filter_height, filter_width, in_channels, out_channels]
                    # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

                    # [batch, di, s+w-1, 1]
            conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
            print(conv_trans.get_shape())
            return conv_trans

# zero padding to inputs for wide convolution
def pad_for_wide_conv(x, w):
    print(x)
    print(w)
    return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

def cos_sim(v1, v2):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
    dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")
    return dot_products / (norm1 * norm2)
        
def euclidean_score(v1, v2):
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
    return 1 / (1 + euclidean)

def make_attention_mat(x1, x2):
    # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
    # x2 => [batch, height, 1, width]
    # [batch, width, wdith] = [batch, s, s]
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
    return 1 / (1 + euclidean)

def w_pool(variable_scope, x, attention, s, w, model_type):
    # x: [batch, di, s+w-1, 1]
    # attention: [batch, s+w-1]
    with tf.variable_scope(variable_scope + "-w_pool"):
        if model_type == "ABCNN2" or model_type == "ABCNN3":
            pools = []
            # [batch, s+w-1] => [batch, 1, s+w-1, 1]
            attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

            for i in range(s):
                # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                pools.append(tf.reduce_sum(x[:, :, i:i + w, :] * attention[:, :, i:i + w, :],
                                                   axis=2,
                                                   keep_dims=True))

                # [batch, di, s, 1]
                w_ap = tf.concat(pools, axis=2, name="w_ap")
        else:
            w_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, w),
                        strides=1,
                        padding="VALID",
                        name="w_ap")
                    # [batch, di, s, 1]

        return w_ap

def all_pool(variable_scope, x, s, w, di):
    with tf.variable_scope(variable_scope + "-all_pool"):
        pool_width = s + w - 1
        d = di
        all_ap = tf.layers.average_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=(1, pool_width),
                    strides=1,
                    padding="VALID",
                    name="all_ap")
        
        all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

        return all_ap_reshaped

def CNN_layer(variable_scope, x1, x2, s, d, w, di, model_type, l2_reg,reuse=False):
    # x1, x2 = [batch, d, s, 1]
    with tf.variable_scope(variable_scope,reuse=reuse):
        if model_type == "ABCNN1" or model_type == "ABCNN3":
            with tf.name_scope("att_mat"):
                aW=tf.get_variable(name="aW", shape=(s, d),
                              initializer=tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))
                aW_left = tf.get_variable(name="aW_left", shape=(s, d),
                              initializer=tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))
                aW_right = tf.get_variable(name="aW_right", shape=(s, d),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))
                        # [batch, s, s]
                att_mat = make_attention_mat(x1, x2)

                        # [batch, s, s] * [s,d] => [batch, s, d]
                        # matrix transpose => [batch, d, s]
                        # expand dims => [batch, d, s, 1]
                x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
                x2_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW)), -1)

                        # [batch, d, s, 2]
                x1 = tf.concat([x1, x1_a], axis=3)
                x2 = tf.concat([x2, x2_a], axis=3)

        left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1, w), d=d, di=di, w=w, l2_reg=l2_reg, reuse=False)
        right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2, w), d=d, di=di, w=w, l2_reg=l2_reg, reuse=True)

        left_attention, right_attention = None, None

        if model_type == "ABCNN2" or model_type == "ABCNN3":
            # [batch, s+w-1, s+w-1]
            att_mat = make_attention_mat(left_conv, right_conv)
                    # [batch, s+w-1], [batch, s+w-1]
            left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)
            
        left_wp = w_pool(variable_scope="left", x=left_conv, attention=left_attention, s=s, w=w, model_type=model_type)
        left_ap = all_pool(variable_scope="left", x=left_conv, s=s, w=w, di=di)
        right_wp = w_pool(variable_scope="right", x=right_conv, attention=right_attention, s=s, w=w, model_type=model_type)
        right_ap = all_pool(variable_scope="right", x=right_conv, s=s, w=w, di=di)

        return left_wp, left_ap, right_wp, right_ap
 