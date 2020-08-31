'''
Author: Lei He
Date: 2020-08-24 10:57:30
LastEditTime: 2020-08-31 16:05:30
Description: 
Github: https://github.com/heleidsn
'''
import tensorflow as tf
import numpy as np

from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
from scripts_final.policies import FeedForwardPolicy

# vgg with global average pooling
def modified_cnn_vgg_big_v3(scaled_images, **kwargs):
    '''
    @description: a bigger CNN than before, according to https://github.com/hill-a/stable-baselines/issues/735
                    CNN1: 8 filters size 5*5 stride 2
                    CNN2: 8 filters size 3*3 stride 1
                    CNN3: 8 filters size 3*3 stride 1
                  deleted the global average pooling because it lost the obstacle position
    @param {type} 
    @return: 
    '''
    activ = tf.nn.relu

    num_direct_features = 6
    other_features = tf.contrib.slim.flatten(scaled_images[..., -1])
    other_features = other_features[:, :num_direct_features]
    scaled_images = scaled_images[..., :-1]

    layer_1 = activ(conv(scaled_images, 'cnn/c1', n_filters=8, filter_size=3, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))       # 80 100
    layer_1_max_pool = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='cnn/pool1')                  # 40 50
    layer_2 = activ(conv(layer_1_max_pool, 'cnn/c2', n_filters=8, filter_size=3, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))    # 40 50
    layer_2_max_pool = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='cnn/pool2')                  # 20 25
    layer_3 = activ(conv(layer_2_max_pool, 'cnn/c3', n_filters=8, filter_size=3, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))   # 20 25
    layer_3_max_pool = tf.nn.max_pool(layer_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='cnn/pool3')                  # 10 13

    # using global average pooling as the final CNN output
    img_output = tf.keras.layers.GlobalAveragePooling2D()(layer_3_max_pool)

    concat = tf.concat((img_output, other_features), axis=1)
    layers = [layer_1, layer_2, layer_3]

    return concat, layers


class CustomPolicyVGGBigV3(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicyVGGBigV3, self).__init__(*args, **kwargs, layers=[64, 32], cnn_extractor=modified_cnn_vgg_big_v3, feature_extraction="cnn", layer_norm=False)
