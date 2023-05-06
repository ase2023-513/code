# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time
import utils.utils_activation as utils
import os
from scipy import misc
from scipy import ndimage
import PIL
import io
import pickle

slim = tf.contrib.slim

tf.flags.DEFINE_string('model_name', 'mobilenet_v1', 'The Model used to generate adv.')
tf.flags.DEFINE_string('layer_name','MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6','The layer to be attacked.')
tf.flags.DEFINE_string('input_dir', 'class_data_new/class_', 'Input directory with images.')
tf.flags.DEFINE_string('output_dir', './adv/', 'Output directory with images.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')
tf.flags.DEFINE_float('alpha', 1.6, 'Step size.')
tf.flags.DEFINE_integer('batch_size', 1, 'How many images process at one time.')
tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')
tf.flags.DEFINE_string('GPU_ID', '0', 'which GPU to use.')
tf.flags.DEFINE_integer('image_size', 224, 'size of each input images.')

FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID

def get_opt_layers(layer_name):
    opt_operations = []
    operations = tf.get_default_graph().get_operations()
    for op in operations:
        print(op.name)
        if layer_name == op.name:
            opt_operations.append(op.outputs[0])
            shape=op.outputs[0][:FLAGS.batch_size].shape
            break
    return opt_operations,shape


def get_coverage(opt_operations):
    for layer in opt_operations:
        batch_size = FLAGS.batch_size
        tensor = layer[:batch_size]
        mean_ = tf.reduce_mean(tensor, [1,2])
        mean_mid = tf.stack([mean_] * tensor.shape[2],1)
        mean_tensor = tf.stack([mean_mid] * tensor.shape[1],1)
        wts_good = tensor < mean_
        wts_good = tf.to_float(wts_good)
        wts_bad = tensor > mean_
        wts_bad = tf.to_float(wts_bad)
        coverage = wts_good - wts_bad
    return coverage

def get_tensor(opt_operations):
    layer = opt_operations[0]
    batch_size = FLAGS.batch_size
    tensor = layer[:batch_size]
    return tensor


def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1:
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square)
    return nor_grad


def main(_):

    if FLAGS.model_name in ['vgg_16','vgg_19', 'resnet_v1_50','resnet_v1_152']:
        eps = FLAGS.max_epsilon
        alpha = FLAGS.alpha
    else:
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        alpha = FLAGS.alpha * 2.0 / 255.0

    num_iter = FLAGS.num_iter
    momentum = FLAGS.momentum

    image_preprocessing_fn = utils.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    checkpoint_path = utils.checkpoint_paths[FLAGS.model_name]
    layer_name=FLAGS.layer_name
    num_classes = 1000 + utils.offset[FLAGS.model_name]

    with tf.Graph().as_default():
        # Prepare graph
        ori_input  = tf.placeholder(tf.float32, shape=batch_shape)
        network_fn = utils.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
        x=ori_input
        logits, end_points = network_fn(x)

        problity=tf.nn.softmax(logits,axis=1)
        pred = tf.argmax(logits, axis=1)
        one_hot = tf.one_hot(pred, num_classes)


        opt_operations,shape = get_opt_layers(layer_name)
        coverage = get_coverage(opt_operations)
        inter_tensor = get_tensor(opt_operations)


        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,checkpoint_path)
            all_coverages = []
            all_activations = []
            all_activations_max = []
            all_activations_min = []
            for i in range(1000):
                coverages = []
                inter_tensors = []
                for images,names,labels in utils.load_image(FLAGS.input_dir+str(i+1)+"/", FLAGS.image_size,FLAGS.batch_size, i+1):
                    #print(names)
                    if len(images.shape) == 3:
                        images_exp = np.expand_dims(np.copy(images), axis=-1)
                        images_stack = np.concatenate((images_exp,images_exp,images_exp),axis=-1)
                        images_tmp=image_preprocessing_fn(images_stack)
                    else:
                        if images.shape[-1] == 4:
                            images_tmp=image_preprocessing_fn(np.copy(images[:,:,:,:-1]))
                        else:
                            images_tmp=image_preprocessing_fn(np.copy(images))
                    if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                        labels=labels-1

                    # obtain true label
                    labels= to_categorical(np.concatenate([labels,labels],axis=-1),num_classes)
                    pred_ = sess.run(pred, feed_dict={ori_input: images_tmp})
                    coverage_tmp = sess.run(coverage, feed_dict={ori_input:images_tmp})
                    coverages.append(coverage_tmp)
                    inter_tensor_tmp = sess.run(inter_tensor, feed_dict={ori_input:images_tmp})
                    inter_tensors.append(inter_tensor_tmp)
                coverages = np.array(coverages)
                coverages_mean = np.mean(coverages, axis = 0)
                inter_tensors = np.array(inter_tensors)
                activations_mean = np.mean(inter_tensors, axis = 0)
                all_coverages.append(coverages_mean[0,:,:,:])
                all_activations.append(activations_mean[0,:,:,:])
                activations_max = np.max(inter_tensors, axis = 0)
                activations_min = np.min(inter_tensors, axis = 0)
                all_activations_max.append(activations_max[0,:,:,:])
                all_activations_min.append(activations_min[0,:,:,:])
            
            all_coverages = np.array(all_coverages)
            all_activations = np.array(all_activations)
            with open("coverage_"+FLAGS.model_name+"_new.pkl","wb") as f:
                pickle.dump(all_coverages, f)
            with open("activation_"+FLAGS.model_name+"_new.pkl","wb") as f:
                pickle.dump(all_activations, f)
            all_activations_max = np.array(all_activations_max)
            all_activations_min = np.array(all_activations_min)
            
            with open("activation_"+FLAGS.model_name+"_min_new.pkl","wb") as f:
                pickle.dump(all_activations_min, f)
            with open("activation_"+FLAGS.model_name+"_max_new.pkl","wb") as f:
                pickle.dump(all_activations_max, f)
            
                
if __name__ == '__main__':
    tf.app.run()