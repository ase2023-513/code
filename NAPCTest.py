"""Implementation of NAPCTest."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time
import utils.utils_NAPC
import os
from scipy import misc
from scipy import ndimage
import PIL
import io
import pickle

slim = tf.contrib.slim

tf.flags.DEFINE_string('model_name', 'vgg_16', 'The Model used to generate adv.')
tf.flags.DEFINE_string('layer_name','vgg_16/conv3/conv3_3/Relu','The layer to be attacked.')
tf.flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')
tf.flags.DEFINE_string('output_dir', './Vgg16/NAPCTest/', 'Output directory with images.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')
tf.flags.DEFINE_float('alpha', 1.6, 'Step size.')
tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')
tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')
tf.flags.DEFINE_string('GPU_ID', '0', 'which GPU to use.')
tf.flags.DEFINE_integer('image_size', 224, 'size of each input images.')
tf.flags.DEFINE_float('mix', 0.5, 'threshold for coverage.')

FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID

def get_opt_layers(layer_name):
    opt_operations = []
    operations = tf.get_default_graph().get_operations()
    for op in operations:
        if layer_name == op.name:
            opt_operations.append(op.outputs[0])
            shape=op.outputs[0][:FLAGS.batch_size].shape
            break
    return opt_operations,shape


def get_mask(opt_operations,activation_ph,activation_ph_min,activation_ph_max):
    for layer in opt_operations:
        batch_size = FLAGS.batch_size
        tensor = layer[batch_size:]
        mask_neg = tensor < FLAGS.mix*activation_ph+(1-FLAGS.mix)*activation_ph_min
        mask_neg = tf.to_float(mask_neg)
        mask_neg_mid = tensor < activation_ph
        mask_neg_mid = tf.to_float(mask_neg_mid) - mask_neg
        mask_pos = tensor > FLAGS.mix*activation_ph+(1-FLAGS.mix)*activation_ph_max
        mask_pos = tf.to_float(mask_pos)
        mask_pos_mid = tensor > activation_ph
        mask_pos_mid = tf.to_float(mask_pos_mid) - mask_pos

        
    return mask_neg, mask_neg_mid, mask_pos_mid, mask_pos


def get_nc_loss(opt_operations,mask_pos,mask_neg,activation_ph,mask_ph_neg_mid,mask_ph_pos_mid):
    loss = 0
    for layer in opt_operations:
        batch_size = FLAGS.batch_size
        tensor = layer[:batch_size]
        loss += tf.reduce_sum((mask_neg+mask_ph_neg_mid) * layer[batch_size:]) / tf.cast(tf.size(layer),tf.float32)
        loss -= tf.reduce_sum((mask_pos+mask_ph_pos_mid) * layer[batch_size:]) / tf.cast(tf.size(layer),tf.float32)

    loss = loss / len(opt_operations)
    return loss


def get_activation(opt_operations,mask_ph_pos,mask_ph_neg,activation_ph):
    for layer in opt_operations:
        batch_size = FLAGS.batch_size
        tensor = layer[batch_size:]
        wts_good = tensor < activation_ph
        wts_good = tf.to_float(wts_good)
        wts_bad = tensor > activation_ph
        wts_bad = tf.to_float(wts_bad)   
        
    return wts_bad, wts_good

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

    with tf.Graph().as_default():
        # Prepare graph
        ori_input  = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input = tf.placeholder(tf.float32, shape=batch_shape)
        num_classes = 1000 + utils.offset[FLAGS.model_name]
        label_ph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size*2,num_classes])
        accumulated_grad_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)
        amplification_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)

        network_fn = utils.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
        x=tf.concat([ori_input,adv_input],axis=0)
        logits, end_points = network_fn(x)

        problity=tf.nn.softmax(logits,axis=1)
        pred = tf.argmax(logits, axis=1)
        one_hot = tf.one_hot(pred, num_classes)

        entropy_loss = tf.losses.softmax_cross_entropy(one_hot[:FLAGS.batch_size], logits[FLAGS.batch_size:])

        opt_operations,shape = get_opt_layers(layer_name)
        #mask_ph = tf.placeholder(dtype=tf.float32, shape=shape)
        mask_ph_pos = tf.placeholder(dtype=tf.float32, shape=shape)
        mask_ph_neg = tf.placeholder(dtype=tf.float32, shape=shape)
        activation_ph = tf.placeholder(dtype=tf.float32, shape=shape)
        activation_ph_min = tf.placeholder(dtype=tf.float32, shape=shape)
        activation_ph_max = tf.placeholder(dtype=tf.float32, shape=shape)
        mask_ph_neg_mid = tf.placeholder(dtype=tf.float32, shape=shape)
        mask_ph_pos_mid = tf.placeholder(dtype=tf.float32, shape=shape)

        mask_neg_, mask_neg_mid_, mask_pos_mid_, mask_pos_ = get_mask(opt_operations,activation_ph,activation_ph_min,activation_ph_max)

        mask_activation, mask_deactivation = get_activation(opt_operations,mask_ph_pos,mask_ph_neg,activation_ph)


        loss = get_nc_loss(opt_operations,mask_ph_pos,mask_ph_neg,activation_ph,mask_ph_neg_mid,mask_ph_pos_mid)

        gradient=tf.gradients(loss,adv_input)[0]

        noise = gradient
        adv_input_update = adv_input
        amplification_update = amplification_ph

        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
        noise = momentum * accumulated_grad_ph + noise
        adv_input_update = adv_input_update + alpha * tf.sign(noise)

        saver=tf.train.Saver()
        with tf.Session() as sess:
            with open("coverage_"+FLAGS.model_name+"_new.pkl","rb") as f:
                coverages = pickle.load(f)
            with open("activation_"+FLAGS.model_name+"_new.pkl","rb") as f:
                activations = pickle.load(f)
            with open("activation_"+FLAGS.model_name+"_min_new.pkl","rb") as f:
                activations_min = pickle.load(f)
            with open("activation_"+FLAGS.model_name+"_max_new.pkl","rb") as f:
                activations_max = pickle.load(f)

            saver.restore(sess,checkpoint_path)
            count=0
            for images,names,labels in utils.load_image(FLAGS.input_dir, FLAGS.image_size,FLAGS.batch_size):
                if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']: 
                    condition_coverage = coverages[labels,:,:,:]
                    condition_actiation = activations[labels,:,:,:]
                    condition_actiation_min = activations_min[labels,:,:,:]
                    condition_actiation_max = activations_max[labels,:,:,:]
                else:
                    condition_coverage = coverages[labels-1,:,:,:]
                    condition_actiation = activations[labels-1,:,:,:]
                    condition_actiation_min = activations_min[labels-1,:,:,:]
                    condition_actiation_max = activations_max[labels-1,:,:,:]
                count+=FLAGS.batch_size
                if count%100==0:
                    print("Generating:",count)

                images_tmp=image_preprocessing_fn(np.copy(images))
                if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                    labels=labels-1

                # obtain true label
                labels= to_categorical(np.concatenate([labels,labels],axis=-1),num_classes)

                images_adv=images

                images_adv=image_preprocessing_fn(np.copy(images_adv))

                grad_np=np.zeros(shape=batch_shape)
                amplification_np=np.zeros(shape=batch_shape)
                mask_one = np.ones(shape=shape, dtype=np.float32)
                mask_zeros = np.zeros(shape=shape, dtype=np.float32)

                for i in range(num_iter):
                    if i == 0:
                        mask_neg, mask_neg_mid, mask_pos_mid, mask_pos, mask_act_0, mask_deact_0=sess.run([mask_neg_, mask_neg_mid_, mask_pos_mid_, mask_pos_, mask_activation, mask_deactivation],
                                              feed_dict={ori_input:images_tmp,adv_input:images_adv,
                                                         activation_ph:condition_actiation,activation_ph_min:condition_actiation_min,activation_ph_max:condition_actiation_max,label_ph:labels})
                        
                    # optimization
                    images_adv, grad_np, amplification_np=sess.run([adv_input_update, noise, amplification_update],
                                              feed_dict={ori_input:images_tmp,adv_input:images_adv,mask_ph_pos:mask_pos,mask_ph_pos_mid:mask_pos_mid,mask_ph_neg:mask_neg,mask_ph_neg_mid:mask_neg_mid,
                                                         activation_ph:condition_actiation,label_ph:labels,accumulated_grad_ph:grad_np,amplification_ph:amplification_np})
                    images_adv = np.clip(images_adv, images_tmp - eps, images_tmp + eps)

                images_adv = inv_image_preprocessing_fn(images_adv)
                utils.save_image(images_adv, names, FLAGS.output_dir)
            print("name", FLAGS.model_name)
            print("epsilon", FLAGS.max_epsilon)
            print("threshold", FLAGS.mix)
            print("=================================================")

if __name__ == '__main__':
    tf.app.run()