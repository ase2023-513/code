import tensorflow as tf
import numpy as np
import argparse
import utils
import csv
import os
from tensorflow.keras.utils import to_categorical
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_names=['vgg_16']

def verify(model_name,ori_image_path,adv_image_path):

    checkpoint_path=utils.checkpoint_paths[model_name]

    if model_name=='adv_inception_v3' or model_name=='ens3_adv_inception_v3' or model_name=='ens4_adv_inception_v3':
        model_name='inception_v3'
    elif model_name=='adv_inception_resnet_v2' or model_name=='ens_adv_inception_resnet_v2':
        model_name='inception_resnet_v2'

    num_classes=1000+utils.offset[model_name]

    network_fn = utils.nets_factory.get_network_fn(
        model_name,
        num_classes=(num_classes),
        is_training=False)

    image_preprocessing_fn = utils.normalization_fn_map[model_name]
    image_size = utils.image_size[model_name]

    batch_size=200
    image_ph=tf.placeholder(dtype=tf.float32,shape=[batch_size,image_size,image_size,3])
    label_ph = tf.placeholder(tf.float32, shape=[batch_size*2,num_classes])

    logits, _ = network_fn(image_ph)
    confidence = tf.nn.softmax(logits) * label_ph[:batch_size]
    print(confidence.shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.get_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess,checkpoint_path)

        ori_pre=[]
        adv_pre=[]
        ground_truth=[]

        for images,names,labels in utils.load_image(adv_image_path, image_size, batch_size):
            if model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                labels=labels-1
            labels= to_categorical(np.concatenate([labels,labels],axis=-1),num_classes)
            images=image_preprocessing_fn(images)
            pres=sess.run(confidence,feed_dict={image_ph:images,label_ph:labels})
            ori_pre.extend(pres)
    tf.reset_default_graph()
    ori_pre=np.array(ori_pre)
    return ori_pre


def main(ori_path='./dataset/images/',adv_path='./adv/',output_file='./log.csv'):
    ori_accuracys=[]
    adv_accuracys=[]
    adv_successrates=[]
    with open(output_file,'a+',newline='') as f:
        writer=csv.writer(f)
        writer.writerow([adv_path])
        writer.writerow(model_names)
        for model_name in model_names:
            print(model_name)
            ori_pre=verify(model_name,ori_path,adv_path)
            ori_accuracy = np.sum(ori_pre)/1000
        writer.writerow(adv_successrates)
    print(ori_accuracy)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ori_path', default='./dataset/images/')
    parser.add_argument('--adv_path',default='./Vgg16/NAPCTest/')
    parser.add_argument('--output_file', default='./log.csv')
    args=parser.parse_args()
    main(args.ori_path,args.adv_path,args.output_file)
