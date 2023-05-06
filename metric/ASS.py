import tensorflow as tf
import numpy as np
import argparse
import utils
import csv
import os
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
    image_ph1=tf.placeholder(dtype=tf.float32,shape=[batch_size,image_size,image_size,3])
    image_ph2=tf.placeholder(dtype=tf.float32,shape=[batch_size,image_size,image_size,3])

    ssim = tf.image.ssim(image_ph1, image_ph2, max_val=4.0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.get_default_graph()
        ssims=[]
        for images_ori,images_adv in utils.load_two_image(ori_image_path,adv_image_path, image_size, batch_size):
            images_ori=image_preprocessing_fn(images_ori)
            images_adv=image_preprocessing_fn(images_adv)
            ssim_out=sess.run(ssim,feed_dict={image_ph1:images_ori, image_ph2:images_adv})
            ssims.append(ssim_out)
    tf.reset_default_graph()

    return ssims


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
            ssim_all=verify(model_name,ori_path,adv_path)
            ssim_ave = np.sum(ssim_all)/1000
        writer.writerow(adv_successrates)
    print(ssim_ave)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ori_path', default='./dataset/images/')
    parser.add_argument('--adv_path',default='./Vgg16/NAPCTest/')
    parser.add_argument('--output_file', default='./log.csv')
    args=parser.parse_args()
    main(args.ori_path,args.adv_path,args.output_file)
