import tensorflow as tf
import numpy as np
import argparse
import utils
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_names=['vgg_16']

def verify(model_name,ori_image_path,adv_image_path,batch_size):
    image_preprocessing_fn = utils.normalization_fn_map[model_name]
    with tf.Session() as sess:
        image_size = utils.image_size[model_name]
        batch_size=1
        l2_diff = 0
        num = 0
        for images_ori,images_adv in utils.load_two_image(ori_image_path,adv_image_path, image_size, batch_size):
            images_ori=image_preprocessing_fn(images_ori)
            images_adv=image_preprocessing_fn(images_adv)
            l2_diff += np.mean(np.linalg.norm(images_ori-images_adv, axis=1,ord=np.inf))
            num += batch_size
            if num % 100 == 0:
                print(num)
    return l2_diff * batch_size

def main(ori_path='./dataset/images/',adv_path='./adv/',output_file='./log.csv'):
    ori_accuracys=[]
    adv_accuracys=[]
    adv_successrates=[]
    batch_size = 50
    with open(output_file,'a+',newline='') as f:
        writer=csv.writer(f)
        writer.writerow([adv_path])
        writer.writerow(model_names)
        for model_name in model_names:
            l2=verify(model_name,ori_path,adv_path,batch_size)
            norm = np.sum(l2)/1000
        writer.writerow(adv_successrates)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ori_path', default='./dataset/images/')
    parser.add_argument('--adv_path',default='./Vgg16/NAPCTest/')
    parser.add_argument('--output_file', default='./log.csv')
    args=parser.parse_args()
    main(args.ori_path,args.adv_path,args.output_file)
