# -*- coding:utf-8 -*-

import csv
import os
import glob
import sys
import cv2
import numpy as np
import shutil
label_name_dict={'person': 0,
              'r-helmet': 1,
              'y-helmet': 2, 'b-helmet': 3, 'w-helmet': 4,
              'head': 5,'y-vest':6,'arm':7,'shank':8,'o-helmet':9,'b-vest':10,'r-vest':11}
label_list=list(label_name_dict.keys())

class PascalVOC2CSV(object):
    def __init__(self, xml=[], ann_path='./annotations.csv', classes_path='./classes.csv'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param ann_path: ann_path
        :param classes_path: classes_path
        '''
        self.xml = xml
        self.ann_path = ann_path
        self.classes_path = classes_path
        self.label = []
        self.annotations = []

        self.data_transfer()
        self.write_file()

    def data_transfer(self):
        for num, xml_file in enumerate(self.xml):
            # if num==50:break
            try:
                # print(xml_file)
                # 进度输出
                sys.stdout.write('\r>> Converting image %d/%d' % (
                    num + 1, len(self.xml)))
                sys.stdout.flush()
                jpg_path=xml_file.replace('.txt','.jpg')
                jpg_path=jpg_path.replace('txt','image')
                img=cv2.imread(jpg_path)
                img_width = img.shape[1]
                img_height = img.shape[0]
                with open(xml_file) as f:

                    for line in f.readlines():
                        curline = line.strip().split(' ')
                        if len(line) < 5: return None, None, None

                        cur_label = int(curline[0])
                        label_name = label_list[cur_label]
                        self.supercategory = label_name
                        self.filen_ame=jpg_path
                        x, y, w, h = int(float(curline[1]) * img_width), int(float(curline[2]) * img_height), int(
                            float(curline[3]) * img_width), int(float(curline[4]) * img_height)
                        xmin = x - w // 2
                        ymin = y - h // 2
                        xmax = x + w // 2
                        ymax = y + h // 2
                        xmin = np.where(xmin <= 0, 1, xmin)
                        ymin = np.where(ymin <= 0, 1, ymin)
                        xmax = np.where(xmax >= img_width, img_width - 1, xmax)
                        ymax = np.where(ymax >= img_height, img_height - 1, ymax)
                        if ymax<=ymin or xmax<=xmin:continue
                        self.annotations.append(
                            [os.path.join('/media/lihansheng/5603d8f3-704d-4241-903d-fa30f6c8f69c/waikuai/gongrenfuzhuang/data_fullannotation/split/val/image', self.filen_ame), xmin, ymin, xmax, ymax,
                             self.supercategory])
            except:
                continue

        sys.stdout.write('\n')
        sys.stdout.flush()

    def write_file(self, ):
        with open(self.ann_path, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(self.annotations)

        class_name = sorted(self.label)
        class_ = []
        for num, name in enumerate(class_name):
            class_.append([name, num])
        with open(self.classes_path, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(class_)


txt_file = glob.glob('/media/lihansheng/5603d8f3-704d-4241-903d-fa30f6c8f69c/waikuai/gongrenfuzhuang/data_fullannotation/split/val/txt/*.txt')
def from_file_move_image(source,target_root):
    if not os.path.exists(os.path.join(target_root,'txt')):
        os.makedirs(os.path.join(target_root,'txt'))
    if not os.path.exists(os.path.join(target_root,'image')):
        os.makedirs(os.path.join(target_root,'image'))
    txt_file=[]
    for file in os.listdir(source):
        type=file.split('.')[-1]
        if type=='txt':
            txt_file.append(file)
    for txt in txt_file:
        name=txt.split('.')[0]
        source_img_path=os.path.join(source,name+'.jpg')
        if not os.path.exists(source_img_path):
            print('missing!!!!!continue')
            continue
        source_txt_path=os.path.join(source,txt)
        target_img_path=os.path.join(target_root,'image',name+'.jpg')
        target_txt_path = os.path.join(target_root, 'txt', name + '.txt')
        shutil.copy(source_img_path,target_img_path)
        shutil.copy(source_txt_path, target_txt_path)
def count(img_dir,txt_dir):
   missing = {}
   img_num=len(os.listdir(img_dir))
   txt_num = len(os.listdir(txt_dir))
   print('num_images:%d\tnum_txts:%d'%(img_num,txt_num))
   for img in os.listdir(img_dir):
       file_name=img.split('.')[0]
       type=file_name.split('_')[0]
       if type not in missing.keys():
           missing[type]={}
           missing[type]['images']=0
           missing[type]['txt'] = 0
       missing[type]['images'] += 1
       txt_path=os.path.join(txt_dir,file_name+'.txt')
       if os.path.exists(txt_path):
           missing[type]['txt']+=1
   for key in missing.keys():
       print('%s has %d images and %d annotates'%(key,int(missing[key]['images']),int(missing[key]['txt'])))
# from_file_move_image('/media/lihansheng/移动固态硬盘/lajifenleishuju/label_extra/label_bull','/media/lihansheng/5603d8f3-704d-4241-903d-fa30f6c8f69c/waikuai/lajifenlei/data')
PascalVOC2CSV(txt_file)
# count('/media/lihansheng/5603d8f3-704d-4241-903d-fa30f6c8f69c/waikuai/lajifenlei/data/image','/media/lihansheng/5603d8f3-704d-4241-903d-fa30f6c8f69c/waikuai/lajifenlei/data/txt')
