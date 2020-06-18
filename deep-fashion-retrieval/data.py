# -*- coding: utf-8 -*-

import torch.utils.data as data
import torch
from config import *
import os
from PIL import Image
import random

ALLOWED_CATEGORIES = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 26, 29, 32, 34, 35, 36, 39, 47]
CATEGORY_TO_INDEX = {cat: i for i, cat in enumerate(ALLOWED_CATEGORIES)}

class Fashion_attr_prediction(data.Dataset):
    def __init__(self, type="train", transform=None, target_transform=None, crop=False, img_path=None, custom=False):
        # print(f'init called with img_path {img_path}')
        self.transform = transform
        self.target_transform = target_transform
        self.crop = crop
        # type_all = ["train", "test", "all", "triplet", "single"]
        # print(f'Custom2: {custom}')
        # print(f'Crop: {crop}')
        self.type = type
        if type == "single":
            self.img_path = img_path
            # print(f'img_path in init: {img_path}')
            if custom:
                self.bbox = dict()
                self.read_bbox(custom=True)
            return
        self.train_list = []
        self.train_dict = {i: [] for i in range(CATEGORIES)}
        # self.train_dict = {i: [] for i in [x - 1 for x in ALLOWED_CATEGORIES]}
        self.test_list = []
        self.all_list = []
        self.bbox = dict()
        self.anno = dict()

        if type == "all" and custom:
            self.read_bbox(scrapped=True)
            self.all_list = list(self.bbox.keys())
            print(f'all_list: {self.all_list}')
        else:
            self.read_partition_category()
            self.read_bbox()


        # if not custom:
        #     self.read_partition_category()
        # else:
        #     self.read_scrapped()  # read list of scrapped imgs into self.all_list


    def __len__(self):
        if self.type == "all":
            return len(self.all_list)
        elif self.type == "train":
            return len(self.train_list)
        elif self.type == "test":
            return len(self.test_list)
        else:
            return 1

    def read_partition_category(self):
        list_eval_partition = os.path.join(DATASET_BASE, r'Eval', r'list_eval_partition.txt')
        list_category_img = os.path.join(DATASET_BASE, r'Anno', r'list_category_img.txt')
        partition_pairs = self.read_lines(list_eval_partition)
        category_img_pairs = self.read_lines(list_category_img)
        for k, v in category_img_pairs:
            v = int(v)
            self.anno[k] = v - 1
            # Uncomment section if you only want to train on first 20 categories
            # if v <= 20:
            #     self.anno[k] = v - 1
            # Uncomment section if you only want to train on ALLOWED_CATEGORIES
            # if v in ALLOWED_CATEGORIES:
            #     # self.anno[k] = v - 1    # image_name: category_id-1
            #     self.anno[k] = CATEGORY_TO_INDEX[v]
            #     # print(f'v was {v}, saved val is {self.anno[k]}')
        for k, v in partition_pairs:
            if k in self.anno:
                if v == "train":
                    self.train_list.append(k)
                    self.train_dict[self.anno[k]].append(k)
                else:
                    # Test and Val
                    self.test_list.append(k)
        # a = {cat: len(self.train_dict[cat]) for cat in self.train_dict}
        # print(f'train_dict: {a}')
        self.all_list = self.test_list + self.train_list
        # print('train_list: ', self.train_list)
        # print('test_list: ', self.test_list)
        # print('all_list: ', self.all_list)
        random.shuffle(self.train_list)
        random.shuffle(self.test_list)
        random.shuffle(self.all_list)

    def read_bbox(self, custom=False, scrapped=False):
        if custom:
            list_bbox = os.path.join(DATASET_BASE, r'list_bbox_custom.txt')
        elif scrapped:
            list_bbox = os.path.join(DATASET_BASE, r'scrapped', 'list_bbox_scrapped.txt')
        else:
            list_bbox = os.path.join(DATASET_BASE, r'Anno', r'list_bbox.txt')
            # list_bbox = os.path.join(DATASET_BASE, r'list_bbox.txt')
        pairs = self.read_lines(list_bbox)
        for k, x1, y1, x2, y2 in pairs:
            img_full_path = os.path.join(DATASET_BASE, k)
            self.bbox[img_full_path] = [x1, y1, x2, y2]
        # print(f'bbox: {self.bbox}')

    def read_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()[2:]
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
        return pairs

    def read_crop(self, img_path):
        # print(f'read_crop called with img_path {img_path}')
        img_full_path = os.path.join(DATASET_BASE, img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if self.crop:
            # x1, y1, x2, y2 = self.bbox[img_path]
            if img_path in self.bbox:
                x1, y1, x2, y2 = [int(x) for x in self.bbox[img_path]]
            else:
                x1, y1, x2, y2 = 0, 0, img.size[0], img.size[1]
            # print(x1, y1, x2, y2)
            if x1 < x2 <= img.size[0] and y1 < y2 <= img.size[1]:
                img = img.crop((x1, y1, x2, y2))
                # print(f'img {img_path} cropped to size {img.size}')
        return img

    def __getitem__(self, index):
        # print(f'getitem called with index {index}')
        if self.type == "triplet":
            img_path = self.train_list[index]
            target = self.anno[img_path]
            img_p = random.choice(self.train_dict[target])
            # print('train_dict:')
            # for k, v in self.train_dict.items():
            #         print(f'{k}: {len(v)}')
            # print(f'choice: {random.choice(list(filter(lambda x: x != target, range(20))))}')
            img_n = random.choice(self.train_dict[random.choice(list(filter(lambda x: x != target, range(20))))])
            # print(f'img_n: {img_n}')
            img = self.read_crop(img_path)
            img_p = self.read_crop(img_p)
            img_n = self.read_crop(img_n)
            if self.transform is not None:
                img = self.transform(img)
                img_p = self.transform(img_p)
                img_n = self.transform(img_n)
            return img, img_p, img_n

        if self.type == "single":
            # print('in self.type == single')
            img_path = self.img_path
            img = self.read_crop(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img

        if self.type == "all":
            img_path = self.all_list[index]
        elif self.type == "train":
            img_path = self.train_list[index]
        else:
            img_path = self.test_list[index]
        # target = self.anno[img_path]
        img = self.read_crop(img_path)

        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # Adding to make scrapped data work:
        if self.type != "all":
            target = self.anno[img_path]
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, img_path if self.type == "all" else target


class Fashion_inshop(data.Dataset):
    def __init__(self, type="train", transform=None):
        self.transform = transform
        self.type = type
        self.train_dict = {}
        self.test_dict = {}
        self.test_query_dict = {}
        self.test_gallery_dict = {}
        self.train_list = []  # Stores item ids eg id_00000001.
        self.test_list = []
        # self.test_query_list = []
        self.test_query_paths_list = []  # List of paths of query imgs
        self.test_gallery_list = []
        self.all_path = []
        self.cloth = self.readcloth()
        self.read_train_test()

    def read_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()[2:]
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
        return pairs

    def readcloth(self):
        lines = self.read_lines(os.path.join(DATASET_BASE, 'in_shop', 'list_bbox_inshop.txt'))
        if CATEGORIES == 20:  # Only read tops
            valid_lines = list(filter(lambda x: x[1] == '1', lines))
            names = set(list(map(lambda x: x[0], valid_lines)))
        else:  # Read clothes of all 3 types (upper-body clothes, lower-body clothes, full-body clothes)
            names = set(x[0] for x in lines)
        return names

    def read_train_test(self):
        lines = self.read_lines(os.path.join(DATASET_BASE, 'in_shop', 'list_eval_partition.txt'))
        valid_lines = list(filter(lambda x: x[0] in self.cloth, lines))
        for line in valid_lines:
            # s = self.train_dict if line[2] == 'train' else self.test_dict
            if line[2] == 'train':
                s = self.train_dict
            elif line[2] == 'query':
                s = self.test_query_dict
            else:
                s = self.test_gallery_dict
            if line[1] not in s:
                s[line[1]] = [line[0]]  # item_id: list of img_paths
            else:
                s[line[1]].append(line[0])
        # print(f"self.test_query_dict: {self.test_query_dict}")
        # print(f"self.test_gallery_dict: {self.test_gallery_dict}")

        def clear_single(d):  # If only single image for id, delete that id from dict
            keys_to_delete = []
            for k, v in d.items():
                if len(v) < 2:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                d.pop(k, None)
        clear_single(self.train_dict)
        # clear_single(self.test_dict)
        # self.train_list, self.test_list = list(self.train_dict.keys()), list(self.test_dict.keys())
        self.train_list = list(self.train_dict.keys())
        # self.test_query_list = list(self.test_query_dict.keys())
        self.test_gallery_list = list(self.test_gallery_dict.keys())
        # print(f"len(self.test_query_list): {len(self.test_query_list)}")
        # print(f"len(self.test_gallery_list): {len(self.test_gallery_list)}")

        for v in self.train_dict.values():
            self.all_path += v
        self.train_len = len(self.all_path)
        # print(f"self.train_len{self.train_len}")
        # for v in list(self.test_dict.values()):
        #     self.all_path += v
        for v in self.test_query_dict.values():
            self.all_path += v
            self.test_query_paths_list += v
        self.test_query_len = len(self.all_path) - self.train_len
        # print(f"self.test_query_len{self.test_query_len}")
        for v in self.test_gallery_dict.values():
            self.all_path += v
        self.test_gallery_len = len(self.all_path) - self.train_len - self.test_query_len
        # print(f"self.test_gallery_len{self.test_gallery_len}")
        # self.test_len = len(self.all_path) - self.train_len

    def process_img(self, img_path):
        img_full_path = os.path.join(DATASET_BASE, 'in_shop', img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        if self.type == 'train':
            return len(self.train_list)
        elif self.type == 'test':
            # return len(self.test_list)
            return len(self.test_query_paths_list)
        else:
            return len(self.all_path)

    def __getitem__(self, item):
        if self.type == 'all':
            img_path = self.all_path[item]
            img = self.process_img(img_path)
            return img, img_path
        # s_d = self.train_dict if self.type == 'train' else self.test_dict
        # s_l = self.train_list if self.type == 'train' else self.test_list
        elif self.type == 'train':
            s_d = self.train_dict
            s_l = self.train_list
            imgs = s_d[s_l[item]]
            img_triplet = random.sample(imgs, 2)
            img_other_id = random.choice(list(range(0, item)) + list(range(item + 1, len(s_l))))
            img_other = random.choice(s_d[s_l[img_other_id]])
            img_triplet.append(img_other)
            return list(map(self.process_img, img_triplet))  # Returns list of 2 +ve & 1 -ve img.
        else:
            # For self.type == 'test'
            img_path = self.test_query_paths_list[item]
            img = self.process_img(img_path)
            return img, img_path


# class Fashion_inshop(data.Dataset):
#     def __init__(self, type="train", transform=None):
#         self.transform = transform
#         self.type = type
#         self.train_dict = {}
#         self.test_dict = {}
#         self.train_list = []  # Stores item ids eg id_00000001.
#         self.test_list = []  # Stores item ids eg id_00000001.
#         self.all_path = []
#         self.cloth = self.readcloth()
#         self.read_train_test()
#
#     def read_lines(self, path):
#         with open(path) as fin:
#             lines = fin.readlines()[2:]
#             lines = list(filter(lambda x: len(x) > 0, lines))
#             pairs = list(map(lambda x: x.strip().split(), lines))
#         return pairs
#
#     def readcloth(self):
#         lines = self.read_lines(os.path.join(DATASET_BASE, 'in_shop', 'list_bbox_inshop.txt'))
#         if CATEGORIES == 20:  # Only read tops
#             valid_lines = list(filter(lambda x: x[1] == '1', lines))
#             names = set(list(map(lambda x: x[0], valid_lines)))
#         else:  # Read clothes of all 3 types (upper-body clothes, lower-body clothes, full-body clothes)
#             names = set(x[0] for x in lines)
#         return names
#
#     # def read_train_test(self):
#     #     lines = self.read_lines(os.path.join(DATASET_BASE, 'in_shop', 'list_eval_partition.txt'))
#     #     valid_lines = list(filter(lambda x: x[0] in self.cloth, lines))
#     #     for line in valid_lines:
#     #         s = self.train_dict if line[2] == 'train' else self.test_dict
#     #         if line[1] not in s:
#     #             s[line[1]] = [line[0]]
#     #         else:
#     #             s[line[1]].append(line[0])
#
#     def read_train_test(self):
#         lines = self.read_lines(os.path.join(DATASET_BASE, 'in_shop', 'list_eval_partition.txt'))
#         valid_lines = list(filter(lambda x: x[0] in self.cloth, lines))
#         for line in valid_lines:
#             s = self.train_dict if line[2] == 'train' else self.test_dict
#             if line[1] not in s:
#                 s[line[1]] = [line[0]]  # item_id: list of img_paths
#             else:
#                 s[line[1]].append(line[0])
#
#         def clear_single(d):  # If only single image for id, delete that id from dict
#             keys_to_delete = []
#             for k, v in d.items():
#                 if len(v) < 2:
#                     keys_to_delete.append(k)
#             for k in keys_to_delete:
#                 d.pop(k, None)
#         clear_single(self.train_dict)
#         clear_single(self.test_dict)
#         self.train_list, self.test_list = list(self.train_dict.keys()), list(self.test_dict.keys())
#         for v in list(self.train_dict.values()):
#             self.all_path += v
#         self.train_len = len(self.all_path)
#         for v in list(self.test_dict.values()):
#             self.all_path += v
#         self.test_len = len(self.all_path) - self.train_len
#
#     def process_img(self, img_path):
#         img_full_path = os.path.join(DATASET_BASE, 'in_shop', img_path)
#         with open(img_full_path, 'rb') as f:
#             with Image.open(f) as img:
#                 img = img.convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         return img
#
#     def __len__(self):
#         if self.type == 'train':
#             return len(self.train_list)
#         elif self.type == 'test':
#             return len(self.test_list)
#         else:
#             return len(self.all_path)
#
#     def __getitem__(self, item):
#         if self.type == 'all':
#             img_path = self.all_path[item]
#             img = self.process_img(img_path)
#             return img, img_path
#         s_d = self.train_dict if self.type == 'train' else self.test_dict
#         s_l = self.train_list if self.type == 'train' else self.test_list
#         imgs = s_d[s_l[item]]
#         img_triplet = random.sample(imgs, 2)
#         img_other_id = random.choice(list(range(0, item)) + list(range(item + 1, len(s_l))))
#         img_other = random.choice(s_d[s_l[img_other_id]])
#         img_triplet.append(img_other)
#         return list(map(self.process_img, img_triplet))  # Returns list of 2 +ve & 1 -ve img.
