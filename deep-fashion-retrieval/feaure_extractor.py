# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from config import *
from utils import *
from torch.autograd import Variable
from data import Fashion_attr_prediction, Fashion_inshop
from net import f_model, c_model, p_model


main_model = f_model(model_path=DUMPED_MODEL).cuda(GPU_ID)
color_model = c_model().cuda(GPU_ID)
pooling_model = p_model().cuda(GPU_ID)
# extractor = FeatureExtractor(main_model, color_model, pooling_model)
extractor = FeatureExtractorWithClassification(main_model, color_model, pooling_model)


# def dump_dataset(loader, deep_feats, color_feats, labels):
#     for batch_idx, (data, data_path) in enumerate(loader):
#         print(f'batch_idx: {batch_idx}')
#         print(f'data.shape: {data.shape}')
#         print(f'len(data_path): {len(data_path)}')

#         data = Variable(data).cuda(GPU_ID)
#         deep_feat, color_feat = extractor(data)
#         print(f'deep_feat.shape: {deep_feat.shape}')
#         print(f'color_feat: {color_feat.shape}')
#         for i in range(len(data_path)):
#             path = data_path[i]
#             feature_n = deep_feat[i].squeeze()
#             color_feature_n = color_feat[i]
#             # dump_feature(feature, path)

#             deep_feats.append(feature_n)
#             color_feats.append(color_feature_n)
#             labels.append(path)

#         if batch_idx % LOG_INTERVAL == 0:
#             print("{} / {}".format(batch_idx * EXTRACT_BATCH_SIZE, len(loader.dataset)))

def dump_dataset(loader, classes, deep_feats, color_feats, labels, allowed_inds=None):
    for batch_idx, (data, data_path) in enumerate(loader):
        # print(f'batch_idx: {batch_idx}')
        # print(f'data.shape: {data.shape}')
        # print(f'data_path[0]: {data_path[0]}')

        list_bbox_eastern = os.path.join(DATASET_BASE, r'scrapped', 'list_bbox_scrapped_eastern.txt')
        with open(list_bbox_eastern) as fin:
            lines = fin.readlines()[2:]
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
            eastern_pths = set(os.path.join(DATASET_BASE, pair[0]) for pair in pairs)

        list_btd = os.path.join(DATASET_BASE, r'scrapped', 'list_bbox_scrapped_btd.txt')
        with open(list_btd) as fin:
            lines = fin.readlines()[2:]
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
            btd_pths = set(os.path.join(DATASET_BASE, line.strip()) for line in lines)
            print(f'btd_paths: {btd_pths}')

        data = Variable(data).cuda(GPU_ID)
        cls, deep_feat, color_feat = extractor(data)  # color_feat is list ndarrays, other two are ndarrays
        # print(f'cls.shape: {cls.shape}')
        # print(f'deep_feat.shape: {deep_feat.shape}')
        # print(f'color_feat[0] type: {type(color_feat[0])}')
        # print(f'color_feat[0] shape: {color_feat[0].shape}')

        for i in range(len(data_path)):
            path = data_path[i]

            if data_path[i] in eastern_pths:
                # print(f'{data_path[i]} in eastern_pths')
                class_n = 38
            elif data_path[i] in btd_pths:
                class_n = 5
            elif allowed_inds is not None:
                class_n = allowed_inds[cls[i][allowed_inds].argmax()] + 1
            else:
                class_n = cls[i].argmax() + 1  # +1 because index starts from 0 but category labels start from 1

            # print(f'cls[i]: {cls[i]}')
            # print(f'cls[i] argmax+1: {cls[i].argmax() + 1}')
            # print(f'cls[i][allowed_inds]: {cls[i][allowed_inds]}')
            # print(f'cls[i][allowed_inds].argmax() + 1: {cls[i][allowed_inds].argmax() + 1}')
            # print(f'allowed_inds[cls[i][allowed_inds].argmax()]: {allowed_inds[cls[i][allowed_inds].argmax()]}')
            feature_n = deep_feat[i].squeeze()
            color_feature_n = color_feat[i]
            # dump_feature(feature, path)

            classes.append(class_n)
            deep_feats.append(feature_n)
            color_feats.append(color_feature_n)
            labels.append(path)

        if batch_idx % LOG_INTERVAL == 0:
            print("{} / {}".format(batch_idx * EXTRACT_BATCH_SIZE, len(loader.dataset)))


def dump(custom=False):
    print(f'dump function called with custom: {custom}')

    if not custom:
        all_loader = torch.utils.data.DataLoader(
            Fashion_attr_prediction(type="all", transform=data_transform_test),
            batch_size=EXTRACT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
        )
        classes = []
        deep_feats = []
        color_feats = []
        labels = []
        # dump_dataset(all_loader, deep_feats, color_feats, labels)
        dump_dataset(all_loader, classes, deep_feats, color_feats, labels)

        if ENABLE_INSHOP_DATASET:
            inshop_loader = torch.utils.data.DataLoader(
                Fashion_inshop(type="all", transform=data_transform_test),
                batch_size=EXTRACT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
            )
            dump_dataset(inshop_loader, classes, deep_feats, color_feats, labels)

        feat_all = os.path.join(DATASET_BASE, 'all_feat.npy')
        color_feat_all = os.path.join(DATASET_BASE, 'all_color_feat.npy')
        feat_list = os.path.join(DATASET_BASE, 'all_feat.list')
        with open(feat_list, "w") as fw:
            fw.write("\n".join(labels))
        np.save(feat_all, np.vstack(deep_feats))
        np.save(color_feat_all, np.vstack(color_feats))
        print("Dumped to all_feat.npy, all_color_feat.npy and all_feat.list.")

    else:
        all_loader = torch.utils.data.DataLoader(
            Fashion_attr_prediction(type="all", transform=data_transform_test, custom=True, crop=True),
            batch_size=EXTRACT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
        )
        classes = []
        deep_feats = []
        color_feats = []
        labels = []

        # Classes with items in the dataset
        allowed_inds = np.array([ 4,  9, 10, 15, 16, 17, 18, 21, 25, 28, 31, 33, 37])

        dump_dataset(all_loader, classes, deep_feats, color_feats, labels, allowed_inds)

        class_all = os.path.join(DATASET_BASE, 'custom_all_class.npy')
        feat_all = os.path.join(DATASET_BASE, 'custom_all_feat.npy')
        color_feat_all = os.path.join(DATASET_BASE, 'custom_all_color_feat.npy')
        feat_list = os.path.join(DATASET_BASE, 'custom_all_feat.list')
        with open(feat_list, "w") as fw:
            fw.write("\n".join(labels))

        np.save(class_all, np.vstack(classes))
        np.save(feat_all, np.vstack(deep_feats))
        np.save(color_feat_all, np.vstack(color_feats))
        print("Dumped to custom_all_class.npy, custom_all_feat.npy, custom_all_color_feat.npy and custom_all_feat.list.")


def get_inshop_test_db():
    print("Extracting features of in-shop test images.")
    db_dict = dict()  # dataset_type: dataset_db. dataset_db is (deep_feats, color feats, labels)

    for dataset_type in ("test_gallery", "test_query"):
        dataset = Fashion_inshop(type=dataset_type, transform=data_transform_test)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=EXTRACT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
        )

        classes = []
        deep_feats = []
        color_feats = []
        labels = []
        dump_dataset(loader, classes, deep_feats, color_feats, labels)
        db_dict[dataset_type] = (deep_feats, color_feats, labels)

    deep_feats, color_feats, labels = db_dict['test_query']
    length = dataset.test_query_len
    deep_feats, color_feats, labels = deep_feats[-length:], color_feats[-length:], labels[-length:]
    query_feat_dict = {labels[i]: (deep_feats[i], color_feats[i]) for i in range(len(labels))}

    deep_feats, color_feats, labels = db_dict['test_gallery']
    length = dataset.test_gallery_len
    deep_feats, color_feats, labels = deep_feats[-length:], color_feats[-length:], labels[-length:]

    return query_feat_dict, (deep_feats, color_feats, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrapped", help="extract features from scrapped dataset rather than deepfashion",
        action="store_true")
    args = parser.parse_args()
    if args.scrapped:
        print("Extracting features from scrapped dataset.")
        dump(custom=True)
    else:
        print("Extracting features from deepfashion dataset.")
        dump()


