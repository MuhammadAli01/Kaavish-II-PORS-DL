# -*- coding: utf-8 -*-

import os
import argparse
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

def dump_dataset(loader, classes, deep_feats, color_feats, labels):
    for batch_idx, (data, data_path) in enumerate(loader):
        # print(f'batch_idx: {batch_idx}')
        # print(f'data.shape: {data.shape}')
        # print(f'len(data_path): {len(data_path)}')

        data = Variable(data).cuda(GPU_ID)
        cls, deep_feat, color_feat = extractor(data)  # color_feat is list ndarrays, other two are ndarrays
        # print(f'cls.shape: {cls.shape}')
        # print(f'deep_feat.shape: {deep_feat.shape}')
        # print(f'color_feat[0] type: {type(color_feat[0])}')
        # print(f'color_feat[0] shape: {color_feat[0].shape}')
        for i in range(len(data_path)):
            path = data_path[i]
            class_n = cls[i].argmax() + 1  # +1 because index starts from 0 but category labels start from 1
            # print(f'cls[i] argmax+1: {cls[i].argmax() + 1}')
            feature_n = deep_feat[i].squeeze()
            color_feature_n = color_feat[i]
            # dump_feature(feature, path)

            classes.append(class_n)
            deep_feats.append(feature_n)
            color_feats.append(color_feature_n)
            labels.append(path)

        if batch_idx % LOG_INTERVAL == 0:
            print("{} / {}".format(batch_idx * EXTRACT_BATCH_SIZE, len(loader.dataset)))


def dump(custom=False, inshop_test=False):
    print(f'dump function called with custom: {custom} and inshop_test: {inshop_test}')
    if inshop_test:
        all_loader = torch.utils.data.DataLoader(
            Fashion_attr_prediction(type="all", transform=data_transform_test, custom=True, crop=True),
            batch_size=EXTRACT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
        )
        classes = []
        deep_feats = []
        color_feats = []
        labels = []
        dump_dataset(all_loader, classes, deep_feats, color_feats, labels)

        class_all = os.path.join(DATASET_BASE, 'custom_all_class.npy')
        feat_all = os.path.join(DATASET_BASE, 'custom_all_feat.npy')
        color_feat_all = os.path.join(DATASET_BASE, 'custom_all_color_feat.npy')
        feat_list = os.path.join(DATASET_BASE, 'custom_all_feat.list')
        with open(feat_list, "w") as fw:
            fw.write("\n".join(labels))

        np.save(class_all, np.vstack(classes))
        np.save(feat_all, np.vstack(deep_feats))
        np.save(color_feat_all, np.vstack(color_feats))
        print(
            "Dumped to custom_all_class.npy, custom_all_feat.npy, custom_all_color_feat.npy and custom_all_feat.list.")

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
        dump_dataset(all_loader, classes, deep_feats, color_feats, labels)

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


