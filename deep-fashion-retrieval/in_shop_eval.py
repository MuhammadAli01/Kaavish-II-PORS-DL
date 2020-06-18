# -*- coding: utf-8 -*-

import random

from data import Fashion_inshop
from retrieval import load_feat_db, get_deep_color_top_n


def eval(retrieval_top_n=10):
    dataset = Fashion_inshop()
    length = dataset.test_len
    deep_feats, color_feats, labels = load_feat_db()
    deep_feats, color_feats, labels = deep_feats[-length:], color_feats[-length:], labels[-length:]
    feat_dict = {labels[i]: (deep_feats[i], color_feats[i]) for i in range(len(labels))}

    include_once = 0
    include_zero = 0
    include_times = 0
    should_include_times = 0

    count_retrieved = 0
    for iter_id, item_id in enumerate(dataset.test_list):
        print(f'item_id: {item_id}')
        item_imgs = dataset.test_dict[item_id]
        print(f'item_imgs: {item_imgs}')  # list of imgs of item with id item_id.
        item_img = random.choice(item_imgs)
        result = get_deep_color_top_n(feat_dict[item_img], deep_feats, color_feats, labels, retrieval_top_n)
        print(f'result: {result}')
        keys = list(map(lambda x: x[0], result))
        included = list(map(lambda x: x in item_imgs, keys))
        print(f'included: {included}', '\n')

        if included.count(True) >= 2:
            count_retrieved += 1
            # print('retrieved')
        # else:
            # print('retrieval failed')
        # print('\n')

        should_include_times += (len(item_imgs) - 1)
        include_once += (1 if included.count(True) >= 2 else 0)
        include_zero += (1 if included.count(True) <= 1 else 0)
        include_times += (included.count(True) - 1)

        if iter_id % 100 == 0:
            print("Progress: k={}, {}/{}".format(retrieval_top_n-1, iter_id, len(dataset.test_list)))
            # print("{}/{}, is included: {}/{}, included times: {}/{}".format(iter_id, len(dataset.test_list),
            #       include_once, include_once + include_zero,
            #       include_times, should_include_times))

    accuracy = count_retrieved / len(dataset.test_list)
    print(f'Accuracy: {accuracy}%')
    return accuracy
    # return include_times, should_include_times, include_once, include_zero


if __name__ == '__main__':
    print(eval())
