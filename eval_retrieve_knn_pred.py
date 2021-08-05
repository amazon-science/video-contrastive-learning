"""The functions for VCLR video retrieval downstream (train a kNN)
Code partially borrowed from
https://github.com/YihengZhang-CV/SeCo-Sequence-Contrastive-Learning/blob/main/eval_svm_feature_perf.py.

MIT License
Copyright (c) 2020 YihengZhang-CV
"""


import numpy as np
import argparse
import os
import json

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def main():
    parser = argparse.ArgumentParser('retrieval eval')
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--trainsplit', type=str, required=True)
    parser.add_argument('--valsplit', type=str, required=True)
    parser.add_argument('--num_replica', type=int, default=8)
    parser.add_argument('--data-source', type=str)
    args = parser.parse_args()

    for i in range(args.num_replica):
        os.path.exists(os.path.join(args.output_dir, 'feature_{}_{}.npy'.format(args.trainsplit, i)))
        os.path.exists(os.path.join(args.output_dir, 'feature_{}_cls_{}.npy'.format(args.trainsplit, i)))
        os.path.exists(os.path.join(args.output_dir, 'feature_{}_{}.npy'.format(args.valsplit, i)))
        os.path.exists(os.path.join(args.output_dir, 'feature_{}_cls_{}.npy'.format(args.valsplit, i)))
        os.path.exists(os.path.join(args.output_dir, 'vid_num_{}.npy'.format(args.trainsplit)))
        os.path.exists(os.path.join(args.output_dir, 'vid_num_{}.npy'.format(args.valsplit)))

    vid_num_train = np.load(os.path.join(args.output_dir, 'vid_num_{}.npy'.format(args.trainsplit)))
    train_padding_num = vid_num_train[0] % args.num_replica
    vid_num_val = np.load(os.path.join(args.output_dir, 'vid_num_{}.npy'.format(args.valsplit)))
    val_padding_num = vid_num_val[0] % args.num_replica

    feat_train = []
    feat_train_cls = []
    for i in range(args.num_replica):
        feat_train.append(np.load(os.path.join(args.output_dir, 'feature_{}_{}.npy'.format(args.trainsplit, i))))
        feat_train_cls.append(
            np.load(os.path.join(args.output_dir, 'feature_{}_cls_{}.npy'.format(args.trainsplit, i))))
    if train_padding_num > 0:
        for i in range(train_padding_num, args.num_replica):
            feat_train[i] = feat_train[i][:-1, :]
            feat_train_cls[i] = feat_train_cls[i][:-1]
    feat_train = np.concatenate(feat_train, axis=0).squeeze()
    feat_train_cls = np.concatenate(feat_train_cls, axis=0).squeeze()
    print('feat_train: {}'.format(feat_train.shape))
    print('feat_train_cls: {}'.format(feat_train_cls.shape))

    feat_val = []
    feat_val_cls = []
    for i in range(args.num_replica):
        feat_val.append(np.load(os.path.join(args.output_dir, 'feature_{}_{}.npy'.format(args.valsplit, i))))
        feat_val_cls.append(np.load(os.path.join(args.output_dir, 'feature_{}_cls_{}.npy'.format(args.valsplit, i))))
    if val_padding_num > 0:
        for i in range(val_padding_num, args.num_replica):
            feat_val[i] = feat_val[i][:-1, :]
            feat_val_cls[i] = feat_val_cls[i][:-1]
    feat_val = np.concatenate(feat_val, axis=0)
    feat_val_cls = np.concatenate(feat_val_cls, axis=0)
    print('feat_val: {}'.format(feat_val.shape))
    print('feat_val_cls: {}'.format(feat_val_cls.shape))

    # kNN retrieval
    if args.valsplit == 'test':
        ks = [3]
    else:
        ks = [1, 5, 10, 20, 50]
    topk_correct = {k: 0 for k in ks}

    class_top = 1
    if args.data_source == 'ucf':
        class_num = 101
    elif args.data_source == 'hmdb':
        class_num = 51
    else:
        raise Exception('The data-source argument no assigned!')
    class_correct = {cls: 0 for cls in range(0, class_num)}
    class_total = {cls: 0 for cls in range(0, class_num)}

    X_train = feat_train
    y_train = feat_train_cls
    X_test = feat_val
    y_test = feat_val_cls

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)

    for k in ks:
        # print(k)
        top_k_indices = indices[:, :k]
        if args.valsplit == 'test':
            print(top_k_indices)
            np.save(os.path.join(args.output_dir, 'top_k_indices.npy'), top_k_indices)
        # print(top_k_indices.shape, y_test.shape)
        for ind, test_label in zip(top_k_indices, y_test):
            labels = y_train[ind]
            if test_label in labels:
                # print(test_label, labels)
                topk_correct[k] += 1
                if k == class_top:
                    class_correct[test_label] += 1
            if k == class_top:
                class_total[test_label] += 1

    for k in ks:
        correct = topk_correct[k]
        total = len(X_test)
        print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct / total))

    # save label
    if args.valsplit != 'test':
        label_file = os.path.join(args.output_dir, 'class_retrieval_vclr.txt')
        f = open(label_file, 'w')
        for k in class_correct.keys():
            correct = class_correct[k]
            total = class_total[k]
            info = 'Classs-{}, Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(
                k, class_top, correct, total, correct / total)
            print(info)
            f.write(info)
            f.write('\n')
        f.close()

    with open(os.path.join(args.output_dir, 'topk_correct.json'), 'w') as fp:
        json.dump(topk_correct, fp)


if __name__ == '__main__':
    main()
