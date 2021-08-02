import numpy as np
import os
import csv
import argparse

import torchvision.transforms as transforms
from PIL import Image


def loading_ucf_lists():
    dataset_root = "/home/ubuntu/data/ucf101"
    split = 'split_1'
    # data frame root
    dataset_frame_root = os.path.join(dataset_root, 'rawframes')

    # data list file
    train_list_file = os.path.join(dataset_root, 'ucfTrainTestlist',
                                   'ucf101_' + 'train' + '_' + split + '_rawframes' + '.txt')
    test_list_file = os.path.join(dataset_root, 'ucfTrainTestlist',
                                  'ucf101_' + 'test' + '_' + split + '_rawframes' + '.txt')

    # load vid samples
    samples_train = _load_list(train_list_file, dataset_frame_root)
    samples_test = _load_list(test_list_file, dataset_frame_root)

    return samples_train, samples_test


def loading_hmdb_lists():
    dataset_root = "/home/ubuntu/data/hmdb51/"
    split = 'split_1'
    # data frame root
    dataset_frame_root = os.path.join(dataset_root, 'rawframes')

    # data list file
    train_list_file = os.path.join(dataset_root, 'testTrainMulti_7030_splits',
                                   'hmdb51_' + 'train' + '_' + split + '_rawframes' + '.txt')
    test_list_file = os.path.join(dataset_root, 'testTrainMulti_7030_splits',
                                  'hmdb51_' + 'test' + '_' + split + '_rawframes' + '.txt')

    # load vid samples
    samples_train = _load_list(train_list_file, dataset_frame_root)
    samples_test = _load_list(test_list_file, dataset_frame_root)

    return samples_train, samples_test


def _load_list(list_root, dataset_frame_root):
    with open(list_root, 'r') as f:
        lines = f.readlines()
    vids = []
    for k, l in enumerate(lines):
        lsp = l.strip().split(' ')
        # path, frame, label
        vid_root = os.path.join(dataset_frame_root, lsp[0])
        vid_root, _ = os.path.splitext(vid_root)
        # use splitetxt twice because there are some video root like: abseiling/9EnSwbXxu5g.mp4.webm
        vid_root, _ = os.path.splitext(vid_root)
        vids.append((vid_root, int(lsp[1]), int(lsp[2])))

    return vids


def _get_imgs(frame_root, frame_idx, transform):
    frame = Image.open(os.path.join(frame_root, 'img_{:05d}.jpg'.format(frame_idx)))
    frame.convert('RGB')
    frame_aug = transform(frame)

    return np.array(frame_aug)


def retrieval_imgs(samples, idx, transform):
    frame_root, frame_num, cls = samples[idx]
    frame_indices = np.round(np.linspace(1, frame_num, num=3)).astype(np.int64)

    # get query images
    imgs = []
    for frame_idx in frame_indices:
        imgs.append(_get_imgs(frame_root, frame_idx, transform))

    out_img = Image.fromarray(np.concatenate(imgs, axis=1))

    return frame_root.split('/')[7], out_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval visualization')
    parser.add_argument('--data-source', type=str)
    args = parser.parse_args()


    if args.data_source == "ucf":
        samples_train, samples_query = loading_ucf_lists()
    elif args.data_source == "hmdb":
        samples_train, samples_query = loading_hmdb_lists()
    else:
        raise Exception("Please assigne the data-source argument!")

    top_k_indices = np.load('./model/eval_retrieval/top_k_indices.npy')
    transform_list = [transforms.CenterCrop(224)]
    img_transform = transforms.Compose(transform_list)

    save_folder = './model/eval_retrieval/imgs'
    os.makedirs(save_folder, exist_ok=True)

    label_dict = dict()

    for idx, top_k in enumerate(top_k_indices):
        query_label, query = retrieval_imgs(samples_query, idx, img_transform)
        query_root = os.path.join(save_folder, query_label)
        os.makedirs(query_root, exist_ok=True)

        query.save(os.path.join(query_root, 'query.png'))

        # top k images
        top = 1
        top_k_label = []
        for topk_idx in top_k:
            key_label, key = retrieval_imgs(samples_train, topk_idx, img_transform)
            key.save(os.path.join(query_root, 'top_{}.png'.format(top)))
            top_k_label.append(key_label)
            top += 1

        label_dict[query_label] = top_k_label

    # save label
    label_file = os.path.join(save_folder, 'label_dict.txt')
    f = open(label_file, 'w')
    for k, v in label_dict.items():
        print(k, ":", v)
        f.write(k + ':' + str(v))
        f.write('\n')
    f.close()
