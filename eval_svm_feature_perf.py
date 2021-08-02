import numpy as np
import argparse
import os
import liblinear.liblinearutil as liblinearsvm


def main():
    parser = argparse.ArgumentParser('svm_perf')
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--trainsplit', type=str, required=True)
    parser.add_argument('--valsplit', type=str, required=True)
    parser.add_argument('--num_replica', type=int, default=8)
    parser.add_argument('--cost', type=float, default=1.0)
    parser.add_argument('--primal', action='store_true', default=False)
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
        feat_train_cls.append(np.load(os.path.join(args.output_dir, 'feature_{}_cls_{}.npy'.format(args.trainsplit, i))))
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

    print('form svm problem')
    svm_problem = liblinearsvm.problem(feat_train_cls, feat_train)
    if args.primal:
        print('L2-regularized L2-loss support vector classification (primal), cost={}'.format(args.cost))
        svm_parameter = liblinearsvm.parameter('-s 2 -n 32 -c {}'.format(args.cost))
        svm_filename = 'multicore_linearsvm_primal_c{}.svmmodel'.format(args.cost)
    else:
        print('L2-regularized L2-loss support vector classification (dual), cost={}'.format(args.cost))
        svm_parameter = liblinearsvm.parameter('-s 1 -n 32 -c {}'.format(args.cost))
        svm_filename = 'multicore_linearsvm_dual_c{}.svmmodel'.format(args.cost)
    print('train svm')
    svm_model = liblinearsvm.train(svm_problem, svm_parameter)
    print('save svm')
    liblinearsvm.save_model(os.path.join(args.output_dir, svm_filename), svm_model)
    print('eval svm')
    pd_label, pd_acc, pd_val = liblinearsvm.predict(feat_val_cls, feat_val, svm_model)
    eval_acc, eval_mse, eval_scc = liblinearsvm.evaluations(feat_val_cls, pd_label)
    print('{}/{}'.format(pd_acc, eval_acc))
    with open(os.path.join(args.output_dir, svm_filename + '.txt'), 'w') as f:
        f.write('{}/{}'.format(pd_acc, eval_acc))
    print('Done')


if __name__ == '__main__':
    main()