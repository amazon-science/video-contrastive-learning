import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Transfer backbone format from VCLR style to MMAction2 style")
    parser.add_argument('--mm-weights', type=str,
                        help='the template weights file of mmaction model')
    parser.add_argument('--vclr-weights', type=str,
                        help='the directory of raw videos')
    parser.add_argument('--tsm', action='store_true', default=False,
                        help='stored results for tsm!')
    args = parser.parse_args()

    ######################## VCLR Pretrained Weights ########################
    vclr_params = torch.load(args.vclr_weights, map_location='cpu')['model']

    # for k, v in vclr_params.items():
    #     print(k)

    ######################## mmaction Pretrained Weights ########################
    mm_params = torch.load(args.mm_weights, map_location='cpu')

    # for k, v in mm_param['state_dict'].items():
    #     print(k)
    # print(len(mm_param['state_dict'].keys()))

    ######################## Transfer Weights ########################
    vclr_param_keys = list(vclr_params.keys())[:318]
    mm_param_keys = list(mm_params['state_dict'].keys())[:318]

    print(len(vclr_param_keys))
    print(len(mm_param_keys))

    idx = 0
    for k1, k2 in zip(mm_param_keys, vclr_param_keys):
        print(k1, "<================>", k2)
        mm_params['state_dict'][k1] = vclr_params[k2]
        idx += 1

    ######################## Save MMaction2 Weights ########################
    if args.tsm:
        torch.save(mm_params, 'vclr_mm_tsm.pth')
    else:
        torch.save(mm_params, 'vclr_mm.pth')
