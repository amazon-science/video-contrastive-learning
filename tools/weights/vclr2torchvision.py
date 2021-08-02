import argparse
import torch
import torchvision

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Transfer backbone format from VCLR style to Torchvision style")
    parser.add_argument('--vclr-weights', type=str,
                        help='the directory of raw videos')
    args = parser.parse_args()

    ######################## VCLR Pretrained Weights ########################
    vclr_params = torch.load(args.vclr_weights, map_location='cpu')['model']

    for k, v in vclr_params.items():
        print(k)

    ######################## torchvision ResNet50 ########################
    torch_res50 = torchvision.models.resnet50(pretrained=False)
    torch_params = torch_res50.state_dict()

    for key in torch_params.keys():
        print(key)

    print(len(torch_params.keys()))

    ######################## Transfer Weights ########################
    vclr_param_keys = list(vclr_params.keys())[:318]
    torch_param_keys = list(torch_params.keys())[:318]

    print(len(vclr_param_keys))
    print(len(torch_param_keys))

    idx = 0
    for k1, k2 in zip(torch_param_keys, vclr_param_keys):
        print(k1, "<================>", k2)
        torch_params[k1] = vclr_params[k2]
        idx += 1

    ######################## Save MMaction2 Weights ########################
    torch.save(torch_params, 'vclr_torch.pth')
