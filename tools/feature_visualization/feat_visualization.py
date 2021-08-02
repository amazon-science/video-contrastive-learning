import os
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision import models

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Image Preprocessing
def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(img)
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img


# calculate grad-cam, and visualization
def cam_show_img(img, feature_map, grads, out_dir, indices):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
    grads = grads.reshape([grads.shape[0], -1])
    weights = np.mean(grads, axis=1)
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam_%05d.jpg" % indices)
    cv2.imwrite(path_cam_img, cam_img)


def grad_cam(img_path, output_dir, net, indices=0):
    # acquire gradient
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    # acquire feature map
    def farward_hook(module, input, output):
        fmap_block.append(output)

    fmap_block = list()
    grad_block = list()

    net.layer4.register_forward_hook(farward_hook)
    net.layer4.register_backward_hook(backward_hook)

    img = cv2.imread(img_path, 1)
    img_input = img_preprocess(img)

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())

    # backward
    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    # generate cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # save cam
    cam_show_img(img, fmap, grads_val, output_dir, indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize Feature Map through Cam-Grad")
    parser.add_argument('--image-dir', type=str, default='./images',
                        help='the directory of raw images')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='the directory of raw videos')
    parser.add_argument('--weights-type', type=str, default='imagenet',
                        help='the type of the pretrained weights. ["random", "imagenet", "custom"]')
    parser.add_argument('--weights-path', type=str, default='vclr_torch.pth',
                        help='the path of pretrained weights if the weights_type is "custom"')
    args = parser.parse_args()

    path_imgs = []
    for img_path in sorted(os.listdir(args.image_dir)):
        if '.DS_Store' in img_path:
            continue
        path_imgs.append(os.path.join(args.image_dir, img_path))

    assert path_imgs, 'No images at %s' % args.image_dir

    # load resnet pretrained model
    assert args.weights_type in ['random', 'imagenet', 'custom'], "Not supported weights type: %s" % args.weights_type
    if args.weights_type == 'imagenet':
        net = models.resnet50(pretrained=True)
        print("Visualizing ImageNet pretrained weights!")
    else:
        net = models.resnet50(pretrained=False)
        if args.weights_type == 'custom':
            state_dict = torch.load(args.weights_path)
            net.load_state_dict(state_dict)
            print("Visualizing Custom pretrained weights!")
        else:
            print("Visualizing random initialized weights!")
    net.eval()
    print(net)

    os.makedirs(args.output_dir, exist_ok=True)

    for id, img_path in enumerate(path_imgs):
        grad_cam(img_path, args.output_dir, net, indices=id)
