import os
import argparse
import numpy as np
import cv2
import random
import shutil
from tqdm import tqdm
from models import MODELS


def main(opt):
    Model = MODELS[opt.model + "_" + opt.engine]
    model = Model(opt.weights, opt.input_shape, opt.device)
    imgs = list()
    print("Calculating image scores...")
    for fn in tqdm(os.listdir(opt.src)):
        fp = os.path.join(opt.src, fn)
        img = cv2.imread(fp)
        cls, cls_prob, latency = model.infer(img)
        score = cls_prob[opt.cls]
        imgs.append((fp, score))
    imgs.sort(key=lambda x: x[1])
    # Pareto principle 80/20
    hard_num = int(0.2*len(imgs))
    hard_imgs = imgs[:hard_num]
    easy_imgs = imgs[hard_num:]
    assert len(easy_imgs) > opt.drop, \
        "Number of images to be dropped must be less than 0.8 of the dataset"
    easy_imgs = random.sample(easy_imgs, len(easy_imgs)-opt.drop)
    imgs = hard_imgs + easy_imgs
    print("Writing new dataset...")
    os.makedirs(opt.dst, exist_ok=True)
    for fp, score in tqdm(imgs):
        shutil.copy(fp, opt.dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        type=str,
                        required=True,
                        help='path to source directory')
    parser.add_argument('--dst',
                        type=str,
                        required=True,
                        help='path to destination directory')
    parser.add_argument('--drop',
                        type=float,
                        required=True,
                        help='number of images to be dropped')
    parser.add_argument('--model',
                        type=str,
                        default="classifier",
                        help='model type to extract feature')
    parser.add_argument('--engine',
                        type=str,
                        default='onnx',
                        help='engine type (onnx, mnn, torch)')
    parser.add_argument('--weights',
                        type=str,
                        required=True,
                        help='path to model weights')
    parser.add_argument('--device',
                        type=str,
                        required=True,
                        help='device to run model')
    parser.add_argument('--input-shape',
                        type=int,
                        nargs="+",
                        default=[384, 384],
                        help='model input shape')
    parser.add_argument('--cls',
                        type=int,
                        help='class for classification')

    opt = parser.parse_args()
    if not 0 < opt.ratio < 1:
        raise Exception("Drop ratio should be in range (0, 1)")
    main(opt)
