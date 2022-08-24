import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
from feature_extractors import FEATURE_EXTRACTORS


def main(opt):
    feature_extractor = FEATURE_EXTRACTORS[opt.model + "_" + opt.engine]
    if opt.cluster_model == "kmeans":
        pass
    else:
        raise NotImplementedError("%s is not implemented!" % opt.cluster_model)

    embeddings = list()
    print("Calculating embedding vectors from %s..." % opt.src)
    for fn in tqdm(os.listdir(opt.src)):
        if not (fn.endswith(".jpg") or fn.endswith(".png") or fn.endswith(".jpeg")):
            continue
        fp = os.path.join(opt.src, fn)
        img = cv2.imread(fp)
        embedding, latency = feature_extractor.infer(img)
        embeddings.append(embeddings)

    print("Fitting clustering model...")
    cluster_model.fit(embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        type=str,
                        required=True,
                        help='path to source directory')
    parser.add_argument('--ratio',
                        type=float,
                        default=0.7,
                        help='drop ratio')
    parser.add_argument('--model',
                        type=str,
                        default="basic_backbone",
                        help='model type to extract feature')
    parser.add_argument('--engine',
                        type=str,
                        default='onnx',
                        help='engine type (onnx, mnn, torch)')
    parser.add_argument('--weights',
                        type=str,
                        required=True,
                        help='path to model weights')
    parser.add_argument('--cluster-model',
                        type=str,
                        default='kmeans',
                        help='clustering algorithm')
    parser.add_argument('--k',
                        type=int,
                        required=True,
                        help='number of clusters')
    opt = parser.parse_args()
    if not 0 < opt.ratio < 1:
        raise Exception("Drop ratio should be in range (0, 1)")
    main(opt)
