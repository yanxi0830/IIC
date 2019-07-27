"""
1) given the net trained unsupervised on ImageNet
2) feed training ImageNet images into the network and get unsupervised cluster labels
3) visualize the clustering on tensorboard
"""
import argparse
import os
import pickle

import torch

import code.archs as archs
from code.utils.cluster.cluster_eval import cluster_eval
from code.utils.cluster.data import cluster_twohead_create_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ind", type=int, required=True)
    parser.add_argument("--out_root", type=str,
                        default="./out")

    given_config = parser.parse_args()

    given_config.out_dir = os.path.join(given_config.out_root,
                                        str(given_config.model_ind))

    reloaded_config_path = os.path.join(given_config.out_dir, "config.pickle")
    print("Loading restarting config from: %s" % reloaded_config_path)
    with open(reloaded_config_path, "rb") as config_f:
        config = pickle.load(config_f)
    assert (config.model_ind == given_config.model_ind)

    if not hasattr(config, "twohead"):
        config.twohead = ("TwoHead" in config.arch)

    config.double_eval = False  # no double eval, not training (or saving config)

    net = archs.__dict__[config.arch](config)
    model_path = os.path.join(config.out_dir, "best_net.pytorch")
    net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    net.cuda()
    net = torch.nn.DataParallel(net)

    print(net)


if __name__ == "__main__":
    main()
    print("Done!")