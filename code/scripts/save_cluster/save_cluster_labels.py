"""
1) given the net trained unsupervised on ImageNet
2) feed training ImageNet images into the network and get unsupervised cluster labels
3) visualize the clustering on tensorboard
"""
import argparse
import os
import pickle
import sys

import torch
import torch.utils.data
import torchvision.datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.insert(0, '/scratch/hdd001/home/yanxi/IIC')
import code.archs as archs
from code.datasets.clustering.downsampled_imagenet import ImageNetDS
from code.utils.cluster.cluster_eval import _clustering_get_data
from code.utils.cluster.transforms import sobel_make_transforms
from code.utils.cluster.cluster_eval import cluster_eval
from code.utils.cluster.data import cluster_twohead_create_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str,
                        default="./out")

    given_config = parser.parse_args()

    given_config.out_dir = given_config.save_dir

    reloaded_config_path = os.path.join(given_config.out_dir, "config.pickle")
    print("Loading restarting config from: %s" % reloaded_config_path)
    with open(reloaded_config_path, "rb") as config_f:
        config = pickle.load(config_f)
        print(config)

    if not hasattr(config, "twohead"):
        config.twohead = ("TwoHead" in config.arch)

    config.double_eval = False  # no double eval, not training (or saving config)

    net = archs.__dict__[config.arch](config)
    model_path = os.path.join(given_config.out_dir, "best_net.pytorch")
    net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.eval()

    # print(net)
    tf1, tf2, tf3 = sobel_make_transforms(config)
    # Pass each image in net to get cluster prediction
    # print(tf1)
    # print(tf2)
    # print(tf3)

    dataset = torchvision.datasets.CIFAR10(config.dataset_root, train=True, transform=tf3)
    # dataset = ImageNetDS(config.dataset_root + '/downsampled-imagenet-32/', 32, train=True, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_sz, shuffle=False)

    flat_predss_all, flat_targets_all, = _clustering_get_data(config, net, dataloader, sobel=True,
                                                              using_IR=False,
                                                              verbose=True)
    print(len(flat_predss_all), flat_predss_all[0].shape)
    print(flat_targets_all.shape)
    print(flat_targets_all[:50])
    print(flat_predss_all[0][:50], flat_predss_all[1][:50], flat_predss_all[2][:50], flat_predss_all[3][:50])
    print(config.num_sub_heads)

    cluster_labels = flat_predss_all[0].cpu().numpy()
    actual_labels = flat_targets_all.cpu().numpy()

    # visualize each cluster
    view_dataset = torchvision.datasets.CIFAR10(config.dataset_root, train=True,
                                                transform=torchvision.transforms.ToTensor())
    for c in range(config.output_k_B):
        cluster_indices = np.where(cluster_labels == c)[0]
        gt_indices = np.where(actual_labels == c)[0]

        c_dataloader = torch.utils.data.DataLoader(view_dataset, batch_size=64, shuffle=False,
                                                   sampler=SubsetRandomSampler(cluster_indices))
        gt_dataloader = torch.utils.data.DataLoader(view_dataset, batch_size=64, shuffle=False,
                                                    sampler=SubsetRandomSampler(gt_indices))

        for (images, targets) in c_dataloader:
            print("saving cluster {}".format(c), images.shape)
            torchvision.utils.save_image(images, 'c{}.png'.format(c))
            break

        for (images, targets) in gt_dataloader:
            print("sanity check gt classes {}".format(c), images.shape)
            torchvision.utils.save_image(images, 'gt{}.png'.format(c))
            break


if __name__ == "__main__":
    main()
    print("Done!")
