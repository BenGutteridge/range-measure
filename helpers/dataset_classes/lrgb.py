"""
Adapted from https://github.com/vijaydwivedi75/lrgb.git
https://github.com/HySonLab/Multires-Graph-Transformer.git
https://github.com/hamed1375/Exphormer.git
"""

import hashlib
import os.path as osp
import pickle
import shutil
from loguru import logger
import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from tqdm import tqdm
import os


class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(
        self,
        root="data",
        smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
        dir_name=None,
        subset=None,
    ):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.
        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.
        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']
        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        dir_name = (
            "peptides-functional"
            + (f"_{dir_name}" if dir_name else "")
            + (f"_subset={subset}" if subset else "")
        )

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, dir_name)
        self.subset = subset

        self.url = "https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1"
        self.version = (
            "701eb743e899f4d793f0e13c8fa5a1b4"  # MD5 hash of the intended dataset file
        )
        self.url_stratified_split = "https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "peptide_multi_class_dataset.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), "w").close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "peptide_multi_class_dataset.csv.gz")
        )
        smiles_list = data_df["smiles"]

        if self.subset is not None:
            logger.warning(
                f"Using a subset of the dataset of size {self.subset}. For dev ONLY."
            )
            smiles_list = smiles_list[: self.subset]
            data_df = data_df.iloc[: self.subset]

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([eval(data_df["labels"].iloc[i])])

            data_list.append(data)

        if self.pre_transform is not None:
            print("Applying pre_transform of graphs...")
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """Get dataset splits.
        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        if self.subset is not None:
            splits = [(self.subset // 10) * i for i in [6, 2, 2]]
            return {
                "train": torch.arange(splits[0]),
                "val": splits[0] + torch.arange(splits[1]),
                "test": splits[1] + torch.arange(splits[2]),
            }

        split_file = osp.join(self.root, "splits_random_stratified_peptide.pickle")
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)

        return split_dict


class VOCSuperpixels(InMemoryDataset):
    r"""The VOCSuperpixels dataset which contains image superpixels and a semantic segmentation label
    for each node superpixel.

    Construction and Preparation:
    - The superpixels are extracted in a similar fashion as the MNIST and CIFAR10 superpixels.
    - In VOCSuperpixels, the number of superpixel nodes <=500. (Note that it was <=75 for MNIST and
    <=150 for CIFAR10.)
    - The labeling of each superpixel node is done with the same value of the original pixel ground
    truth  that is on the mean coord of the superpixel node

    - Based on the SBD annotations from 11355 images taken from the PASCAL VOC 2011 dataset. Original
    source `here<https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal>`_.

    num_classes = 21
    ignore_label = 255

    color map
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow,
    11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train,
    20=tv/monitor

    Splitting:
    - In the original image dataset there are only train and val splitting.
    - For VOCSuperpixels, we maintain train, val and test splits where the train set is AS IS. The original
    val split of the image dataset is used to divide into new val and new test split that is eventually used
    in VOCSuperpixels. The policy for this val/test splitting is below.
    - Split total number of val graphs into 2 sets (val, test) with 50:50 using a stratified split proportionate
    to original distribution of data with respect to a meta label.
    - Each image is meta-labeled by majority voting of non-background grouth truth node labels. Then new val
    and new test is created with stratified sampling based on these meta-labels. This is done for preserving
    same distribution of node labels in both new val and new test
    - Therefore, the final train, val and test splits are correspondingly original train (8498), new val (1428)
    and new test (1429) splits.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): Option to select the graph construction format.
            If :obj: `"edge_wt_only_coord"`, the graphs are 8-nn graphs with the edge weights computed based on
            only spatial coordinates of superpixel nodes.
            If :obj: `"edge_wt_coord_feat"`, the graphs are 8-nn graphs with the edge weights computed based on
            combination of spatial coordinates and feature values of superpixel nodes.
            If :obj: `"edge_wt_region_boundary"`, the graphs region boundary graphs where two regions (i.e.
            superpixel nodes) have an edge between them if they share a boundary in the original image.
            (default: :obj:`"edge_wt_region_boundary"`)
        slic_compactness (int, optional): Option to select compactness of slic that was used for superpixels
            (:obj:`10`, :obj:`30`). (default: :obj:`30`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = {
        10: {
            "edge_wt_only_coord": "https://www.dropbox.com/s/rk6pfnuh7tq3t37/voc_superpixels_edge_wt_only_coord.zip?dl=1",
            "edge_wt_coord_feat": "https://www.dropbox.com/s/2a53nmfp6llqg8y/voc_superpixels_edge_wt_coord_feat.zip?dl=1",
            "edge_wt_region_boundary": "https://www.dropbox.com/s/6pfz2mccfbkj7r3/voc_superpixels_edge_wt_region_boundary.zip?dl=1",
        },
        30: {
            "edge_wt_only_coord": "https://www.dropbox.com/s/toqulkdpb1jrswk/voc_superpixels_edge_wt_only_coord.zip?dl=1",
            "edge_wt_coord_feat": "https://www.dropbox.com/s/xywki8ysj63584d/voc_superpixels_edge_wt_coord_feat.zip?dl=1",
            "edge_wt_region_boundary": "https://www.dropbox.com/s/8x722ai272wqwl4/voc_superpixels_edge_wt_region_boundary.zip?dl=1",
        },
    }

    def __init__(
        self,
        root,
        name="edge_wt_region_boundary",
        slic_compactness=30,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        subset=None,
    ):
        self.name = name
        self.slic_compactness = slic_compactness
        self.subset = subset
        assert split in ["train", "val", "test"]
        assert name in [
            "edge_wt_only_coord",
            "edge_wt_coord_feat",
            "edge_wt_region_boundary",
        ]
        assert slic_compactness in [10, 30]
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f"{split}.pt")
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["train.pickle", "val.pickle", "test.pickle"]

    @property
    def raw_dir(self):
        return osp.join(
            self.root,
            "slic_compactness_" + str(self.slic_compactness),
            self.name,
            "raw",
        )

    @property
    def processed_dir(self):
        return osp.join(
            self.root,
            "slic_compactness_" + str(self.slic_compactness),
            self.name,
            "processed",
        )

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url[self.slic_compactness][self.name], self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, "voc_superpixels_" + self.name), self.raw_dir)
        os.unlink(path)

    def process(self):
        for split in ["train", "val", "test"]:
            with open(osp.join(self.raw_dir, f"{split}.pickle"), "rb") as f:
                graphs = pickle.load(f)

            if self.subset is not None:
                logger.warning(
                    f"Using a subset of the dataset of size {self.subset}/{len(graphs)}. For dev ONLY."
                )
                subset_split = [  # 80:10:10 by default; for dev only so doesn't matter
                    int(self.subset * 0.8),
                    int(self.subset * 0.1),
                    int(self.subset * 0.1),
                ]
                if split == "train":
                    graphs = graphs[: subset_split[0]]
                elif split == "val":
                    graphs = graphs[subset_split[0] : subset_split[0] + subset_split[1]]
                elif split == "test":
                    graphs = graphs[subset_split[0] + subset_split[1] :]

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f"Processing {split} dataset")

            data_list = []
            for idx in indices:
                graph = graphs[idx]

                """
                Each `graph` is a tuple (x, edge_attr, edge_index, y)
                    Shape of x : [num_nodes, 14]
                    Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
                    Shape of edge_index : [2, num_edges]
                    Shape of y : [num_nodes]
                """

                x = graph[0].to(torch.float)
                edge_attr = graph[1].to(torch.float)
                edge_index = graph[2]
                y = torch.LongTensor(graph[3])

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(
                self.collate(data_list), osp.join(self.processed_dir, f"{split}.pt")
            )
