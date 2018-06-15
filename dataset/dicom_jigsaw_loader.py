# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle

import matplotlib.pyplot as plt
import pydicom
import random
import tablib
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

CONTEXT_FILE = 'stored_context.pickle'
JIGSAWS = 120
VALFRAC = 0.25
BLOCK_SIDE = 255


def partition_train_val_sets(data):
    all_files = set(data['dicom_file'])
    val_count = int(len(all_files) * VALFRAC)
    assert val_count > 0
    val_files = set(random.sample(all_files, val_count))
    train_files = all_files - val_files
    return train_files, val_files


def normalise(img, bits):
    """
    return the image in the float range 0..1
    :param img:
    :return:
    """
    return img / (2 ** bits - 1)


def get_defined_block(row):
    """given a row from data, return the block"""
    ds = pydicom.dcmread(row['path'])
    layer, x, y, side = row['layer'], row['x'], row['y'], row['side']
    return ds.pixel_array[layer][x:x + side, y:y + side]


def define_random_blocks(path, side, count, maxtries=1000):
    """Returns a descriptor for up to count blocks that have at least 0.75 coverage"""
    ds = pydicom.dcmread(path)
    layers, x, y = ds.pixel_array.shape
    max_x, max_y = x - side, y-side
    tries = 0
    blocks = []
    while tries < maxtries:
        tries += 1
        layer, x, y = random.randrange(layers), random.randrange(max_x), random.randrange(max_y)
        block = ds.pixel_array[layer][x:x + side, y:y + side]
        norm = normalise(block, ds.BitsStored)
        mask = block > 0.0
        mask_mean = mask.mean()
        if mask_mean < 0.75:
            # print('min: %s, max %s, skip due to low coverage %s' % (min, max, mask_mean))
            continue
        min, mean, max = norm.min(), norm.mean(), norm.max()
        print('cov: %4.3f, min: %4.3f, max %4.3f, mean %4.3f, tries %d' % (mask_mean, min, max, mean, tries))
        blocks.append(dict(path=path, layer=layer, x=x, y=y, side=side, coverage=mask_mean))
        if len(blocks) >= count:
            break
    return blocks


def study_date_to_iso(sd):
    return "%s-%s-%s" % (sd[0:4], sd[4:6], sd[6:])


def get_patient_id(path):
    dirs = path.split(os.sep)
    pats = [d for d in dirs if 'hca' in d.lower()]
    if pats:
        return pats[0]
    return None


class DicomDataset(data.Dataset):
    def __init__(self, data_path, classes=1000, train=True):
        self.data_path = data_path.split()
        self.data = tablib.Dataset(headers='dicom_file patient exam_date rows cols layers lateral view desc'.split())
        self._data_scrape(self.data_path)
        self.permutations = self._retrieve_permutations(classes)
        self._retrieve_context()
        self.__image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            # transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])
        # we have computed/retrieved the train and val lists, now we select which we need
        self.myblocks = self.blocks['train'] if train else self.blocks['val']

    def __getitem__(self, index):
        """

        :param index:
        :return:
        BORD = pixel border
        SIDE = side of output square
        """
        BORD = 5
        SIDE = 75

        img = get_defined_block(self.myblocks[index])
        s = img.shape[0] / 3.0
        tiles = [None] * 9

        for n in range(9):
            i = n // 3
            j = n % 3
            orig = [int(s * i + BORD), int(s * j + BORD)]
            end = [int(o + SIDE) for o in orig]
            tile = img[orig[0]:end[0], orig[1]:end[1]]
            # print(i, j, orig, end, tile.shape)
            # Normalize the patches independently to avoid low level features shortcut
            tile = torch.from_numpy(tile/tile.max())
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)
        return data, order, tiles

    def plot_item(self, index):
        """Plots 18 squares. First 9 are the permuted tiles. Last 9 are original.
        Prints to console the permutation index and permutation.
        You can see how the permutation dictates the placement of the tiles from the
        last 9 into the top 9 tiles.

        :param index: index into the dataset

        """
        data, order, tiles = self[index]
        fig = plt.figure(figsize=(6, 3))
        print("order: %s, %s" % (order, self.permutations[order]))
        for i in range(9):
            fig.add_subplot(6, 3, i+1)
            plt.imshow(data[i])
        for i in range(9):
            fig.add_subplot(6, 3, i+1+9)
            plt.imshow(tiles[i])
        plt.show()

    def __len__(self):
        return len(self.myblocks)

    def _dicom_info(self, path, verbose=False):
        """extract info from file at path."""
        try:
            df = pydicom.read_file(path)
            # exception if non 3D
            nof = df.NumberOfFrames
            if nof == 1:
                if verbose:
                    print(path)
                    print('non-3D dicom file')
                    print()
                return

            # remove up to and including root of the patient repo
            # path = path[path.index('patients'):]
    
            exam_date = study_date_to_iso(df.StudyDate)
            desc = df.SeriesDescription
            desc.replace(' Breast Tomosynthesis Image', '')
            lateral = 'L' if 'L ' in desc else 'R'
            view = 'CC' if 'CC' in desc else 'MLO'
            pat = get_patient_id(path)
            print(pat)
            self.data.append((path, pat, exam_date, df.Rows, df.Columns, nof, lateral, view, desc))

        except OSError as ex:
            print('dicom error? %s' % path)
            print(ex)
            print()

        except IOError as ex:
            print('no such file %s' % path)
            print(ex)
            print()

        except pydicom.errors.InvalidDicomError:
            if verbose:
                print(path)
                print('non-dicom')
                print()

        except AttributeError:
            # it's dicom, but has no NumberOfFrames, therefore non 3D
            if verbose:
                print(path)
                print('non-3D dicom file')
                print()

    def _next_file(self, fods):
        """generator for files from args and os walk, ignoring special files"""
        fods = [fods] if type(fods) == str else fods
        for fod in fods:
            if os.path.isfile(fod):
                yield fod
            if os.path.isdir(fod):
                for root, dirs, files in os.walk(fod):
                    for name in files:
                        yield os.path.join(root, name)

    def _data_scrape(self, data_path):
        """traverse the patient images and populate an index of key features"""
        for f in self._next_file(data_path):
            self._dicom_info(f)

    def _retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm

    def _retrieve_context(self):
        try:
            self.context = pickle.load(open(CONTEXT_FILE, 'rb'))
            print('loaded context from %s' % CONTEXT_FILE)
            # check the set of files is still the same as when blocks computed
            pickled = self.context['data']
            fresh = self.data
            diffs = [(a, fresh[i]) for i, a in enumerate(pickled) if a != fresh[i]]
            print('DIFFS:', diffs)
            assert not diffs
            self.blocks = self.context['blocks']
            self.data = self.context['data']
            self.from_context = True
        except IOError as ex:
            print("failed loading context from %s, so we'll compute afresh" % CONTEXT_FILE)
            print(ex)
            self._compute_blocks()
            context = dict(blocks=self.blocks, data=self.data)
            pickle.dump(context, open(CONTEXT_FILE, 'wb'))

    def _compute_blocks(self):
        self.train_files, self.val_files = partition_train_val_sets(self.data)
        self.blocks = dict(train=[], val=[])
        print('compute train blocks')
        while len(self.blocks['train']) < JIGSAWS:
            for f in self.train_files:
                print(f)
                try:
                    self.blocks['train'].extend(define_random_blocks(f, BLOCK_SIDE, 5))
                except TypeError as ex:
                    print("failed reading file %s" % f)
                    print(ex)
        print('compute val blocks')
        while len(self.blocks['val']) < JIGSAWS:
            for f in self.val_files:
                print(f)
                try:
                    self.blocks['val'].extend(define_random_blocks(f, BLOCK_SIDE, 5))
                except TypeError as ex:
                    print("failed reading file %s" % f)
                    print(ex)

