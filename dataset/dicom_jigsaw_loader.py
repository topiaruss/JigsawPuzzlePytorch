# -*- coding: utf-8 -*-
import numpy as np
import os

import matplotlib.pyplot as plt
import pydicom
import random
import tablib
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

TILE_INFO_FILE = 'tile_info.pickle'
JIGSAWS = 120
VALFRAC = 0.25


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


def define_random_tiles(path, side, count, maxtries=1000):
    """Returns a descriptor for up to count tiles that have at least 0.75 coverage"""
    ds = pydicom.dcmread(path)
    layers, x, y = ds.pixel_array.shape
    max_x, max_y = x - side, y-side
    tries = 0
    tiles = []
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
        tiles.append(dict(path=path, x=x, y=y, coverage=mask_mean))
        if len(tiles) >= count:
            break
    return tiles


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
        self._retrieve_tiles()
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
        self.mytiles = self.tiles['test']

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]

        img = Image.open(framename).convert('RGB')
        if np.random.rand() < 0.30:
            img = img.convert('LA').convert('RGB')

        if img.size[0] != 255:
            img = self.__image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)

        return data, int(order), tiles

    def __len__(self):
        return len(self.names)

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

    def _retrieve_tiles(self):
        try:
            self.tiles = np.load(TILE_INFO_FILE)
        except IOError as ex:
            print("failed loading %s")
            print(ex)
            self.tiles = self._compute_tiles()
            np.save(TILE_INFO_FILE, self.tiles)

    def _compute_tiles(self):
        train_files, val_files = partition_train_val_sets(self.data)
        self.tiles = dict(train=[], val=[])
        print('compute train tiles')
        while len(self.tiles['train']) < JIGSAWS:
            for f in train_files:
                print(f)
                try:
                    self.tiles['train'].extend(define_random_tiles(f, 255, 5))
                except TypeError as ex:
                    print("failed reading file %s" % f)
                    print(ex)
        print('compute val tiles')
        while len(self.tiles['val']) < JIGSAWS:
            print(f)
            for f in val_files:
                try:
                    self.tiles['val'].extend(define_random_tiles(f, 255, 5))
                except TypeError as ex:
                    print("failed reading file %s" % f)
                    print(ex)
