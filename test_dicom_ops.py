import pydicom
import tablib

from .dataset.dicom_jigsaw_loader import (partition_train_val_sets,
                                          define_random_tiles,
                                          DicomDataset)


def test_partition_test_val_sets():
    headers = ('dicom_file',)
    data = [('a/b',), ('b/c',), ('c/d',), ('d/e',)]
    data = tablib.Dataset(*data, headers=headers)
    t, v = partition_train_val_sets(data)
    assert len(t) > 0
    assert len(v) > 0
    assert v.isdisjoint(t)
    assert len(t) + len(v) == len(data)


def test_select_random_tiles():
    ff = '/home/pc1/data/PEONY-2018-06/patients/HCA217594/2018-05-02/PAT01/BTO/BTO03'
    tiles = define_random_tiles(ff, 255, 10)
    assert len(tiles) == 10

def test_dicom_instance():
    dp = """/home/pc1/data/PEONY-2018-06/patients/HCA110723/
            /home/pc1/data/PEONY-2018-06/patients/HCA18595/
            /home/pc1/data/PEONY-2018-06/patients/HCA265541/
            /home/pc1/data/PEONY-2018-06/patients/HCA217594/"""
    dds = DicomDataset(dp)