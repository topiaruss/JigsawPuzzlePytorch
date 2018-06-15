import pydicom
import tablib
import os

from .dataset.dicom_jigsaw_loader import (partition_train_val_sets,
                                          define_random_blocks,
                                          DicomDataset,
                                          CONTEXT_FILE)


def test_partition_test_val_sets():
    headers = ('dicom_file',)
    data = [('a/b',), ('b/c',), ('c/d',), ('d/e',)]
    data = tablib.Dataset(*data, headers=headers)
    t, v = partition_train_val_sets(data)
    assert len(t) > 0
    assert len(v) > 0
    assert v.isdisjoint(t)
    assert len(t) + len(v) == len(data)


def test_select_random_blocks():
    ff = '/home/pc1/data/PEONY-2018-06/patients/HCA217594/2018-05-02/PAT01/BTO/BTO03'
    blocks = define_random_blocks(ff, 255, 10)
    assert len(blocks) == 10


def test_dicom_instance_clean():
    dp = """/home/pc1/data/PEONY-2018-06/patients/HCA110723/
            /home/pc1/data/PEONY-2018-06/patients/HCA18595/
            /home/pc1/data/PEONY-2018-06/patients/HCA265541/
            /home/pc1/data/PEONY-2018-06/patients/HCA217594/"""
    if os.path.isfile(CONTEXT_FILE):
        os.remove(CONTEXT_FILE)
    dds = DicomDataset(dp)


def test_dicom_instance_retrieve():
    # lazily assumes that context will have been dumped in previous test
    dp = """/home/pc1/data/PEONY-2018-06/patients/HCA110723/
            /home/pc1/data/PEONY-2018-06/patients/HCA18595/
            /home/pc1/data/PEONY-2018-06/patients/HCA265541/
            /home/pc1/data/PEONY-2018-06/patients/HCA217594/"""
    assert os.path.isfile(CONTEXT_FILE)
    dds = DicomDataset(dp)
    assert hasattr(dds, 'from_context')


def test_get_items():
    dp = """/home/pc1/data/PEONY-2018-06/patients/HCA110723/
            /home/pc1/data/PEONY-2018-06/patients/HCA18595/
            /home/pc1/data/PEONY-2018-06/patients/HCA265541/
            /home/pc1/data/PEONY-2018-06/patients/HCA217594/"""
    assert os.path.isfile(CONTEXT_FILE)
    dds = DicomDataset(dp)
    assert hasattr(dds, 'from_context')

    jj = dds[0]
    pass


def test_get_items():
    dp = """/home/pc1/data/PEONY-2018-06/patients/HCA110723/
            /home/pc1/data/PEONY-2018-06/patients/HCA18595/
            /home/pc1/data/PEONY-2018-06/patients/HCA265541/
            /home/pc1/data/PEONY-2018-06/patients/HCA217594/"""
    assert os.path.isfile(CONTEXT_FILE)
    dds = DicomDataset(dp)
    assert hasattr(dds, 'from_context')
    dds.plot_item(0)
