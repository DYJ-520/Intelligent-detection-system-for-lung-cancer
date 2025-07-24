import csv
import functools
import glob
import os
import random
from collections import namedtuple
import SimpleITK as sitk
import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import PatientCoordTuple, patientCoord2voxelCoord
from util.logconf import logging


def get_ct_paths(series_uid):
    mhd_path = glob.glob(f'data-unversioned/data/subset*/{series_uid}.mhd')
    if mhd_path:
        return mhd_path[0], mhd_path[0].replace('.mhd', '.raw')
    return None, None


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache('seg_data_2d')

CandidateInfoTuple = namedtuple('CandidateInfoTuple',
                                'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob('data-unversioned/data/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    if not presentOnDisk_set:
        log.warning("No data found, using dummy series UIDs.")
        presentOnDisk_set = {'dummy_series_1', 'dummy_series_2'}

    candidateInfo_list = []
    annotations_path = 'data/annotations_with_malignancy.csv'
    if os.path.exists(annotations_path):
        with open(annotations_path, "r") as f:
            for row in list(csv.reader(f))[1:]:
                series_uid = row[0]
                if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                    continue

                annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
                annotationDiameter_mm = float(row[4])
                isMal_bool = {'False': False, 'True': True}[row[5]]

                candidateInfo_list.append(
                    CandidateInfoTuple(
                        True,
                        True,
                        isMal_bool,
                        annotationDiameter_mm,
                        series_uid,
                        annotationCenter_xyz,
                    )
                )

    candidates_path = 'data/candidates.csv'
    if os.path.exists(candidates_path):
        with open(candidates_path, "r") as f:
            for row in list(csv.reader(f))[1:]:
                series_uid = row[0]
                if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                    continue

                isNodule_bool = bool(int(row[4]))
                if not isNodule_bool:
                    candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
                    candidateInfo_list.append(
                        CandidateInfoTuple(
                            False,
                            False,
                            False,
                            0.0,
                            series_uid,
                            candidateCenter_xyz,
                        )
                    )

    candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
    return candidateInfo_list


@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid,
                                      []).append(candidateInfo_tup)

    return candidateInfo_dict


class Ct:
    def __init__(self, series_uid):
        self.series_uid = series_uid
        mhd_path, _ = get_ct_paths(series_uid)

        if mhd_path is None or not os.path.exists(mhd_path):
            log.warning(f"CT file not found for series_uid: {series_uid}. Using dummy data.")
            self.hu_a = np.random.rand(128, 512, 512).astype(np.float32) * 2000 - 1000
            self.origin_xyz = PatientCoordTuple(0.0, 0.0, 0.0)
            self.vxSize_xyz = PatientCoordTuple(1.0, 1.0, 1.0)
            self.direction_a = np.eye(3)
        else:
            ct_mhd = sitk.ReadImage(mhd_path)
            self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
            self.origin_xyz = PatientCoordTuple(*ct_mhd.GetOrigin())
            self.vxSize_xyz = PatientCoordTuple(*ct_mhd.GetSpacing())
            self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        self.positive_mask = self.buildAnnotationMask()
        self.positive_indexes = (self.positive_mask.sum(axis=(1, 2))
                                 .nonzero()[0].tolist())

    def buildAnnotationMask(self):
        mask_a = np.zeros_like(self.hu_a, dtype=bool)

        candidateInfo_list = getCandidateInfoDict().get(self.series_uid, [])

        for candidateInfo_tup in candidateInfo_list:
            if not candidateInfo_tup.isNodule_bool:
                continue

            center_irc = patientCoord2voxelCoord(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )

            radius = int((candidateInfo_tup.diameter_mm / 2) / self.vxSize_xyz[0]) + 1

            ci, cr, cc = int(center_irc.index), int(center_irc.row), int(center_irc.col)
            z, y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
            sphere = z ** 2 + y ** 2 + x ** 2 <= radius ** 2

            z_min = max(0, ci - radius)
            z_max = min(mask_a.shape[0], ci + radius + 1)
            y_min = max(0, cr - radius)
            y_max = min(mask_a.shape[1], cr + radius + 1)
            x_min = max(0, cc - radius)
            x_max = min(mask_a.shape[2], cc + radius + 1)

            sphere_z_min = radius - (ci - z_min)
            sphere_z_max = radius + (z_max - ci)
            sphere_y_min = radius - (cr - y_min)
            sphere_y_max = radius + (y_max - cr)
            sphere_x_min = radius - (cc - x_min)
            sphere_x_max = radius + (x_max - cc)

            mask_a[z_min:z_max, y_min:y_max, x_min:x_max] = np.logical_or(
                mask_a[z_min:z_max, y_min:y_max, x_min:x_max],
                sphere[sphere_z_min:sphere_z_max, sphere_y_min:sphere_y_max, sphere_x_min:sphere_x_max]
            )

        return mask_a


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 contextSlices_count=3,
                 fullCt_bool=False,
                 ):
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            ct = getCt(series_uid)
            index_count = ct.hu_a.shape[0]

            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in ct.positive_indexes]

        log.info("{!r}: {} {} series, {} slices".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.sample_list),
        ))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx]

        ct = getCt(series_uid)

        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1

        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(0, min(context_ndx, ct.hu_a.shape[0] - 1))
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, series_uid, slice_ndx
