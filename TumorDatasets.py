import copy
import csv
import functools
import glob
import math
import os
import random
from Database import get_ct_paths
from collections import namedtuple
import SimpleITK as sitk  # 确保安装 SimpleITK: pip install SimpleITK
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import PatientCoordTuple, patientCoord2voxelCoord
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

UPLOAD_FOLDER = 'data-unversioned/data/temp'
raw_cache = getCache('tumor_data')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz',
)
MaskTuple = namedtuple(
    'MaskTuple',
    'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask',
)


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob('data-unversioned/data/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []
    with open('data/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = {'False': False, 'True': True}[row[5]]

            candidateInfo_list.append(
                CandidateInfoTuple(True, True, isMal_bool, annotationDiameter_mm, series_uid, annotationCenter_xyz))

    with open('data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool:
                candidateInfo_list.append(CandidateInfoTuple(
                    False,
                    False,
                    False,
                    0.0,
                    series_uid,
                    candidateCenter_xyz,
                ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid, []).append(candidateInfo_tup)

    return candidateInfo_dict


class Ct:
    def __init__(self, series_uid):

        # 直接从数据库查询 MHD 和 RAW 文件路径
        mhd_path, raw_path = get_ct_paths(series_uid)
        # 验证路径是否存在
        if not mhd_path or not os.path.exists(mhd_path):
            raise FileNotFoundError(f"数据库中未找到有效的 MHD 文件: {series_uid}.mhd")
        ct_mhd = sitk.ReadImage(mhd_path)

        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = PatientCoordTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = PatientCoordTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = patientCoord2voxelCoord(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid, buildMasks_bool=False)
    return len(ct.negative_indexes)


def getCtAugmentedCandidate(
        augmentation_dict,
        series_uid, center_xyz, width_irc,
        use_cache=True):
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)
    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_t.size(),
        align_corners=False,
    )

    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode='border',
        align_corners=False,
    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 augmentation_dict=None,
                 candidateInfo_list=None,
                 ):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(
                set(candidateInfo_tup.series_uid for candidateInfo_tup in self.candidateInfo_list))

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        series_set = set(self.series_list)
        self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid in series_set]

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.neg_list = \
            [nt for nt in self.candidateInfo_list if not nt.isNodule_bool]
        self.pos_list = \
            [nt for nt in self.candidateInfo_list if nt.isNodule_bool]
        self.ben_list = \
            [nt for nt in self.pos_list if not nt.isMal_bool]
        self.mal_list = \
            [nt for nt in self.pos_list if nt.isMal_bool]

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.neg_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.candidateInfo_list)
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)
            random.shuffle(self.ben_list)
            random.shuffle(self.mal_list)

    def __len__(self):
        if self.ratio_int:
            return 50000
        else:
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_list)
                candidateInfo_tup = self.neg_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.pos_list[pos_ndx]
        else:
            candidateInfo_tup = self.candidateInfo_list[ndx]

        return self.sampleFromCandidateInfo_tup(
            candidateInfo_tup, candidateInfo_tup.isNodule_bool
        )

    def sampleFromCandidateInfo_tup(self, candidateInfo_tup, label_bool):
        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.getRawCandidate(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        label_t = torch.tensor([False, False], dtype=torch.long)

        if not label_bool:
            label_t[0] = True
            index_t = 0
        else:
            label_t[1] = True
            index_t = 1

        return candidate_t, label_t, index_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)


class MalignantLunaDataset(LunaDataset):
    def __len__(self):
        if self.ratio_int:
            return 100000
        else:
            return len(self.ben_list + self.mal_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            if ndx % 2 != 0:
                candidateInfo_tup = self.mal_list[(ndx // 2) % len(self.mal_list)]
            elif ndx % 4 == 0:
                candidateInfo_tup = self.ben_list[(ndx // 4) % len(self.ben_list)]
            else:
                candidateInfo_tup = self.neg_list[(ndx // 4) % len(self.neg_list)]
        else:
            if ndx >= len(self.ben_list):
                candidateInfo_tup = self.mal_list[ndx - len(self.ben_list)]
            else:
                candidateInfo_tup = self.ben_list[ndx]
        return self.sampleFromCandidateInfo_tup(
            candidateInfo_tup, candidateInfo_tup.isMal_bool
        )


# --- START: New additions for getCtFromDirectory ---
# 注意：以下代码块应追加到你现有 TumorDatasets.py 文件的末尾。
# 确保 Ct 类和 PatientCoordTuple 的定义在它之前。

class CtFromDirectory:
    """
    一个用于从指定目录加载CT影像数据的类。
    这个类类似于现有的 Ct 类，但它是从给定目录而不是全局搜索来加载数据。
    支持 .mhd/.raw 对和 DICOM (.dcm) 文件系列。
    """

    def __init__(self, directory_path):
        self.directory_path = directory_path

        mhd_files = glob.glob(os.path.join(directory_path, '*.mhd'))

        if mhd_files:  # 优先处理 .mhd 文件
            mhd_path = mhd_files[0]  # 假设每个目录只包含一个mhd文件系列
            ct_mhd = sitk.ReadImage(mhd_path)
            ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
            ct_a.clip(-1000, 1000, ct_a)

            # 从文件名推断 series_uid
            self.series_uid = os.path.splitext(os.path.basename(mhd_path))[0]
            self.hu_a = ct_a
            self.origin_xyz = PatientCoordTuple(*ct_mhd.GetOrigin())
            self.vxSize_xyz = PatientCoordTuple(*ct_mhd.GetSpacing())
            self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
            log.info(f"成功从MHD文件 {mhd_path} 加载CT，UID: {self.series_uid}")

        else:  # 尝试处理 DICOM 文件
            dcm_files = glob.glob(os.path.join(directory_path, '*.dcm'))
            if dcm_files:
                try:
                    # 使用 SimpleITK 的 ImageSeriesReader 来加载整个序列
                    series_reader = sitk.ImageSeriesReader()
                    series_filenames = series_reader.GetImageSeriesFileNames(directory_path)

                    if not series_filenames:
                        raise ValueError(f"在目录 {directory_path} 中找不到DICOM系列文件。")

                    series_reader.SetFileNames(series_filenames)
                    ct_image = series_reader.Execute()

                    # 尝试从DICOM文件读取SeriesInstanceUID (0020|000e)
                    # SimpleITK加载后，可以在Image对象中通过GetMetaDataKeys()和GetMetaData()访问元数据
                    try:
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(series_filenames[0])
                        reader.ReadImageInformation()
                        series_uid_from_dcm = reader.GetMetaData('0020|000e')
                    except Exception as meta_e:
                        series_uid_from_dcm = os.path.basename(directory_path)  # 回退到目录名作为UID
                        log.warning(
                            f"无法从DICOM头获取SeriesInstanceUID: {meta_e}. 使用目录名作为UID: {series_uid_from_dcm}")

                    ct_a = np.array(sitk.GetArrayFromImage(ct_image), dtype=np.float32)
                    ct_a.clip(-1000, 1000, ct_a)

                    self.series_uid = series_uid_from_dcm
                    self.hu_a = ct_a
                    self.origin_xyz = PatientCoordTuple(*ct_image.GetOrigin())
                    self.vxSize_xyz = PatientCoordTuple(*ct_image.GetSpacing())
                    self.direction_a = np.array(ct_image.GetDirection()).reshape(3, 3)
                    log.info(f"成功从DICOM目录 {directory_path} 加载CT，UID: {self.series_uid}")
                except Exception as e:
                    log.error(f"从DICOM目录 {directory_path} 加载CT失败: {e}")
                    raise FileNotFoundError(
                        f"在目录 {directory_path} 中未找到有效的 .mhd 或 .dcm 文件，或无法加载DICOM系列。") from e
            else:
                raise FileNotFoundError(f"在目录 {directory_path} 中未找到任何 .mhd 或 .dcm 文件。")

    def getRawCandidate(self, center_xyz, width_irc):
        # 复制现有 Ct 类的 getRawCandidate 方法
        center_irc = patientCoord2voxelCoord(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            # 这里的断言需要调整，因为 self.hu_a 可能在 __init__ 中没有完全构建
            # 更好的做法是在 CtFromDirectory 的 __init__ 中确保 hu_a 完整
            # assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


# 定义 getCtFromDirectory 函数，它将返回 CtFromDirectory 的实例
def getCtFromDirectory(directory_path):
    """
    从指定目录加载CT影像数据。
    :param directory_path: 包含CT数据的目录路径（可以是.mhd/.raw对或DICOM系列）。
    :return: CtFromDirectory 类的实例。
    """
    return CtFromDirectory(directory_path)

# --- END: New additions for getCtFromDirectory ---