import csv
import functools
import glob
import os
import copy
from util import util as myUtil
import SimpleITK as sitk
import numpy as np
import torch
import torch.cuda
import random
import math

import torch.nn.functional as F
from collections import namedtuple
from util.disk import getCache
from torch.utils.data import Dataset
from util.util import logging

#数据集处理相关的调试、信息、警告、错误和严重错误级别的消息都将被记录下来
log = logging.getLogger('Datasets')
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# 获取缓存对象，并设置缓存目录
raw_cache = getCache('raw_data')

# 创建一个命名元组
CandidateInfoTuple = namedtuple(    #候选和标记结节信息
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

@functools.lru_cache(1) # 保存函数运行的返回值（同一个参数保存一次，如果相同参数再调用就不用真正调用，访问cache就好了）
def getCandidateInfoList(requireOnDisk_bool=True):
    # 把所有的*.mhd文件名读到一个列表中（包含完整路径）
    mhd_list = glob.glob('data-unversioned/data/subset*/*.mhd')
    # 把CT扫描标识保存到一个集合中
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list} #包括子集序列号和mhd的序列号series_uid
    # 读取标注结节信息
    diameter_dict = {}
    with open('data/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]]) #坐标
            annotationDiameter_mm = float(row[4]) #结节直径大小
            # 把标注信息保存到一个字典中，其中键值为CT扫描文件标识
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )
    # 读取候选结节信息
    candidateInfo_list = []
    with open('data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            # 过滤掉没有CT扫描文件的候选结节。
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            # 把候选结节的良、恶标识转换为布尔值
            isNodule_bool = bool(int(row[4]))
            # 把候选结节的中心坐标转换为一个元组（X,Y,Z）
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            # 匹配确诊恶性肿瘤，并补齐肿瘤直径
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        isNodule_bool = 0
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break
            # 把候选结节信息添加到命名元组的列表中
            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))
    # 对列表进行降序排序
    candidateInfo_list.sort(reverse=True)
    # 返回候选元组列表
    return candidateInfo_list

class CT:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/data/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)   #读取mhd文件，返回为image类型,并且加载到内存中
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)  #转换为np数组
        ct_a.clip(-1000, 1000, ct_a)    #裁剪

        self.series_uid = series_uid
        self.hu_a = ct_a #体素

        self.origin_xyz = myUtil.PatientCoordTuple(*ct_mhd.GetOrigin())     #物理原点
        self.vxSize_xyz = myUtil.PatientCoordTuple(*ct_mhd.GetSpacing())       #体素间距
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)     #处理CT图像轴向与患者身体轴向不完全对齐的情况

    def getRawCandidate(self, center_xyz, width_irc):   #从整个CT图像中找到所需要的结节的小块图像数据
        # 把候选结节的患者坐标转换为图像存储的体素位置索引坐标
        center_irc = myUtil.patientCoord2voxelCoord(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )
        slice_list = []
        # 从CT扫描的体素数组中选出候选结节区域(图像切割)
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2)) #切割起始点
            end_ndx = int(start_ndx + width_irc[axis]) #切割结束点
            assert center_val >= 0 and center_val < self.hu_a.shape[axis],repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])
            slice_list.append(slice(start_ndx, end_ndx))
        ct_chunk = self.hu_a[tuple(slice_list)]
        # 返回结节对应的三维体素数组和其对应的中心索引元组
        return ct_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return CT(series_uid)
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc): #返回精确切片后的ct和中心坐标
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

def getCtAugmentedCandidate(
        augmentation_dict,#要对图像做那种增强？翻转平移还是噪声？
        series_uid, center_xyz, width_irc,
        use_cache=True): #是否查看缓存
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)   #把数组转化为张量

    transform_t = torch.eye(4) #记录变化情况

    for i in range(3):
        if 'flip' in augmentation_dict:         #旋转（全部通过仿射变换实现）->改变对角线上元素的正负
            if random.random() > 0.5:
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:        #随机平移，->改变第三列
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i,3] = offset_float * random_float

        if 'scale' in augmentation_dict:         #随机缩放，->改变对角线元素的绝对值大小
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float

    # --- 新增的剪切（Shear）变换代码块 ---
    if 'shear' in augmentation_dict:
        # 获取剪切强度参数，例如 0.1 表示最大10%的剪切
        shear_float = augmentation_dict['shear']

        # 为 X-Y, X-Z, Y-Z 平面生成随机剪切系数
        # random.random() * 2 - 1 会生成 [-1, 1] 范围的随机数
        shear_xy = (random.random() * 2 - 1) * shear_float
        shear_xz = (random.random() * 2 - 1) * shear_float
        shear_yz = (random.random() * 2 - 1) * shear_float
        # 对于对称剪切，可以进一步生成 Y-X, Z-X, Z-Y 剪切系数
        # 但通常我们只需要考虑主要方向的剪切，非对角线元素对图像的影响是相互的。
        # 这里的剪切矩阵构建方式是常见的简单剪切实现。
        shear_matrix_t = torch.tensor([
            [1,      shear_xy, shear_xz, 0],
            [shear_xy, 1,      shear_yz, 0], # Y轴相对于X轴倾斜 / X轴相对于Y轴倾斜，这里简化为shear_yx = shear_xy
            [shear_xz, shear_yz, 1,      0], # Z轴相对于X轴倾斜 / X轴相对于Z轴倾斜，这里简化为shear_zx = shear_xz
            [0,      0,      0,      1],
        ], dtype=torch.float32)

        transform_t @= shear_matrix_t  #数据增强

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2       #旋转，构建旋转矩阵
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t #矩阵乘法，融合旋转矩阵与仿射变换矩阵


    """
    反向映射->我们是有仿射矩阵了，按道理可以直接用这个矩阵对ct图片进行变换，但是乘出来发现有小数。所以
    我们可以用新的矩阵，然后推导应该来自于原来的什么位置，如果有小数，那么采用插值（估计）的方法
    """
    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            ct_t.size(),
            align_corners=False,
        )

    augmented_chunk = F.grid_sample( #插值
            ct_t,
            affine_t,
            padding_mode='border',
            align_corners=False,
        ).to('cpu')

    if 'noise' in augmentation_dict: #噪声->给每个位置随机+-值
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc #返回图像和中心位置


class LunaDataset(Dataset):    #用于加载和处理CT扫描数据
    def __init__(self,
                 val_stride=0,     #步伐，多少步挑出一个数据做验证集
                 isValSet_bool=None,    #是否是验证集
                 series_uid=None,    #CT标识
                 sortby_str='random',     #排序，如果希望一张ct在一起可以选 = series_uid
                 ratio_int=0,    #正负样本比例 = 0不更改， = 1则一个正样本一个负样本
                 augmentation_dict=None,    #包含数据增强的信息
                 candidateInfo_list=None,      #关于候选结节的信息
            ):

        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidateInfo_list:  #有的话直接从这里面读
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:    #没有再从外存读
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        # 只保留与series_uid匹配的候选结节
        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
        # # 取验证集
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride] #取验证集
            assert self.candidateInfo_list  #验证集为空则报错

        elif val_stride > 0:
            #  处理训练集，从训练集中剔除验证样本
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list  #训练集为空则报错

        if sortby_str == 'random':  #随机打乱顺序
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':    #按照id排序
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.negative_list = [      #不是结节的样本
            nt for nt in self.candidateInfo_list if not nt.isNodule_bool
        ]
        self.pos_list = [     #是结节的样本
            nt for nt in self.candidateInfo_list if nt.isNodule_bool
        ]

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format( #打印初始化日志
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.negative_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self): #打乱两种样本的顺序
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)

    def __len__(self):  #返回数据集有多少个样本
        if self.ratio_int:
            return 200000
        else:
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):  #要取负样本
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)
                candidateInfo_tup = self.negative_list[neg_ndx]
            else:  #要取正样本
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.pos_list[pos_ndx]
        else:  #不进行正负样本平衡
            candidateInfo_tup = self.candidateInfo_list[ndx]

        width_irc = (32, 48, 48) #剪切的图像尺寸

        if self.augmentation_dict:  #采用数据增强
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:  #不采用数据增强，从cache中拿数据
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:    #从磁盘拿数据
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.getRawCandidate(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ],
            dtype=torch.long,
        )

        return candidate_t, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)

# 模块中追加一个测试类，方便模块功能测试
class testDataset:   #测试类，打印日志
    def __init__(self,arg):
        self.arg = arg
        log.info("init {}".format(type(self).__name__))

    def main(self):
        log.info("Starting {}".format(type(self).__name__))
        candidateList = getCandidateInfoList(requireOnDisk_bool=True)
        log.info("数据集中的数据总量: {}".format(len(candidateList)))
        trainDataset = LunaDataset(val_stride=10, isValSet_bool=True, )
        log.info("训练集中的数据总量: {}".format(len(trainDataset)))
        valDataset = LunaDataset(isValSet_bool=True, val_stride=10)
        log.info("验证集中的数据总量: {}".format(len(valDataset)))

