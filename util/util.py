import collections
import datetime
import time
import numpy as np

from util.logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

VoxelCoordTuple = collections.namedtuple('VoxelCoordTuple', ['index', 'row', 'col'])
PatientCoordTuple = collections.namedtuple('PatientCoordTuple', ['x', 'y', 'z'])
def voxelCoord2patientCoord(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1] # cri_a 此时是 (x_voxel, y_voxel, z_voxel) 顺序

    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    direction_matrix_a = np.array(direction_a) # 确保是 NumPy 数组

    # 核心转换公式
    coords_xyz = (direction_matrix_a @ (cri_a * vxSize_a)) + origin_a
    return PatientCoordTuple(*coords_xyz)
def patientCoord2voxelCoord(coord_xyz, origin_xyz, vxSize_xyz, direction_a): #物理坐标转换为虚拟坐标
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return VoxelCoordTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))
def importstr(module_str, from_=None):
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)
    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
    return module

def enumerateWithEstimate(      #进度条工具，告诉进度，预估多久完成
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
   
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # ... <1>
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len-start_ndx)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))
