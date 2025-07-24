import os
import numpy as np
import torch
import zipfile
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from segmentModel import UNetWrapper # 假设这些模块兼容CPU
from model import LunaModel # 假设这些模块兼容CPU
from TumorModel import LunaModel as TumorLunaModel # 假设这些模块兼容CPU

from util.util import voxelCoord2patientCoord # 假设这些模块兼容CPU

import scipy.ndimage.measurements as measurements
from TumorDatasets import CandidateInfoTuple # 假设这些模块兼容CPU
import scipy.ndimage.morphology as morphology
from segmentDsets import Luna2dSegmentationDataset # 假设这些模块兼容CPU
from torch.utils.data import DataLoader
from TumorDatasets import LunaDataset # 假设这些模块兼容CPU
from skimage.measure import marching_cubes
from Database import insert_ct_record
from TumorDatasets import getCt
import json
from skimage.transform import resize

import numpy as np


app = Flask(__name__)  #采用flask web框架
CORS(app, resources={r"/*": {"origins": "*"}})  # 允许所有来源访问所有路由
app.config['UPLOAD_FOLDER'] = 'data-unversioned/data/temp'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 模型路径配置
MODEL_PATHS = {
    "unet": "data-unversioned/seg/models/seg/seg_2025-07-01_15.35.13_none.best .state",
    "nodule_cls": "data-unversioned/nodule/models/nodule-model/cls_2025-06-30_14.27.22_nodule-comment.best.state",
    "malignancy_cls": "data-unversioned/tumor/models/tumor_cls/cls_2025-07-04_10.55.21_finetune-depth2.best.state"
}

def load_models():
    """加载所有预训练模型"""
    # 将设备设置为 "cpu"
    device = torch.device("cpu")

    # 加载U-Net分割模型
    unet_model = UNetWrapper(
        in_channels=7,
        n_classes=1,
        depth=3,
        wf=4,
        padding=True,
        batch_norm=True,
        up_mode='upconv',
    )
    # 强制加载到CPU
    unet_dict = torch.load(MODEL_PATHS["unet"], map_location=device)
    unet_model.load_state_dict(unet_dict['model_state'])
    unet_model.eval()   #评估模式，冻结模型参数变化，批量归一化均值与方差采用以往数据

    # 加载结节分类模型
    nodule_model = LunaModel()
    # 强制加载到CPU
    nodule_dict = torch.load(MODEL_PATHS["nodule_cls"], map_location=device)
    nodule_model.load_state_dict(nodule_dict['model_state'])
    nodule_model.eval()

    # 加载恶性分类模型
    malignancy_model = TumorLunaModel()
    # 强制加载到CPU
    malignancy_dict = torch.load(MODEL_PATHS["malignancy_cls"], map_location=device)
    malignancy_model.load_state_dict(malignancy_dict['model_state'])
    malignancy_model.eval()

    return unet_model, nodule_model, malignancy_model


@app.route('/health', methods=['GET'])
def health_check(): #API接口
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': '2025-07-03T16:38:58Z',
        'service': 'CT肺结节诊断系统'
    }), 200

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """处理CT影像预测请求"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    zip = request.files['file']
    if zip.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # 保存上传文件，接收和处理上传的 CT 影像 ZIP 文件，包括保存、解压、识别序列 ID，然后利用加载的 CT 数据进行肺结节的诊断分析，并将结果返回给前端
    zipname = secure_filename(zip.filename)
    file_name = os.path.splitext(zipname)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
    zip.save(zip_path)
    extract_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"extract_{timestamp}")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    # 查找 MHD 和 RAW 文件（这里为了方便只查找了MHD文件的路径）
    mhd_files = [f for f in os.listdir(extract_dir) if f.endswith('.mhd')]
    if not mhd_files:
        raise FileNotFoundError("未找到 MHD 文件")

    mhd_file = mhd_files[0]
    series_uid = os.path.splitext(mhd_file)[0]  # 提取 series_uid
    # 构建完整路径
    mhd_path = os.path.join(extract_dir, mhd_file)
    raw_file = mhd_file.replace('.mhd', '.raw')
    raw_path = os.path.join(extract_dir, raw_file) if os.path.exists(os.path.join(extract_dir, raw_file)) else None
    # 插入数据库（同时保存两个文件路径）
    insert_ct_record(series_uid, mhd_path, raw_path)

    try:
        ct = getCt(series_uid)
        results = process_ct(ct, series_uid)
        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc() # 打印完整的错误堆栈到控制台
        return jsonify({"error": str(e), "message": "An error occurred during CT processing."}), 500


def create_lung_mesh(mask, spacing, origin_xyz1, direction_a, downsample=2): # 签名已包含 direction_a
    # 1. 降采样 mask，对（是否是肺部）掩膜值缩小。比如原来512*512*128，那现在如果缩小的值是2，那么就是256*256*64。然后对于新的图像具体的体素值是对少，采用插值的方法。就是说距离谁更近就是多少，一样近通过别的方法判断。
    mask_small = resize(
        mask,
        (mask.shape[0] // downsample, mask.shape[1] // downsample, mask.shape[2] // downsample),
        order=0, preserve_range=True, anti_aliasing=False
    ).astype(np.uint8)
    # 2. marching_cubes 用 spacing=1，得到降采样体素坐标
    verts, faces, normals, values = marching_cubes(mask_small, level=0.5, spacing=(1, 1, 1))
    # 3. verts: (z, y, x) 体素坐标，转为 (x, y, z)
    verts_xyz = verts[:, [2, 1, 0]] # N x 3 array where each row is (x_voxel, y_voxel, z_voxel)


    spacing_a = np.array(spacing)
    origin_a = np.array(origin_xyz1)
    direction_matrix_a = np.array(direction_a)

    # 计算体素位移
    verts_displacement_voxel_units = verts_xyz * downsample * spacing_a

    # 应用方向矩阵
    transformed_displacement = (direction_matrix_a @ verts_displacement_voxel_units.T).T
    verts_world = transformed_displacement + origin_a


    verts_flat = verts_world.flatten().tolist()   #转成一维的 Python 列表，方便3D渲染
    faces_flat = faces.flatten().tolist()
    return verts_flat, faces_flat   #返回扁平化的


def process_ct(ct, series_uid):
    unet_model, nodule_model, malignancy_model = load_models()
    print("开始CT影像分割...")
    mask = segmentCt(ct, series_uid, unet_model)
    print("CT影像分割完成。")

    print("开始生成肺部3D网格...")
    # 调用 create_lung_mesh 获得肺部网格的初始世界坐标
    lung_verts, lung_faces = create_lung_mesh(mask, ct.vxSize_xyz, ct.origin_xyz, ct.direction_a)
    print("肺部3D网格生成完成。")

    # === 修正：计算一个中心偏移量，并应用于肺部和结节 ===
    # 1. 将肺部顶点列表转换为 NumPy 数组，以便进行数值操作
    #    -1 表示 NumPy 自动推断行数，3 表示每行有3个元素 (x, y, z)
    lung_verts_np = np.array(lung_verts).reshape(-1, 3)

    # 2. 计算肺部模型在世界坐标系中的边界框
    #    np.min(axis=0) 找到每列（X, Y, Z轴）的最小值
    #    np.max(axis=0) 找到每列（X, Y, Z轴）的最大值
    min_xyz = np.min(lung_verts_np, axis=0)
    max_xyz = np.max(lung_verts_np, axis=0)

    # 3. 计算肺部模型的几何中心点
    #    这是 (最小坐标 + 最大坐标) / 2
    lung_geometric_center_xyz = (min_xyz + max_xyz) / 2

    # 4. 将肺部网格顶点平移，使其围绕其几何中心在 (0,0,0) 处
    #    从每个顶点的坐标中减去肺部模型的几何中心，实现居中
    lung_verts_centered = lung_verts_np - lung_geometric_center_xyz
    #    将处理后的 NumPy 数组重新展平为列表，以符合 JSON 格式
    lung_verts = lung_verts_centered.flatten().tolist()

    candidates = groupSegmentationOutput(series_uid, ct, mask)  # 第结节分类模型
    classifications_list = classifyCandidates(ct, candidates, nodule_model, malignancy_model)   # 恶性模型
    nodules = [item for item in classifications_list if item[0] > 0.5]
    result = []  # 列表，每个元素都是字典

    for idx, (prob_nodule, prob_mal, center_xyz, center_irc) in enumerate(nodules):
        # 结节坐标在之前已经通过 voxelCoord2patientCoord 转换为 PatientCoordTuple
        # 将其转换为 NumPy 数组，以便进行数值操作
        nodule_pos_array = np.array([center_xyz.x, center_xyz.y, center_xyz.z])

        # === 修正：将结节位置也减去相同的偏移量 (肺部几何中心) ===
        # 确保结节与肺部在 Three.js 渲染空间中保持相对位置的正确性
        nodule_pos_centered = nodule_pos_array - lung_geometric_center_xyz
        # ======================================================

        result.append({
            "position": nodule_pos_centered.tolist(),  # 使用修正后的结节位置 (转换为列表)
            "nodule_prob": float(prob_nodule),
            "malignancy_prob": float(prob_mal)
        })
    Json_str = {
        "nodules": result,
        "lung_mesh": {
            "vertices": lung_verts,
            "faces": lung_faces
        }
    }
    return Json_str


def classifyCandidates(ct, candidateInfo_list, nodule_model, malignancy_model):
    cls_dl = initClassificationDl(candidateInfo_list)
    classifications_list = []

    for batch_ndx, batch_tup in enumerate(cls_dl):
        input_t, _, _, series_list, center_list = batch_tup
        # 转移到CPU
        input_g = input_t.to("cpu")
        with torch.no_grad():
            _, probability_nodule_g = nodule_model(input_g)      #是结节的概率
            if malignancy_model is not None:
                _, probability_mal_g = malignancy_model(input_g)    #恶性概率
            else:
                probability_mal_g = torch.zeros_like(probability_nodule_g)  #默认为0
        zip_iter = zip(center_list,
            probability_nodule_g[:,1].tolist(),
            probability_mal_g[:,1].tolist())    #打包成元组
        for center_irc, prob_nodule, prob_mal in zip_iter:
            center_xyz = voxelCoord2patientCoord(center_irc,
                direction_a=ct.direction_a,
                origin_xyz=ct.origin_xyz,
                vxSize_xyz=ct.vxSize_xyz,
            )
            cls_tup = (prob_nodule, prob_mal, center_xyz, center_irc)   #物理位置与体素位置
            classifications_list.append(cls_tup)
    return classifications_list

def initClassificationDl(candidateInfo_list):   #初始化分类数据
    cls_ds = LunaDataset(
            sortby_str='series_uid',
            candidateInfo_list=candidateInfo_list,
        )
    cls_dl = DataLoader(
        cls_ds,
        batch_size=16,
        num_workers=0
    )
    return cls_dl


def segmentCt(ct, series_uid, unet_model):
    with torch.no_grad():
        output_a = np.zeros_like(ct.hu_a, dtype=np.float32) #同大小数组
        seg_dl = initSegmentationDl(series_uid)
        for input_t, _, _, slice_ndx_list in seg_dl:
            input_g = input_t.to("cpu")
            prediction_g = unet_model(input_g)
            for i, slice_ndx in enumerate(slice_ndx_list):  #把2D图像组装成3D图像
                output_a[slice_ndx] = prediction_g[i].cpu().numpy()
        mask_a = output_a > 0.5
        mask_a = morphology.binary_erosion(mask_a, iterations=1)    #侵蚀
    return mask_a

def initSegmentationDl(series_uid):
    seg_ds = Luna2dSegmentationDataset(
            contextSlices_count=3,
            series_uid=series_uid,
            fullCt_bool=True,
        )

    seg_dl = DataLoader(
        seg_ds,
        batch_size=4,
        num_workers=0
    )
    return seg_dl


def groupSegmentationOutput(series_uid, ct, clean_a):   #把可疑结节中心坐标保留，并把逻辑坐标转成物理坐标
    candidateLabel_a, candidate_count = measurements.label(clean_a)
    centerIrc_list = measurements.center_of_mass(
        ct.hu_a.clip(-1000, 1000) + 1001,
        labels=candidateLabel_a,
        index=np.arange(1, candidate_count+1),
    )
    candidateInfo_list = []
    for i, center_irc in enumerate(centerIrc_list):
        center_xyz = voxelCoord2patientCoord(
            center_irc,
            ct.origin_xyz,
            ct.vxSize_xyz,
            ct.direction_a,
        )
        assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])   #检查虚拟坐标和物理坐标是否有效
        assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])
        candidateInfo_tup =  CandidateInfoTuple(False, False, False, 0.0, series_uid, center_xyz)
        candidateInfo_list.append(candidateInfo_tup)
    return candidateInfo_list


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)