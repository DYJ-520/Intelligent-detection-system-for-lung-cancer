import matplotlib
import matplotlib.pyplot as plt
from dsets import CT, LunaDataset

matplotlib.use('TkAgg')
# 设置绘制图表时要使用的字体，要在图表种显示中文时，一定要设置图表中要使用中文字体，否则中文会乱码
matplotlib.rc("font", family="SimSun")
# 解决坐标轴刻度“负号”乱码问题
matplotlib.rcParams['axes.unicode_minus'] = False
clim = (-1000.0, 300)

def findPositiveSamples(limit=100):
    ds = LunaDataset()
    positiveSample_list = []
    for sample_tup in ds.candidateInfo_list:
        if sample_tup.isNodule_bool:
            positiveSample_list.append(sample_tup)
        if len(positiveSample_list) >= limit:
            break
    return positiveSample_list
def showCandidate(series_uid, batch_ndx=None, **kwargs):
    ds = LunaDataset(series_uid=series_uid, **kwargs)
    pos_list = [i for i, x in enumerate(ds.candidateInfo_list) if x.isNodule_bool]
    if batch_ndx is None:
        if pos_list:
            batch_ndx = pos_list[0]
        else:
            print("Warning: no positive samples found; using first negative sample.")
            batch_ndx = 0
    ct = CT(series_uid)
    ct_t, pos_t, series_uid, center_irc = ds[batch_ndx]
    ct_a = ct_t[0].numpy()
    fig = plt.figure(figsize=(30, 60))
    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('原始CT扫描索引 {}'.format(int(center_irc[0])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct.hu_a[int(center_irc[0])], clim=clim, cmap='gray')
    subplot = fig.add_subplot(len(group_list) + 4, 3, 2)
    subplot.set_title('原始CT扫描行索引 {}'.format(int(center_irc[1])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct.hu_a[:, int(center_irc[1])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('原始CT扫描列索引 {}'.format(int(center_irc[2])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct.hu_a[:, :, int(center_irc[2])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('候选结节的中心切片索引 {}'.format(int(center_irc[0])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct_a[ct_a.shape[0] // 2], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('候选结节的中心切片的行索引 {}'.format(int(center_irc[1])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct_a[:, ct_a.shape[1] // 2, :], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()
    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('候选结节的中心切片的列索引 {}'.format(int(center_irc[2])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct_a[:, :, ct_a.shape[2] // 2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('候选结节的切片索引 {}'.format(index), fontsize=10)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(8)
            plt.imshow(ct_a[index], clim=clim, cmap='gray')
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)  # 调整行间距（值为高度的比例）
    plt.show()
    print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)
