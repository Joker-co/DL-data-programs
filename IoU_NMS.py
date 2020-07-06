import numpy as np
import torch as t

'''
iou(box1, box2): 计算两个矩形框的IoU
输入：box - [x1,y1,x2,y2]
'''

def iou(box1, box2):
    # 取框的坐标
    ax1,ay1,ax2,ay2 = box1
    bx1,by1,bx2,by2 = box2
    # 计算交叠面积坐标与长宽
    xx1 = max(ax1,bx1)
    yy1 = max(ay1,by1)
    xx2 = min(ax2,bx2)
    yy2 = min(ay2,by2)
    w = max(xx2-xx1,0)
    y = max(yy2-yy1,0)
    # 计算各检测框面积与交叠面积
    area1 = (ax2-ax1)*(ay2-ay1)
    area2 = (bx2-bx1)*(by2-by1)
    area_overlap = w*y
    
    return area_overlap / (area1 + area2 - area_overlap)

'''
bbox_overlap：计算若干dets与gt_boxes的两两IoU
输入：dets - (M,4); gt_boxes - (N,4);
输出：metrix_ov - (M,N)
'''

def bbox_overlap(dets, gt_bboxes):
    # 计算各框数量
    M = dets.shape[0]
    N = gt_bboxes.shape[0]
    # 分别计算dets与gt_bboxes的面积
    area_dets = (dets[:,2]-dets[:,0])*(dets[:,3]-dets[:,1]).view(M,1)
    area_gt = (gt_bboxes[:,2]-gt_bboxes[:,0])*(gt_bboxes[:,3]-gt_bboxes[:,1]).view(1,N)
    # 计算交叠面积
    # expand维度，方便比较坐标值
    dets_pro = dets.view(M,1,4).expand(M,N,4)
    gt_pro = gt_bboxes.view(1,N,4).expand(M,N,4)
    ow = (torch.min(dets_pro[:,:,2], gt_pro[:,:,2]) - torch.max(dets_pro[:,:,0], gt_pro[:,:,0]))
    oh = (torch.min(dets_pro[:,:,3], gt_pro[:,:,3]) - torch.max(dets_pro[:,:,1], gt_pro[:,:,1]))
    ow[ow<0] = 0
    oh[oh<0] = 0
    area_ov = ow*oh
    # 计算union面积
    area_union = area_dets + area_gt - area_ov
    return area_ov / area_union

'''
NMS
输入：dets - [K,5], iou_thresh
输出：dets_pro - [K1,5]
'''

def NMS(dets, iou_thresh):
    # 计算检测框数量
    K = dets.shape[0]
    # 获得坐标值与conf
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    conf = dets[:,4]
    # 计算各检测框面积
    area = (x2-x1)*(y2-y1)
    # 获取conf排序后的顺序
    order = conf.argsort()[::-1]
    # keep用于存放保留的检测框idx
    keep = []
    while order.size > 0:
        # 保留置信度最高的检测框idx
        keep.append(order[0])
        # idx_max为置信度最高的检测框序号
        idx_max = order[0]
        # 求idx_max与其他框的iou
        xx1 = np.maximum(dets[idx_max, 0], dets[order[1:], 0])
        yy1 = np.maximum(dets[idx_max, 1], dets[order[1:], 1])
        xx2 = np.minimum(dets[idx_max, 2], dets[order[1:], 2])
        yy2 = np.minimum(dets[idx_max, 3], dets[order[1:], 3])
        w = np.maximum(xx2 - xx1, 0)
        h = np.maximum(yy2 - yy1, 0)
        iou = w*h / (area[idx_max] + area[order[1:]] - w*h)
        print(iou)
        
        idx_re = np.where(iou < iou_thresh)[0]
        order = order[idx_re+1]
    return dets[keep]

list_bbox = [
        [1,1,3,2],
        [2,0,4,3]
        ]

list_dets = [
        [1,1,3,2],
        [5,2,7,5]
        ]

list_gts = [
        [2,0,4,3],
        [4,4,6,6],
        [2,6,4,8]
        ]
# 验证iou()
# print(iou(list_bbox[0], list_bbox[1]))
# 验证bbox_overlap()

list_dets = np.array([[1,1,3,5,0.9],[1.2,1.4,3.6,4.1,0.8],[4.5,3.2,8,6,0.85],[7,1,9,4,0.7]])

# 验证NMS()
# print(NMS(list_dets, 0.5))
