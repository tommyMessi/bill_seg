#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:45:18 2018
image
@author: lywen
"""
import cv2
from skimage import measure
from PIL import Image
import numpy as np

def resize_im_unet(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    h, w = im.shape[:2]
    newH, newW = h * f, w * f
    newH = max(64, int(newH / 64 + 1) * 64)
    newW = max(64, int(newW / 64 + 1) * 64)
    fx = w / newW
    fy = h / newH
    return cv2.resize(im, (newW, newH), interpolation=cv2.INTER_AREA), fx, fy
    
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)
    boxed_image = Image.new('RGB', size, (128,128,128))
    px,py = (w-new_w)//2,(h-new_h)//2
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    fx = image_w/new_w
    fy = image_h/new_h
    return boxed_image,px,py,fx,fy


def minAreaRect(points):
    """
    多边形外接矩形
    """
    rect=cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = box.reshape((8,)).tolist()
    box = image_location_sort_box(box)
    return box

def get_rbox(pred,fx,fy,single=True,prob=0.5,minArea=1000):
    pred[pred>prob] = 1
    pred[pred<=prob] =0
    if not single:
        labels=measure.label(pred,connectivity=2)
        regions=measure.regionprops(labels)##计算轮廓
        boxes = []
        for region in regions:
            if region.bbox_area>minArea:
                rbox = minAreaRect(region.coords[:,::-1])
                boxes.append(rbox)
                
        boxes = np.array(boxes)
        if len(boxes)>0:
            boxes[:,[0,2,4,6]] = boxes[:,[0,2,4,6]]*fx
            boxes[:,[1,3,5,7]] = boxes[:,[1,3,5,7]]*fy
        return boxes.tolist()
    else:
        indy,indx = np.where(pred==1)
        n = len(indx)
        points = np.zeros((n,2),dtype=int)
        points[:,0]=indx
        points[:,1]=indy
        box = minAreaRect(points)
        box[0] = box[0]*fx
        box[2] = box[2]*fx
        box[4] = box[4]*fx
        box[6] = box[6]*fx
        box[1] = box[1]*fy
        box[3] = box[3]*fy
        box[5] = box[5]*fy
        box[7] = box[7]*fy

    return [box]


        

from scipy.spatial import distance as dist
def _order_points(pts):
    # 根据x坐标对点进行排序
    """
    --------------------- 
    作者：Tong_T 
    来源：CSDN 
    原文：https://blog.csdn.net/Tong_T/article/details/81907132 
    版权声明：本文为博主原创文章，转载请附上博文链接！
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")

def image_location_sort_box(box):
    x1,y1,x2,y2,x3,y3,x4,y4 = box[:8]
    pts = (x1,y1),(x2,y2),(x3,y3),(x4,y4)
    pts = np.array(pts, dtype="float32")
    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = _order_points(pts)
    return [x1,y1,x2,y2,x3,y3,x4,y4]

def solve(box):
     """
     绕 cx,cy点 w,h 旋转 angle 的坐标
     x = cx-w/2
     y = cy-h/2
     x1-cx = -w/2*cos(angle) +h/2*sin(angle)
     y1 -cy= -w/2*sin(angle) -h/2*cos(angle)
     
     h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
     w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
     (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

     """
     x1,y1,x2,y2,x3,y3,x4,y4= box[:8]
     cx = (x1+x3+x2+x4)/4.0
     cy = (y1+y3+y4+y2)/4.0  
     w = (np.sqrt((x2-x1)**2+(y2-y1)**2)+np.sqrt((x3-x4)**2+(y3-y4)**2))/2
     h = (np.sqrt((x2-x3)**2+(y2-y3)**2)+np.sqrt((x1-x4)**2+(y1-y4)**2))/2   
     #x = cx-w/2
     #y = cy-h/2
     sinA = (h*(x1-cx)-w*(y1 -cy))*1.0/(h*h+w*w)*2
     angle = np.arcsin(sinA)
     return angle,w,h,cx,cy
 

    
def xy_rotate_bbox(cx,cy,w,h,angle):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*sin(angle)+cy
    """
        
    cx    = float(cx)
    cy    = float(cy)
    w     = float(w)
    h     = float(h)
    angle = float(angle)
    x1,y1 = rotate(cx-w/2,cy-h/2,angle,cx,cy)
    x2,y2 = rotate(cx+w/2,cy-h/2,angle,cx,cy)
    x3,y3 = rotate(cx+w/2,cy+h/2,angle,cx,cy)
    x4,y4 = rotate(cx-w/2,cy+h/2,angle,cx,cy)
    xmin = min(x1,x2,x3,x4)
    ymin = min(y1,y2,y3,y4)
    xmax = max(x1,x2,x3,x4)
    ymax = max(y1,y2,y3,y4)
    return [xmin,ymin,xmax,ymax]
 
from numpy import cos,sin
def rotate(x,y,angle,cx,cy):
    angle = angle#*pi/180
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*cos(angle)+cy
    return x_new,y_new
 

def draw_boxes(im, bboxes,color=(0,0,0)):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w, _ = im.shape
    thick = int((h + w) / 300)
    i = 0
    for box in bboxes:
       
        x1,y1,x2,y2,x3,y3,x4,y4 = box[:8]
        cx  =np.mean([x1,x2,x3,x4])
        cy  = np.mean([y1,y2,y3,y4])
        cv2.line(tmp,(int(x1),int(y1)),(int(x2),int(y2)),c,2)
        cv2.line(tmp,(int(x2),int(y2)),(int(x3),int(y3)),c,2)
        cv2.line(tmp,(int(x3),int(y3)),(int(x4),int(y4)),c,2)
        cv2.line(tmp,(int(x4),int(y4)),(int(x1),int(y1)),c,2)
        mess=str(i)
        cv2.putText(tmp, mess, (int(cx), int(cy)),0, 1e-3 * h, c, thick // 2)
        i+=1
    return Image.fromarray(tmp)

