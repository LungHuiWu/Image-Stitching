from turtle import st
import cv2
from tqdm import trange, tqdm
import numpy as np

def stitch_img(images, mat, mat_inv, blending_mode):
    
    # Convert to double and normalize. Avoid noise.
    # left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    # right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    shapes = [img.shape[:2] for img in images]
    
    # if (blending_mode == "noBlending"):
    #     stitch_img[:hl, :wl] = left
    corners = []
    for idx, (h,w) in enumerate(shapes):
        corner_np = np.float32([ [0,0],[h-1,0],[h-1,w-1],[0,w-1] ]).reshape(-1,1,2)
        inv_H = mat_inv[idx]
        corner_dst = cv2.perspectiveTransform(corner_np,inv_H)
        corners.append(corner_dst)
    corners = np.array(corners).reshape(-1,1,2)
    
    t,l = corners.min(axis=0)[0].astype(int)
    b,r = corners.max(axis=0)[0].astype(int)
    
    x_offset = 0
    y_offset = 0
    if t < 0:
        x_offset = -t
    if l < 0:
        y_offset = -l
    offset = np.array([x_offset, y_offset])
    
    new_h, new_w = b-t, r-l
    stitch_img = np.zeros((new_h, new_w, 3), dtype="int")
    
    # for idx,H_inv in enumerate(mat_inv):
    #     for i in trange(images[idx].shape[0]):
    #         for j in range(images[idx].shape[1]):
    #             coor = (np.float32([[i,j]])-offset).reshape(-1,1,2)
    #             new_x, new_y = cv2.perspectiveTransform(coor, H_inv)[0][0].astype(int)
    #             if(new_x < 0 or new_x >= new_h or new_y < 0 or new_y >= new_w):
    #                 continue
    #             stitch_img[new_x, new_y, :] = images[idx][i,j,:]
    
    for i in trange(stitch_img.shape[0]):
        for j in range(stitch_img.shape[1]):
            coor = (np.float32([[i,j]])-offset).reshape(-1,1,2)
            for idx,H in enumerate(mat):
                new_x, new_y = cv2.perspectiveTransform(coor, H)[0][0].astype(int)
                if(new_x < 0 or new_x >= shapes[idx][0] or new_y < 0 or new_y >= shapes[idx][1]):
                    continue
                stitch_img[i,j,:] = images[idx][new_x, new_y, :]
            
    return stitch_img
    
def FindCenter(H_mat, center):
    H_mat.insert(center, np.eye(3))
    for i in range(center):
        H_mat[i] = np.linalg.inv(H_mat[i])
    output = [np.eye(3) for _ in range(len(H_mat))]
    beg, end = center, center
    while beg > 0:
        beg -= 1
        output[beg] = H_mat[beg] @ output[beg+1]
    while end < (len(H_mat)-1):
        end += 1
        output[end] = H_mat[end] @ output[end-1]
    output_inv = [np.linalg.inv(mat) for mat in output]
    return output, output_inv

# def stitch_img(left, right, H, blending_mode):
        
#     # Convert to double and normalize. Avoid noise.
#     # left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
#     # Convert to double and normalize.
#     # right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    
#     (hl, wl) = left.shape[:2]
#     (hr, wr) = right.shape[:2]
    
#     # if (blending_mode == "noBlending"):
#     #     stitch_img[:hl, :wl] = left
        
#     right_np = np.float32([ [0,0],[hr-1,0],[hr-1,wr-1],[0,wr-1] ]).reshape(-1,1,2)
#     inv_H = np.linalg.inv(H)
#     right_dst = cv2.perspectiveTransform(right_np,inv_H)
    
#     t,l = right_dst.min(axis=0)[0].astype(int)
#     b,r = right_dst.max(axis=0)[0].astype(int)
    
#     x_offset = 0
#     y_offset = 0
#     if t < 0:
#         x_offset = -t
#     if l < 0:
#         y_offset = -l
#     offset = np.array([x_offset, y_offset])
    
#     new_h, new_w = max(b,hl)-min(t,0), max(r,wl)-min(l,0)
#     stitch_img = np.zeros((new_h, new_w, 3), dtype="int")
#     stitch_img[x_offset:hl+x_offset, y_offset:wl+y_offset ,:] = left
#     for i in trange(stitch_img.shape[0]):
#         for j in range(stitch_img.shape[1]):
#             coor = (np.float32([[i,j]])-offset).reshape(-1,1,2)
#             new_x, new_y = cv2.perspectiveTransform(coor, H)[0][0].astype(int)
#             if(new_x < 0 or new_x >= hr or new_y < 0 or new_y >= wr):
#                 continue
#             stitch_img[i,j,:] = right[new_x, new_y, :]
            
            
#     return stitch_img