import numpy as np
import cv2
from scipy.fft import dst

def homography(pairs):
    # rows = []
    # for i in range(pairs.shape[0]):
    #     p1 = np.append(pairs[i][0:2], 1)
    #     p2 = np.append(pairs[i][2:4], 1)
    #     row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
    #     row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
    #     rows.append(row1)
    #     rows.append(row2)
    # rows = np.array(rows)
    # _, _, V = np.linalg.svd(rows)
    # H = V[-1].reshape(3, 3)
    # H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    src_pts = np.float32(pairs[:,0:2]).reshape(-1,1,2)
    dst_pts = np.float32(pairs[:,2:4]).reshape(-1,1,2)
    # print(src_pts, dst_pts)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    # print(M)
    return M, matchesMask

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        points = matches[:4,:]
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H