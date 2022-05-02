import cv2
import numpy as np
import matplotlib.pyplot as plt

def matching(des1, des2, kp1, kp2, threshold):
    # # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # # Match descriptors.
    # matches = bf.match(des1,des2)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    
    # Sort them in the order of their distance.
    good = sorted(good, key = lambda x:x[0].distance)

    matches = []
    for pair in good:
        match = np.hstack((kp1[pair[0].queryIdx], kp2[pair[0].trainIdx]))
        matches.append(match)
    
    matches = np.array(matches)
    return matches

def plot_matches(matches, total_img, n):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:n, 1], matches[:n, 0], 'xr')
    ax.plot(matches[:n, 3] + offset, matches[:n, 2], 'xr')
     
    ax.plot([matches[:n, 1], matches[:n, 3] + offset], [matches[:n, 0], matches[:n, 2]], 'r', linewidth=0.5)

    plt.show()