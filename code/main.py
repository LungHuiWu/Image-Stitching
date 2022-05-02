from cv2 import goodFeaturesToTrack
import numpy as np
import cv2, os
import argparse
from DoG import Difference_of_Gaussian, draw_kp
from Feature import get_descriptor
from Matching import matching, plot_matches
from Homography import homography, ransac
from ImageStitching import stitch_img, FindCenter
from tqdm import tqdm
import matplotlib.pyplot as plt

#### 開心開心 ####
def main():
    parser = argparse.ArgumentParser(description = 'evaluation function of Difference of Gaussian')
    parser.add_argument('--threshold', default = 5.0, type=float, help = 'threshold value for feature selection')
    parser.add_argument('--image_path', default = './testdata/1.png', help = 'path to input image')
    parser.add_argument('--center', default = 0, type=int, help='The standard index of the paranoma')
    args = parser.parse_args()

    #### SIFT: Getting Features ####
    print("Feature Extracting ...")
    # create DoG class
    DoG = Difference_of_Gaussian(args.threshold)
    
    images = []
    images_rgb = []
    features = []
    for file in sorted(os.listdir(args.image_path)):
        file_name = os.path.join(args.image_path,file)
        img = cv2.imread(file_name,0).astype(np.float32)
        images.append(img)
        img_rgb = cv2.imread(file_name)
        images_rgb.append(img_rgb)
        # find keypoint from DoG and sort it
        kp_fname = 'npy/' + file[:-4] + '.npy'
        if os.path.isfile(kp_fname):
            keypoints = np.load(kp_fname)
        else:
            keypoints = DoG.get_keypoints(img)
            np.save('npy/' + file[:-4], keypoints)
            print('File save to:', file[:-4], '.npy')
        # print(keypoints.shape)
        features.append(keypoints)
        
    #### Get Descriptor ####
    descriptor = []
    for i in range(len(images)):
        des_fname = 'descriptor/' + str(i) + '.npy'
        if os.path.isfile(des_fname):
            des = np.load(des_fname, allow_pickle=True)
        else:
            des = get_descriptor(images[i], features[i])
            np.save('descriptor/' + str(i), des)
            print('File save to:', str(i), '.npy')
        # print(des.shape)
        descriptor.append(des.astype(np.float32))
    
    #### Feature Matching ####
    print("Feature Matching ...")
    # good = matching(descriptor[0], descriptor[1], features[0], features[1], 0.65)
    inliers = []
    for i in range(len(images)-1):
        good = matching(descriptor[i], descriptor[i+1], features[i], features[i+1], 0.65)
        inliers.append(good)
    
    #### Homography ####
    print("Calculating Homography ...")
    H, mask_good = homography(good)
    H_mat = []
    for good in inliers:
        H, mask_good = homography(good)
        H_mat.append(H)
    # print(H)
    # total_img = np.concatenate((images[0], images[1]), axis=1)
    # plot_matches(inliers, total_img, -1) # Good mathces
    
    #### Center ####
    print("Locating Center Image ...")
    mat, mat_inv = FindCenter(H_mat, args.center)
    
    #### Image Stitching ####
    print("Stiching Image ...")
    left_rgb = images_rgb[0]
    right_rgb = images_rgb[1]
    # cv2.imshow('Image', left_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    stitch_name = '../result/result.jpg'
    if os.path.isfile(stitch_name):
        result = cv2.imread(stitch_name)
    else:
        # result = stitch_img(left_rgb, right_rgb, H, "noBlending")
        result = stitch_img(images_rgb, mat, mat_inv, "noBlending")
        cv2.imwrite(stitch_name, result)
        print('Image save to: result.jpg')
    # cv2.imshow('Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
