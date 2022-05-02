from cv2 import threshold
import numpy as np
import cv2
from scipy import ndimage
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 3
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_images(self, image):
        
        pool = []
        
        h,w = image.shape[0], image.shape[1]
        for _ in range(4):
            g_image = cv2.GaussianBlur(image, ksize = (0,0), sigmaX = self.sigma)
            pool.append(cv2.subtract(image, g_image))
            image = g_image
        image = cv2.resize(image, (w//2,h//2), interpolation=cv2.INTER_NEAREST)
        for _ in range(4):
            g_image = cv2.GaussianBlur(image, ksize = (0,0), sigmaX = self.sigma)
            pool.append(cv2.subtract(image, g_image))
            image = g_image
            
        return pool

    def get_keypoints(self, image):
        ### TODO ####
        keypoints = []
        fp = np.ones((3,3,3))
        fp[1,1,1] = 0
        
        for oct in range(self.num_octaves):
            
            scale = 2**oct
            h, w = image.shape[0]//scale, image.shape[1]//scale
            
            image = cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)
            # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
            # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
            gaussian_images = [cv2.GaussianBlur(image, ksize = (0,0), sigmaX = self.sigma ** i) for i in range(self.num_guassian_images_per_octave)]
            gaussian_images[0] = image
            
            # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
            # - Function: cv2.subtract(second_image, first_image)
            dog_images = [cv2.subtract(gaussian_images[j], gaussian_images[j+1]) for j in range(self.num_DoG_images_per_octave)]

            # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
            #         Keep local extremum as a keypoint
            dog = np.array(dog_images).transpose(1, 2, 0)
            # print(dog.max(), dog.min())
            # local_max = ndimage.maximum_filter(dog,footprint=fp,mode='mirror')
            # local_min = ndimage.minimum_filter(dog,footprint=fp,mode='mirror')
            # max_kp = np.argwhere(local_max == dog)
            # min_kp = np.argwhere(local_min == dog)
            # for x,y,_ in max_kp:
            #     keypoints.append([x*scale,y*scale])
            # for x,y,_ in min_kp:
            #     keypoints.append([x*scale,y*scale])
                
            for i in trange(1,h-1):
                for j in range(1,w-1):
                    for k in range(1,self.num_DoG_images_per_octave-1):
                        box = dog[i-1:i+2,j-1:j+2,k-1:k+2]
                        center = dog[i,j,k]
                        local_max, local_min = box.max(), box.min()
                        if center >= local_max and abs(center) > self.threshold:
                            keypoints.append([i*scale,j*scale])
                        if center <= local_min and abs(center) > self.threshold:
                            keypoints.append([i*scale,j*scale])
                            
            image = gaussian_images[self.num_guassian_images_per_octave - 1]
        
        if len(keypoints)==0:
            return np.array([])
        
        keypoints = np.array(keypoints)

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints

def draw_kp(image, kp):
    fig, ax = plt.subplots()
    ax.imshow(np.array(image).astype('uint8'))
    for x,y in kp:
        ax.plot(y, x, 'xr')
    plt.show()