import numpy as np

def get_descriptor(image, feature):
    descriptor = []
    h,w = image.shape
    for x,y in feature:
        if x<9 or y<9:
            continue
        if x>(h-9) or y>(w-9):
            continue
        block = image[x-8:x+8, y-8:y+8]
        gx = image[x-7:x+9, y-8:y+8] - image[x-9:x+7, y-8:y+8] + 1e-8
        gy = image[x-8:x+8, y-7:y+9] - image[x-8:x+8, y-9:y+7]
        mag = np.sqrt(gx**2 + gy**2)
        phi = np.arctan(gy/gx) + np.pi/2
        phi = np.round(phi / (np.pi/4)).astype(int)
        des = np.zeros((16,8))
        for i in range(16):
            for j in range(16):
                m,n = i//4, j//4
                num = m*4+n
                des[num, phi[i,j]] += mag[i,j]
        des = des.flatten()
        
        if len(des) != 128:
            continue

        descriptor.append(des)
    return np.array(descriptor)