# Image Stitching

### Execution

> cd code
>
> python main.py --threshold <DoG Threshold> --image_path <Image Path> --center <target image index>

Example:

> python main.py --threshold 5 --image_path MountainView --center 3

For this example, the DoG(Difference of Gaussian) algorithm takes 5 as its threshold.

It computes all images in the folder "MountainView", in which all images are ordered by their file names.

This folder contains 6 images, the --center argument take the third image as the target coordinate space.

### Demo

##### Input Data

<img src="/Users/LungHuiWu/Dropbox/Mac/Desktop/數位視覺效果/hw2_[76]/MountainView/DSC02958.JPG" alt="DSC02958" style="zoom: 2.4%;" /><img src="/Users/LungHuiWu/Dropbox/Mac/Desktop/數位視覺效果/hw2_[76]/MountainView/DSC02959.JPG" alt="DSC02959" style="zoom:2.4%;" /><img src="/Users/LungHuiWu/Dropbox/Mac/Desktop/數位視覺效果/hw2_[76]/MountainView/DSC02960.JPG" alt="DSC02960" style="zoom:2.4%;" /><img src="/Users/LungHuiWu/Dropbox/Mac/Desktop/數位視覺效果/hw2_[76]/MountainView/DSC02961.JPG" alt="DSC02961" style="zoom:2.4%;" /><img src="/Users/LungHuiWu/Dropbox/Mac/Desktop/數位視覺效果/hw2_[76]/MountainView/DSC02962.JPG" alt="DSC02962" style="zoom:2.4%;" /><img src="/Users/LungHuiWu/Dropbox/Mac/Desktop/數位視覺效果/hw2_[76]/MountainView/DSC02963.JPG" alt="DSC02963" style="zoom:2.4%;" />

##### Output Panorama

![result](/Users/LungHuiWu/Dropbox/Mac/Desktop/數位視覺效果/hw2_[76]/result.png)

