
#import the necessary packages
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
""

img = cv.imread(r"/camcapdata/frame0.jpg")
process_image = img*2
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
plt.imshow(img),plt.colorbar(),plt.show()
plt.imshow(process_image),plt.colorbar(),plt.show()

""

def backremover(img, rect):
    assert img is not None, "file could not be read, check with os.path.exists()"
    process_image = img * 2
    mask = np.zeros(process_image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    print(rect)
    cv.grabCut(process_image,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_new = img*mask2[:,:,np.newaxis]
    return mask2, img_new

rect = (process_image.shape[1]//3, 1, process_image.shape[1]//3, process_image.shape[0]-2)
mask, processed_image = backremover(img, rect)
plt.imshow(mask),plt.colorbar(),plt.show()
plt.imshow(processed_image),plt.colorbar(),plt.show()
""