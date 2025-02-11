import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\massi\OneDrive\Afbeeldingen\Microsat_pictures\Realsat_4.png", cv2.IMREAD_GRAYSCALE)
imcol = cv2.imread(r"C:\Users\massi\OneDrive\Afbeeldingen\Microsat_pictures\Sat_40.png")

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
imcol = cv2.filter2D(imcol, -1, kernel)

""
ret, imgtre = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cnts = cv2.findContours(imgtre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # [-2] indexing takes return value before last (due to OpenCV compatibility issues).
# Find the contour with the maximum area.
c = max(cnts, key=cv2.contourArea)

# Find the minimum area bounding rectangle
# https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)
print(box)
bx_cent = [(box[:,0].max() + box[:,0].min())//2, (box[:,1].max() + box[:,1].min())//2]
print(bx_cent)

# Convert image to BGR (just for drawing a green rectangle on it).
bgr_img = cv2.cvtColor(imgtre, cv2.COLOR_GRAY2BGR)

cv2.drawContours(bgr_img, [box], 0, (0, 255, 0), 2)
cv2.circle(bgr_img,bx_cent, 3, (0,0,0), -1)
# Show images for debugging

plt.plot(), plt.imshow(img, cmap="gray"), plt.title('Original image')
plt.show()
plt.plot(), plt.imshow(bgr_img, cmap="gray"), plt.title('Contoured image')
plt.show()
"""
def Outliner(imcol):
    imgray = cv2.cvtColor(imcol, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    cv2.drawContours(imgray, contours, -1, (255,255,255), 3)

    pa = np.where(c == c[:,0,1].min())[0]
    pa_ = [c[pa][:,0,0].min(), c[pa][:,0,1].max()]
    pb = np.where(c ==c[:,0,1].max())[0]
    pb_ = [c[pb][:,0,0].min(), c[pb][:,0,1].max()]
    pc = np.where(c == c[:,0,0].min())[0]
    pc_ = [c[pc][:,0,0].min(), c[pc][:,0,1].min()]
    pd = np.where(c == c[:,0,0].max())[0]
    pd_ = [c[pd][:,0,0].max(), c[pd][:,0,1].max()]


    pnts = [pa_, pb_, pc_, pd_]

    print(c[:,0,1])
    print(f"c[:,0] min = {c[:,0,0].min()}")
    print(f"index where c[:,0] is max = {pd}")
    print(f"c[:,0] max = {pd_}")
    for i in range(0, len(pnts)):
        cv2.circle(imgray,pnts[i], 2, (255,255,255), -1)
    return imgray

pic = Outliner(imcol)

#plt.plot(), plt.imshow(img, cmap="gray"), plt.title('Original image')
#plt.show()
plt.plot(), plt.imshow(pic), plt.title('Contoured image')
plt.show()

"""