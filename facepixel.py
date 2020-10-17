import numpy as np
import cv2
from matplotlib import pyplot as plt


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cropimage(img,y,x,h,w):
    '''
    returns cropped image
    '''
    #image[start_x:end_x, start_y:end_y]
    crop_img = img[y:y+h, x:x+w]
    return crop_img

def overlayimage(s_img,l_img,x_offset,y_offset):
    '''
    overlays s_img over l_image, with given offsets
    '''
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    return l_img





def pixelfaces(img):
    '''
    find faces and pixel them
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        crop_img = cropimage(img,y,x,h,w)
        ph = int(round(h / 16 ))
        pw = int(round(w / 16 ))
        # Resize input to "pixelated" size
        temp = cv2.resize(crop_img, (pw, ph), interpolation=cv2.INTER_LINEAR)
        output = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        overlayimage(output,img,x,y)
        #show(output)
    

        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    return img


img = cv2.imread('example.jpg')
pixelfaces(img)
show(img)
cv2.imwrite('blurred.jpg',img)
