# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:44:26 2016

@author: Inom Mirzaev

"""
from __future__ import division

import numpy as np
import os , cv2
import pandas as pd

from data_preprocessing import ROWS, COLS
import matplotlib.pyplot as plt


train_data = []

for folder in os.listdir( 'train' ):
    for fname in os.listdir( 'train/'+folder ):
        train_data.append( [folder, 'train/'+folder+'/'+fname] )


train = pd.DataFrame( train_data, columns = ['type' , 'path'] )

np.random.seed( 12 )


fname = train['path'][ np.random.randint( len(train) ) ]

img = cv2.imread( fname , 0 )
#img = cv2.resize( img , ( COLS , ROWS )  , interpolation=cv2.INTER_CUBIC )

clahe = cv2.createCLAHE()

img = clahe.apply(img)

img = cv2.GaussianBlur(img,(5, 5),0)

#img_th = cv2.adaptiveThreshold(img , 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#                                cv2.THRESH_BINARY,11,15)


ret, img_th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#contours = measure.find_contours(img_th, 1.0)
#
#flt_contours = [cont for cont in contours if len(cont)>0  ]
#
#plt.close( 'all')
#fig, ax = plt.subplots()
#ax.imshow(img, 'gray' )
#
#for n, contour in enumerate(flt_contours):
#    ax.plot(contour[:, 1], contour[:, 0], lineCOLS=1)


contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cont_len  =  [len(cont) for cont in contours  ]

mu  = np.mean( cont_len )
sigma = np.std( cont_len )

#filtered_contours = [cont for cont in contours if len(cont)<mu+sigma*0.3 ]

filtered_contours = [cont for cont in contours if len(cont)>10 ]

print mu+sigma*0.3
print mu-sigma*0.3

plt.close('all')


plt.figure()
plt.imshow( img , 'gray' )


plt.figure()
img_new = 255*np.ones_like( img ) 
cv2.drawContours(img_new, filtered_contours ,  -1, (0,255,0), 2)

#img_new = cv2.resize( img_new , ( COLS , ROWS )  , interpolation=cv2.INTER_CUBIC )

plt.imshow( img_new , 'gray')


plt.figure()

img_cont = 255*np.ones_like( img ) 

cv2.drawContours(img_cont, contours ,  -1, (0,255,0), 2)
#ret,img_cont = cv2.threshold(img_cont,127,255,cv2.THRESH_BINARY)

img_cont = cv2.resize( img_cont , ( 240 , 135 )  , interpolation=cv2.INTER_LINEAR )

plt.imshow( img_cont , 'gray')

plt.figure()

img_inv = 255 - img_th

#img_inv = cv2.resize( img_inv , ( COLS , ROWS )  , interpolation=cv2.INTER_CUBIC )

#ret,img_inv = cv2.threshold(img_inv,127,255,cv2.THRESH_BINARY)
 
plt.imshow( img_inv , 'gray' )








