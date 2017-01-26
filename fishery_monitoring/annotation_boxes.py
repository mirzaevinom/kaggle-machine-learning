# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:06:40 2016

@author: Inom
"""

from __future__ import division

import numpy as np
import os, cv2 , json
import pandas as pd
import matplotlib.pyplot as plt

with open('bounding_boxes/alb_labels.json') as json_data:
    
    json_data = json.load(json_data)
    
    num  = np.random.randint(0, len(json_data) )
    fname = json_data[num]['filename']
    
    img  = cv2.imread( 'train/ALB/'+fname , cv2.IMREAD_COLOR)
    
    plt.close('all')
    
    plt.figure(0)
    
    plt.imshow( img )
    plt.hold(True)
    
    my_list = json_data[0]['annotations']
    print my_list
    for nn in range( len(my_list) ):
        
        my_dict = json_data[0]['annotations'][nn]
        
        x = my_dict['x']
        y = my_dict['y']
        plt.figure(0)
        plt.scatter( x, y, s=40, color='red')
        plt.hold(True)
#        
#        w = my_dict['width']
#        h = my_dict['height']
#        new_img=img[y:y+h,x:x+w]
#        
#        plt.figure()
#        plt.imshow( new_img )
#        
        #plt.savefig('output/cut.png', dpi=400)
        
    plt.figure(0)
    #plt.savefig('output/original.png' , dpi=400)