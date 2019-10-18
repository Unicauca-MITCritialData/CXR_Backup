#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:57:16 2019

@author: dmlopez
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:00:57 2019
@author: Brandon
"""
import numpy as np
import math
from scipy import signal
from multiprocessing.dummy import Pool as ThreadPool 

class BokehKernalC:
    
    """ A class for generating bokeh-type blur effect on images """
    def __init__(self, bokeh_type="circular", amount=10, oversample=8, ring_amount=0.1):
        self.size = amount
        self.kernal = np.ndarray([amount, amount])
        self.radius = amount/2.0
        self.sample_precision = 1.0/oversample
        self.oversample = oversample
        self.ring_amount = ring_amount
        self.thread_pool = ThreadPool(3)
        if(bokeh_type is "circular"):
            self.CreateCircularKernal()
        elif(bokeh_type is "square"):
            self.CreateSquareKernal()
        elif(bokeh_type is "hex"):
            self.CreateHexKernal()
            
    def Single2dConvolve(self, image):
        """ Applies a convolution to a single 2d image
        using FFT """
        return np.fft.irfft2( np.fft.rfft2(image) * np.fft.rfft2(self.kernal, image.shape) )
            
    def ApplyConvolution(self, image):
        """DEPRECATED: 
            Traditional convolution is slow for large kernals.
            Rather Use ApplyFFTConvolution"""
        image = np.square(image.astype(float))
        return_arr = np.ndarray(image.shape, float)
        if(image.ndim is 2):
            return self.SingleConvolve(image)
        elif(image.ndim is 3):
            signal.convolve2d(image[:,:,0], self.kernal, mode='same')
            signal.convolve2d(image[:,:,1], self.kernal, mode='same')
            signal.convolve2d(image[:,:,2], self.kernal, mode='same')
        return_arr = np.sqrt(return_arr)
        return np.around(return_arr).astype('B')
    
    def ApplyFFTConvolution(self, image):
        """ Convole the image with bokeh filter using fft method in Single2dConvolve """
        image = np.square(image.astype(float))
        return_arr = np.ndarray(image.shape, float)
        if(image.ndim is 2):
            return_arr = self.Single2dConvolve(image)
        elif(image.ndim is 3):
            im_list = [image[:,:,0],image[:,:,1],image[:,:,2]]
            tmp = self.thread_pool.map(self.Single2dConvolve, im_list)
            return_arr[:,:,0] = tmp[0]
            return_arr[:,:,1] = tmp[1]
            return_arr[:,:,2] = tmp[2]
            self.thread_pool.close() 
            self.thread_pool.join()
        return_arr = np.sqrt(return_arr)
        return np.around(return_arr).astype('B')
    
    def CreateCircularKernal(self):
        critical_dist = math.sqrt(2*(0.5**2))
        for y in range(0, self.size):
            for x in range(0, self.size):
                dist = math.sqrt((y+0.5-self.radius)**2+(x+0.5-self.radius)**2)
                if(dist < (self.radius-critical_dist)):
                    self.kernal[x][y] = 1.0                    
                elif(dist > (self.radius+critical_dist)):
                    self.kernal[x][y] = 0.0
                else:
                    val = 0
                    for i in range(0, self.oversample):
                        for j in range(0, self.oversample):
                            dist_sample = math.sqrt((y+(i+0.5)*self.sample_precision-self.radius)**2 +
                                                    (x+(j+0.5)*self.sample_precision - self.radius)**2)
                            if(dist_sample < self.radius):
                                val += 1.0
                    self.kernal[x][y] = val/(self.oversample ** 2)
        self.kernal = self.kernal/np.sum(self.kernal)
        
    def CreateSquareKernal(self):
        for y in range(0, self.size):
            for x in range(0, self.size):
                self.kernal[x][y] = 1.0
        self.kernal = self.kernal/(self.size ** 2)
        
    def CreateHexKernal(self):
        critical_dist = math.sqrt(2*(0.5**2))
        crit_x = self.radius*np.sqrt(3)/2.0
        for y in range(0, self.size):
            for x in range(0, self.size):
                dist = math.sqrt((y+0.5-self.radius)**2+(x+0.5-self.radius)**2)  
                if(dist < (crit_x-critical_dist) ):
                    self.kernal[y][x] = 1.0
                else:
                    val = 0                              
                    for i in range(0, self.oversample):
                        pos_x = x - self.radius + (i+0.5)*self.sample_precision
                        hex_y = -0.5 * abs(pos_x) + self.radius
                        for j in range(0, self.oversample):
                            pos_y = y - self.radius + (j+0.5)*self.sample_precision                                                       
                            if(abs(pos_x) > crit_x):
                                 val += 0
                            elif(abs(pos_y) > hex_y):
                                 val += 0
                            else:
                                val += 1.0
                    self.kernal[y][x] = val/(self.oversample ** 2)
        self.kernal = self.kernal/np.sum(self.kernal)
        
if __name__ == "__main__":    
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import time
    import PIL

    num_tests = 50         
    img_num = 1
    kern = np.ndarray(shape=[num_tests],dtype=float)
    fft = np.ndarray(shape=[num_tests],dtype=float)
    conv = np.ndarray(shape=[num_tests],dtype=float)
    
    test_image = np.asarray(PIL.Image.open("test_image" + str(img_num) + ".jpg"))
    plt.figure(figsize=(14, 12), dpi= 70, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
   
    # Plot the image for varios Bokeh amounts
    plt.subplot(gs[0])
    imgplot = plt.imshow(test_image)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.show()
    for amount in range(5,5+num_tests):
        start_time = time.clock()    
        bokeh_test = BokehKernalC(bokeh_type="hex", amount=amount, oversample=8)        
        start_time = time.clock()            
        kern[amount-5] = (time.clock()-start_time)
        
        start_time = time.clock()
        bokeh_image = bokeh_test.ApplyFFTConvolution(test_image)
        fft[amount-5] = (time.clock()-start_time)
        
        plt.figure(figsize=(14, 12), dpi= 70, facecolor='w', edgecolor='k')
        plt.subplot(gs[0])
        plt.imshow(bokeh_image)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.subplot(gs[1])
        plt.imshow(bokeh_test.kernal)
        plt.show()

    # PLot loading times for each test
    plt.plot(np.arange(5,5+num_tests), kern)
    plt.plot(np.arange(5,5+num_tests), fft)
    plt.plot(np.arange(5,5+num_tests), conv)
    
    plt.show()