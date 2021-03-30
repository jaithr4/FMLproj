import numpy as np
import pandas as pd
import matplotlib as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

##  -------------------------------------------------
##  Open all of the images and convert to rectangular
##  -------------------------------------------------

##
##  Convert degrees to radians
##
def radicalRadifier(degDat):
    radDat = np.copy(degDat)
    radDat = np.multiply(radDat, 3.14)
    radDat = np.multiply(radDat, 1/360)
    return (radDat)

##
##  Transform the polar coordinate grid to a cartesian grid
##
def cartesianify(refDat):
    k_dat = np.asarray(refDat)
    phi = radicalRadifier(np.copy(k_dat[:,1]))
    theta = radicalRadifier(np.copy(k_dat[:,0]))
    #get r
    r = np.cos(phi)
    #get x,y
    y = np.reshape(np.sin(theta)*r, (-1, 1))
    x = np.reshape(np.cos(theta)*r, (-1,1))
    xyVal = np.hstack((x,y))
    k_dat[:,0:2] = xyVal
    return(k_dat)

##
##  Get rectangular coordinates
##
def coordinator(squished):
    ySorted = squished[squished[:,1].argsort()] #sort by y values
    valOut = np.zeros((np.shape(squished)))
    valOutIndex = 0 #begin iterator
    for x in range(90):
        #grab top first 90 values
        first90 = ySorted[0:90,:]
        xSorted = first90[first90[:,0].argsort()] #sort by x values
        for y in range(90):
            valOut[valOutIndex, :] = [x, y, xSorted[y,2], xSorted[y,3]]
            valOutIndex = valOutIndex + 1
        ySorted = np.copy(ySorted[90:np.size(ySorted, axis=0)])
    return (valOut)
        

##
##  Create a 90x90x2 matrix with the csv data from the file
##
def imageGrabber(refCSVdata):
    k_dat = np.zeros((90,90,2)) #convert to np array
    limit = np.size(refCSVdata, axis = 0)
    for rInd in range(limit):
        rDat = np.copy(refCSVdata[rInd, :])
        k_dat[np.copy(rDat[0]).astype('int'), np.copy(rDat[1]).astype('int')] = np.copy(rDat[2:4])
    return(k_dat)
    
    
##
##  Open up each file, process the data and return a 90x90x2 array
##
def openFile(path):
    inFile = pd.read_csv(path, delimiter = ',')
    ImDat = np.asarray(inFile)
    rectImDat = cartesianify(ImDat)
    rectImDat = coordinator(rectImDat)
    imOut = imageGrabber(rectImDat)
    return (imOut)

##
##  open the image data
##
XComplete = np.zeros((90,90,2,1)) #initialize an array
YComplete = np.zeros((1,2)) 
for azimuth in range(0,360,1):
    for zenith in range(0,30,1):
        path = 'Data with No error/clean/Clean_Rayleigh_scat_Simulation_Sun Zenith-'+str(zenith)+'_Azimuth-'+str(azimuth)+'.txt'
        print(path)
        if os.path.exists(path):
            newDat = np.expand_dims(openFile(path), 3)
            newY = np.asarray([azimuth, zenith])
            newY = np.reshape(newY, (1,2))
            XComplete = np.concatenate(((XComplete, newDat)), axis=3)
            YComplete = np.vstack((YComplete, newY))
XComplete = np.delete(XComplete, 0, axis = 3)
XComplete = np.swapaxes(XComplete, 3, 0)
XComplete = np.swapaxes(XComplete, 1, 2)
YComplete = np.delete(YComplete, 0, axis = 0)