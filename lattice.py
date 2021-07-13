# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:29:27 2020

@author: 40724
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import njit


def arrange(s,N):
    
    dif=0
    x_points=[]
    y_points=[]
    z_points=[]

    for x in range(0,N):
        for y in range (0,N):
            for z in range(0,N):
                if x%2==0 and y%2==0 and z%2==0 or (x*y*z)%2==1:
                    dif=dif+s[x][y][z]
                    x_points.append(x)
                    y_points.append(y)
                    z_points.append(z)            
                else:
                    s[x][y][z]=0
                
   


    while dif:
        for x in range(0,N):
            for y in range (0,N):
                for z in range(0,N):
                    if dif<0:
                        if s[x][y][z]==-1:
                                s[x][y][z]=-1*s[x][y][z]
                                dif=dif+2
                    if dif>0:
                        if s[x][y][z]==1:
                                s[x][y][z]=-1*s[x][y][z]
                                dif=dif-2  
                                
    return dif, x_points, y_points, z_points



def energy(s,N,J,H):
    E=0 
    i = [0,0,0]
    j = [0,0,0]
    k = [0,0,0]
    for x in range(0,N):
        for y in range(0,N):
            for z in range(0,N):
                i[1]=x
                j[1]=y
                k[1]=z
                i[2]=i[1]+1
                i[0]=i[1]-1 
                j[2]=j[1]+1
                j[0]=j[1]-1
                k[2]=k[1]+1
                k[0]=k[1]-1
            
            for el in (i,j,k):
            
                if el[0]==-1:
                    el[0]=N-1
                if el[2]==N:
                    el[2]=0
            
            neighbours = [[i[2], j[2], k[2]], [i[2], j[2], k[0]], [i[2], j[0], k[2]], [i[2], j[0], k[0]],
                          [i[0], j[2], k[2]], [i[0], j[2], k[0]], [i[0], j[0], k[2]], [i[0], j[0], k[0]]]
            
            E=E-H*s[x][y][z]
            
            for neighbour in neighbours:
                E=E-J*s[x][y][z]*s[neighbour[0]][neighbour[1]][neighbour[2]]
                   
    return E/2


def order_parameter(s,N):
    m=0
    N2=int(N/2)
    for x,vx in np.ndenumerate(range(0,N2)):
        for y,vy in np.ndenumerate(range(0,N2)):
            for z,vz in np.ndenumerate(range(0,N2)):               
                    m=m+s[2*x][2*y][2*z]  
    return m
                    
          
def plot_distances(interchange_distances,zn_distances,cu_distances,T):
    
       
        bins = 30     
        figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,8))
        
        
        
        axes[0].hist(interchange_distances, bins, color = 'green', 
            histtype = 'bar', rwidth = 0.8)  
        axes[0].set_xlabel('Interchange distance') 
        axes[0].set_ylabel('No. of pairs') 
        axes[0].set_title('Interchange distances at temperature ' +str(T)+ " K") 
          
        
        axes[1].hist(zn_distances, bins, color = 'green', 
            histtype = 'bar', rwidth = 0.8) 
        axes[1].set_xlabel('Distance') 
        axes[1].set_ylabel('Probability') 
        axes[1].set_title('Distances between subsequent modified Zn sites at temperature ' +str(T)+ " K") 
         
        
        axes[2].hist(cu_distances, bins, color = 'green', 
            histtype = 'bar', rwidth = 0.8) 
        axes[2].set_xlabel('Distance') 
        axes[2].set_ylabel('Probability') 
        axes[2].set_title('Distances between subsequent modified Cu sites at temperature ' +str(T)+ " K") 
        
    
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=0.8)
        figure.show()
        plt.show()                    
                    
                    
def get_distribution(s,N): 
    
    cuzn_distribution=[]
    for x in range(0,N):
        for y in range(0,N):
            for z in range(0,N):
                 if x%2==0 and y%2==0 and z%2==0 or (x*y*z)%2==1:
                    cuzn_distribution.append(s[x][y][z])                      
                    
    return cuzn_distribution               
                    
@njit(fastmath=True)                  
def interchange_distance(cu,zn):                  
    return np.sqrt((cu[0]-zn[0])^2+(cu[1]-zn[1])^2+(cu[2]-zn[2])^2)                 
                    
@njit(fastmath=True)                       
def cu_distance(cu, posCu):
    return np.sqrt((cu[0]-posCu[0])^2+(cu[1]-posCu[1])^2+(cu[2]-posCu[2])^2)                    

@njit(fastmath=True)                    
def zn_distance(zn, posZn):
    return np.sqrt((zn[0]-posZn[0])^2+(zn[1]-posZn[1])^2+(zn[2]-posZn[2])^2)                       
                                
               
                    
                    