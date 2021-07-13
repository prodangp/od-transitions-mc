# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:14:43 2020

optimized version using numba

N=32
n0 = 3000/1000000
nmax = 35768/1500000


no equilibrium:
N = 128
n0 = 4000000
nmax = 4500000

@author: George Prodan

50000

cub 888 - semitransp puncte galbene albastre
"""


import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import lattice
from numba import njit
import time

N = 256
s = [[[2*random.randint(0,1)-1 for x in range(N)] for y in range(N)] for z in range(int(N))]
J=-1
H=1

n0 = 4000000
nmax = 4500000

i=[0,0,0] 
j=[0,0,0] 
k=[0,0,0]

ip=[0,0,0] 
jp=[0,0,0] 
kp=[0,0,0]

kB=1
T=8
TC=700

avgE_array=[]
avgm_array=[]
t=[]
nr=[]

dif, x_points, y_points, z_points  = lattice.arrange(s, N)           
E=lattice.energy(s,N,J,H) 
s=np.array(s)


@njit(fastmath=True, parallel=True) 
def parameters(s,E,T):
        
        i=np.array([0,0,0]) 
        j=np.array([0,0,0]) 
        k=np.array([0,0,0]) 
        
        ip=np.array([0,0,0]) 
        jp=np.array([0,0,0])  
        kp=np.array([0,0,0]) 
        
        n=0
        E_array=[]
        interchange_distances=[]
        zn_distances=[]
        cu_distances=[]
        posZn=[0,0,0]
        posCu=[0,0,0]
        nr_acc=0
        m_array=[]

        for n in range(nmax+1):
            d=0
            
            i[1] = np.random.randint(0, N-1)   
            j[1] = np.random.randint(0, N-1)   
            k[1] = np.random.randint(0, N-1)
            
            while s[i[1]][j[1]][k[1]]!=1:
                i[1] = np.random.randint(0, N-1)   
                j[1] = np.random.randint(0, N-1)   
                k[1] = np.random.randint(0, N-1)
                                       
            ip[1] = np.random.randint(0, N-1)   
            jp[1] = np.random.randint(0, N-1)   
            kp[1] = np.random.randint(0, N-1)           
                           
            while s[ip[1]][jp[1]][kp[1]]!=-1:
                ip[1] = np.random.randint(0, N-1)   
                jp[1] = np.random.randint(0, N-1)   
                kp[1] = np.random.randint(0, N-1)
                               
            for el in (i,j,k,ip,jp,kp):
                
                el[0]=el[1]-1
                el[2]=el[1]+1
                
                if el[0]==-1:
                    el[0]=N-1
                    
                if el[2]==N:
                    el[2]=0
                           
            neighbours = [[i[2], j[2], k[2]], [i[2], j[2], k[0]], [i[2], j[0], k[2]], [i[2], j[0], k[0]],
                          [i[0], j[2], k[2]], [i[0], j[2], k[0]], [i[0], j[0], k[2]], [i[0], j[0], k[0]]]
            neighboursp = [[ip[2], jp[2], kp[2]], [ip[2], jp[2], kp[0]], [ip[2], jp[0], kp[2]], [ip[2], jp[0], kp[0]],
                          [ip[0], jp[2], kp[2]], [ip[0], jp[2], kp[0]], [ip[0], jp[0], kp[2]], [ip[0], jp[0], kp[0]]]   
        
            s1=s[i[1]][j[1]][k[1]]
            s2=s[ip[1]][jp[1]][kp[1]]
            
            sn1=0
            sn2=0
            
            for neighbour in neighbours:
                
                    sn=s[neighbour[0]][neighbour[1]][neighbour[2]] 
                    
                    sn1=sn1+sn
                    
                    if ip[1]==neighbour[0]:
                        if jp[1]==neighbour[1]:
                            if kp[1]==neighbour[2]:
                                d=2*J*(s1-s2)
    
    
            for neighbour in neighboursp:
               
                    sn=s[neighbour[0]][neighbour[1]][neighbour[2]]                 
                    sn2=sn2+sn
                                          
            dE=J*(sn1-sn2)*(s1-s2)+d
            
            interchange = False
           
            if dE < 0 or dE==0:
                
                interchange=True              
                
            if dE > 0:              
                
                r = random.randint(1, 9999999)
                r=r/10000000
                f=np.exp(-1*dE/(kB*T))
                
                if r<f:
                    interchange=True
                        
                    
            if interchange:
                
                s[i[1]][j[1]][k[1]]=-1*s[i[1]][j[1]][k[1]]
                s[ip[1]][jp[1]][kp[1]]=-1*s[ip[1]][jp[1]][kp[1]]               
                E=E+dE
                nr_acc=nr_acc+1
                
                     
            if n>n0:
                
                m=0
                N2=int(N/2)
                for x in range(0,N2):
                    for y in range(0,N2):
                        for z in range(0,N2):               
                            m=m+s[2*x][2*y][2*z]
                            
            if interchange and n>n0:
                    
                cu=[i[1],j[1],k[1]]
                zn=[ip[1],jp[1],kp[1]]
                    
                interchange_distances.append(lattice.interchange_distance(cu,zn))
                    
                if n:     
                                                                    
                    cu_distances.append(lattice.cu_distance(cu,posCu))          
                    zn_distances.append(lattice.zn_distance(zn,posZn))
                else:  
                    print(T)
                        
                posCu=[i[1],j[1],k[1]]
                posZn=[ip[1],jp[1],kp[1]]
                        
            m_array.append(m)           
            E_array.append(E)
                         
        return s,E,nr_acc, m_array, E_array, interchange_distances,zn_distances,cu_distances
    
while T>0:
          
    s, E, nr_acc, m_array, E_array, interchange_distances,zn_distances,cu_distances = parameters(s,E,T)       
    nr.append(nr_acc)
    avgE=np.mean(E_array)
    avgE_array.append(avgE) 
    avgm=np.mean(m_array)  
    avgm_array.append(avgm) 
       
    lattice.plot_distances(interchange_distances,zn_distances,cu_distances,T)
            
    t.append(T)
    T=T-0.1
    

for sec in range(0,8):
    plt.imshow(s[sec])
    plt.colorbar()
    plt.show()

print(s)        
        
plt.plot(t,avgE_array)
plt.show()


orderp=np.array(avgm_array)
m0=orderp[len(orderp)-1]
orderp=orderp/m0

plt.plot(t,orderp)
plt.show()    

plt.plot(t,nr)
plt.show()



fig = plt.figure()
ax = plt.axes(projection="3d")


cuzn_distribution=lattice.get_distribution(s,N)

            

ax.scatter3D(x_points, y_points, z_points, c=cuzn_distribution)
plt.show()











