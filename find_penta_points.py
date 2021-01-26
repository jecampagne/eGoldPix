#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

from math import sin,cos,acos,sqrt,pi, atan2

import pandas as pd

get_ipython().run_line_magic('matplotlib', '')


# In[ ]:


self.xoff7+i*1/n+j*1/(2*n), self.yoff7+j*sqrt(3)/(2*n)


# In[36]:


def xycent(xoff7,yoff7,i,j,n):
    return xoff7+i/n+j/(2*n), yoff7+j*sqrt(3)/(2*n)


# In[37]:


def barycentricCoords(p):
    '''
        input: 'p'is are the position vector of the form [x,y]'. 
        output: l1,l2,l3 are the barycentric co-ordsinates.
    
        ex:
        barycentricCoords([1,2])
        (-1.6547005383792517, 0.3452994616207483, 2.3094010767585034)
        
        D'une maniere generale
        p1=(x1,y1), p2=(x2,y2) p3=(x3,y3)
        T= [[x1-x3,x2-x3],[y1-y3,y2-y3]]
        (l1,l2) = T^(-1) . ( (x,y)-p3 )
        l3 = 1-l2-l3
    '''
    x,y = p[0],p[1]
    # l3*sqrt(3)/2 = y
    l3 = y*2./sqrt(3.)
    # l1 + l2 + l3 = 1
    # 0.5*(l2 - l1) = x
    l2 = x + 0.5*(1 - l3)
    l1 = 1 - l2 - l3
    return l1,l2,l3


# In[38]:


icoTriangs = [(0,i+1,(i+1)%5+1) for i in range(5)] +             [(6,i+7,(i+1)%5+7) for i in range(5)] +             [(i+1,(i+1)%5+1,(7-i)%5+7) for i in range(5)] +             [(i+1,(7-i)%5+7,(8-i)%5+7) for i in range(5)]
icoTriangs=np.array(icoTriangs)
icoTriangs


# In[39]:


def scalProd(p1,p2):
    '''
        input: p1 and p2 are the vetors of form [x0,x1,...,xn]'
        output: is the scalar product of p1 and p2.
    '''
    return sum([p1[i]*p2[i] for i in range(len(p1))])


# In[40]:


def slerp(p0,p1,t):
    '''
        program outputs the spherical linear interpolation 
        of arc defined by p0, p1(around origin).  
        
        input: t=0 -> p0, t=1 -> p1. 
                p0 and p1 are the vetors of form [x,y,z]
        
        output: interpolated coordinates.
        
        https://en.wikipedia.org/wiki/Slerp
        
    '''
    assert abs(scalProd(p0,p0) - scalProd(p1,p1)) < 1e-7
    ang0Cos = scalProd(p0,p1)/scalProd(p0,p0)
    ang0Sin = sqrt(1 - ang0Cos*ang0Cos)
    ang0 = atan2(ang0Sin,ang0Cos)
    l0 = sin((1-t)*ang0)
    l1 = sin(t    *ang0)
    return np.array([(l0*p0[i] + l1*p1[i])/ang0Sin for i in range(len(p0))])


# In[41]:


# map 2D point p to spherical triangle s1,s2,s3 (3D vectors of equal length)
def mapGridpoint2Sogere(p,s1,s2,s3):
    '''
        program outputs the coordinate array of the projection of the input
        coordinates on the unit sphere.   
        inputs:
            - 'p' is the coordinate array of the planer verticies of the closed 
                shape to be projected in the form [x,y,z]'.
            - 's1','s2' and 's3' are the vectors defining plane of the co-ordinates 
                to be projected. 
        output: is the coordinate array of the projected face on the unit sphere.
        
        ex. mapGidpoint2Sogere([0,0.5,0.5],[1,0,0]',[0,1,0]',[0,0,1]')
    '''
    l1,l2,l3 = barycentricCoords(p)
    if abs(l3-1) < 1e-10: return s3
    l2s = l2/(l1+l2)
    p12 = slerp(s1,s2,l2s)
    return slerp(p12,s3,l3)


# In[42]:


def getProjectedPt(p,icoPoints,u,v,w):
    """
    p: 2D point location
    """
    return mapGridpoint2Sogere(p,
                                icoPoints[u,:],
                                icoPoints[v,:],
                                icoPoints[w,:])


# In[43]:


def getIcosaedreVertices():
    """
        outputs location of the icosaedre vertices 3D points
    """
    #golden ratio
    phi = 0.5*(1+sqrt(5)) 
    
    topPoints =         [(phi,1,0)]+        [(phi,-1,0)]+        [(1,0,-phi)]+        [(0,phi,-1)]+        [(0,phi,1)]+        [(1,0,phi)]
    
    topPoints = np.array(topPoints)
    # rot clockwise arround Z pour amener le point 1 en position (1,0,0)
    sinth = 1/sqrt(1+phi**2)
    costh = phi*sinth
    scale = 1/sqrt(1+phi**2)
    rot_mat = scale*np.array([[costh,sinth,0],
                        [-sinth, costh,0],
                        [0,0,1]])
    
    for i in range(len(topPoints)):
        topPoints[i,:] = np.matmul(rot_mat,topPoints[i,:])

    # change de repere
    # X' = -Y, Y'=-Z, Z'=X
    tmp = np.zeros_like(topPoints)
    for i in range(topPoints.shape[0]):
        tmp[i,0] = -topPoints[i,1]
        tmp[i,1] = -topPoints[i,2]
        tmp[i,2] =  topPoints[i,0]
    topPoints = tmp
    
    # points du bas de l'icosaedre
    bottomPoints = np.zeros_like(topPoints)
    for i in range(bottomPoints.shape[0]):
        bottomPoints[i,0] = -topPoints[i,0]
        bottomPoints[i,1] =  topPoints[i,1]
        bottomPoints[i,2] = -topPoints[i,2]

    # icosaedre vertices
    icoPoints=np.vstack((topPoints,bottomPoints))
    
    #return
    return icoPoints


# In[44]:


icoPoints = getIcosaedreVertices()


# In[45]:


icoPoints


# In[46]:


xoff7=-1/2
yoff7=0
n=4
scale = 1/(n*sqrt(3))
fact=0
tol=10**(-6)


# In[88]:


#i,j,th,cas
cases=np.array([
#    [0,1,-2*pi/3,3],
#    [1,n-1,2*pi/3,3],
#    [n-1,0,0.,3],
    [0,n-1,-2*pi/3,4],
    [n-1,1,2*pi/3,4],
    [1,0,0.,4]
])
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,0,1,n),(-2*pi/3,3))
#cases[1,0],cases[1,1],cases[1,2],cases[1,3] =  expand(xycent(xoff7,yoff7,1,n-1,n),(2*pi/3,3))
#cases[2,0],cases[2,1],cases[2,2],cases[2,3] =  expand(xycent(xoff7,yoff7,n-1,0,n),(0,3))
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,0,n-1,n),(-2*pi/3,4))
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,n-1,1,n),(2*pi/3,4))
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,1,0,n),(0,4))


# In[102]:


#pentaBuild=pd.DataFrame(columns=['idx','face','xyc','th','type'])

colors = ['black','red','blue','green','darkred','deepskyblue','forestgreen']
#idx0 = [0,1,11,16,10]
#idx0 = [1,2,12,17,11]
#idx0 = [2,3,13,18,12]
#idx0 = [3,4,14,19,13]
#idx0 = [4,0,10,15,14]
#idx0 = [6,7,15,10,16]
#idx0 = [5,6,16,11,17]
#idx0 = [9,5,17,12,18]
#idx0 = [8,9,18,13,19]
idx0 = [7,8,19,14,15]
infos=[]
for ik,k in enumerate(idx0):
    for iacas,acas in enumerate(cases):
        a = icoTriangs[k,0]
        b = icoTriangs[k,1]
        c = icoTriangs[k,2]
#        print(k,a,b,c)
        th = acas[2]
        xc,yc = xycent(xoff7,yoff7,acas[0],acas[1],n)
        if acas[3] == 3:
            pt2d = np.array([sqrt(3)/2-fact,1/2+fact/sqrt(3)]) # type 3 
        else:
            pt2d = np.array([-sqrt(3)/2+fact,1/2+fact/sqrt(3)]) # type 4 

        rot_mat = scale * np.array([[np.cos(th), -np.sin(th)],
                                    [np.sin(th), np.cos(th)]])
        pt2d = np.matmul(rot_mat,pt2d)
        pt2d[0] += xc
        pt2d[1] += yc
        infos.append([colors[ik],getProjectedPt(pt2d,icoPoints,a,b,c),iacas])

#########
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)    

for ik,i in enumerate(infos):
#    ax.scatter(i[0],i[1],i[2],marker='o',s=15,color="k")
    ax.text(i[1][0],i[1][1],i[1][2],'%s' % (str(ik)), 
                            size=15, zorder=1, color=str(i[0]))
    print(ik,i)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.show()


# In[ ]:





# In[ ]:




