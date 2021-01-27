#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

from math import sin,cos,acos,sqrt,pi, atan2

import pandas as pd

get_ipython().run_line_magic('matplotlib', '')


# In[2]:


# barycentric coords for triangle (-0.5,0),(0.5,0),(0,sqrt(3)/2)
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


# In[3]:


def scalProd(p1,p2):
    '''
        input: p1 and p2 are the vetors of form [x0,x1,...,xn]'
        output: is the scalar product of p1 and p2.
    '''
    return sum([p1[i]*p2[i] for i in range(len(p1))])


# In[4]:


def slerp(p0,p1,t):
    '''
        program outputs the spherical linear interpolation 
        of arc defined by p0, p1(around origin).  
        
        input: t=0 -> p0, t=1 -> p1. 
                p0 and p1 are the vetors of form [x,y,z]
        
        output: interpolated coordinates.
        
        https://en.wikipedia.org/wiki/Slerp
        
    '''
#    assert abs(scalProd(p0,p0) - scalProd(p1,p1)) < 1e-7
    ang0Cos = scalProd(p0,p1)/scalProd(p0,p0)
    ang0Sin = sqrt(1 - ang0Cos*ang0Cos)
    ang0 = atan2(ang0Sin,ang0Cos)
    l0 = sin((1-t)*ang0)
    l1 = sin(t    *ang0)
    return np.array([(l0*p0[i] + l1*p1[i])/ang0Sin for i in range(len(p0))])


# In[5]:


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


# In[6]:


def hexagon(x,y,th,scale,opt, fact=0):
    '''
        this program creates the hexagon of given configuration and size.
        inputs: 
            - x and y are the rectangular coordinates of the center of the hexagon
            - th is the rotation angle measured anticlockwise positive 
                from positive x axis
            - scale : is a contraction/dilation factor
            - opt: 1 full hexagon. 2 half hexagon

            output: planar hexagon (complete/truncated) orientated 
        example:
            hexagon(0,0,np.pi/6,0.5,1)
    '''
    # rotation matrx with scale (th>0 the transformation is anti-clockwise)
    rot_mat = scale * np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th), np.cos(th)]])
    if opt == 1:
        '''
            Hexagone complet
                                   Y
                  0                ^
            1           5          I
                                   I--- > X
            2           4
                  3     
        ''' 
        hex = np.zeros((2,6))
        hex[0,:]= np.array([np.sin(i*np.pi/3) for i in range(6)]) # X-coord
        hex[1,:]= np.array([np.cos(i*np.pi/3) for i in range(6)]) # Y-coord
    
    elif opt == 2:
        '''
            Hexagone tronque
            
                    2               ^
             3             1         I
                                     I
             4             0         I--- >
        ''' 
        hex = np.zeros((2,5))
        hex[0,:]= np.array([sqrt(3)/2,sqrt(3)/2,0,-sqrt(3)/2,-sqrt(3)/2]) # X-ccod
        hex[1,:]= np.array([0,0.5,1,0.5,0]) # Y-coord

    elif opt == 3:
        # point 0 et 1 sont modifiers par rapport au type 2
        hex = np.zeros((2,5))
        hex[0,:]= np.array([sqrt(3)/2-fact,sqrt(3)/2-fact,0,-sqrt(3)/2,-sqrt(3)/2]) # X-ccod
        hex[1,:]= np.array([0,1/2+fact/sqrt(3),1,0.5,0]) # Y-coord

    elif opt == 4:
        # point 3 et 4 sont modifiers par rapport au type 2
        hex = np.zeros((2,5))
        hex[0,:]= np.array([sqrt(3)/2,sqrt(3)/2,0,-sqrt(3)/2+fact,-sqrt(3)/2+fact]) # X-ccod
        hex[1,:]= np.array([0,0.5,1,1/2+fact/sqrt(3),0]) # Y-coord


    hex = np.matmul(rot_mat,hex)
 
    hex[0,:]= x+hex[0,:] 
    hex[1,:]= y+hex[1,:]
    
    return hex


# In[7]:


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
    rot_mat = scale*np.array([[costh,sinth,0.],
                        [-sinth, costh,0.],
                        [0.,0.,1]])
    
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


# In[8]:


def getProjectedFace(hexag,icoPoints,u,v,w):
    """
        outputs the coordinates of projected face on the plane 
        defined by tips of the vectors u,v and w on the unit radius sphere.  
 
         Inputs:

            _ 'hexag' is the coordinate array of the planer verticies of the closed 
             shape to be projected in the form [x,y,z]'.
            _ 'u','v' and 'w' are the vectors defining plane to be projected on 
                 the sphere.
            _ 'icoPoints': icosedre 3D vertices
    
    """
    
    n = hexag.shape[1]
    face = np.zeros((3,n))
    # projecting the input hexagonal mesh on the sphere
    for i in range(n):
        face[:,i] = mapGridpoint2Sogere(hexag[:,i],
                                        icoPoints[u,:],
                                        icoPoints[v,:],
                                        icoPoints[w,:])
    
    return face


# In[9]:


def getProjectedPt(p,icoPoints,u,v,w):
    """
    p: 2D point location
    """
    return mapGridpoint2Sogere(p,
                                icoPoints[u,:],
                                icoPoints[v,:],
                                icoPoints[w,:])


# In[10]:


def getIcoTriangs(modif=True):
    """
    20 faces : 3 vertex per face 
    12 vertex: each vertex is a number 0,...,11
    order : (0,1,2) for face 0 and then following some rotations
            we get the triplet for each face
    """
    nfaces = 20
    icoTriangs = np.zeros((nfaces,3),dtype=int)
    if modif:
        icoTriangs[0] = np.array([0,1,2])
        icoTriangs[1] = np.array([0,2,3])
        icoTriangs[2] = np.array([0,3,4])
        icoTriangs[3] = np.array([0,4,5])
        icoTriangs[4] = np.array([0,5,1])
        #JEC change 12/1/21 poour bottom cap evolue clockwise 
        icoTriangs[5] = np.array([7,8,6])
        icoTriangs[6] = np.array([11,7,6])
        icoTriangs[7] = np.array([10,11,6])
        icoTriangs[8] = np.array([9,10,6])
        icoTriangs[9] = np.array([8,9,6])
        #
        icoTriangs[10]= np.array([2,1,9])
        icoTriangs[11]= np.array([3,2,8])
        icoTriangs[12]= np.array([4,3,7])
        icoTriangs[13]= np.array([5,4,11])
        icoTriangs[14]= np.array([1,5,10])
        #
        icoTriangs[15]= np.array([9,1,10])
        icoTriangs[16]= np.array([8,2,9])
        icoTriangs[17]= np.array([7,3,8])
        icoTriangs[18]= np.array([11,4,7])
        icoTriangs[19]= np.array([10,5,11])
    else:
        icoTriangs = [(0,i+1,(i+1)%5+1) for i in range(5)] +             [(6,i+7,(i+1)%5+7) for i in range(5)] +             [(i+1,(i+1)%5+1,(7-i)%5+7) for i in range(5)] +             [(i+1,(7-i)%5+7,(8-i)%5+7) for i in range(5)]
        icoTriangs=np.array(icoTriangs)

    return icoTriangs


# In[11]:


def Draw_Goldberg_Polyhedron(n,fact=0,return_df=True, modif=False):
        
    nfaces=20
    
    #DF structure
    if return_df:
        df = pd.DataFrame(columns=['idx','type','center','vertices'])
    
    
    fig = plt.figure()
    fig.suptitle(f"n:{n}"+ ("-modif" if modif else "-orig"))

    ax = Axes3D(fig)
    ax.set_xlabel(r'$X$', fontsize=20)
    ax.set_ylabel(r'$Y$', fontsize=20)
    ax.set_zlabel(r'$Z$', fontsize=20)    
    colors = cm.rainbow(np.linspace(0, 1, nfaces))
    
    #icosedre vertices
    icoPoints = getIcosaedreVertices()
    
    #triplet of vertices for each face
    icoTriangs = getIcoTriangs(modif)
    
    # in equilateral trinagle 
    x,y = -1/2,0

    scale = 1/(n*sqrt(3))
    
    #loop over the 20 faces of the icosaedre
#    for k in [0,1,2,3,4]:
    for k in range(nfaces):
        for i in range(n+1):
            for j in range(n-i+1):
                if i==0:
                    th = -2*pi/3
                    if j==1:
                        opt = 3
                    elif j==n-i-1:
                        opt = 4
                    else:
                        opt = 2
                elif j==n-i: 
                    th = 2*pi/3
                    if i==1:
                        opt = 3
                    elif i==n-1:
                        opt = 4
                    else:
                        opt = 2
                elif (j==0 and i != 0):
                    th = 0
                    if i==1:
                        opt = 4
                    elif i==n-1:
                        opt = 3
                    else:
                        opt = 2
                else:
                    opt = 1
                    th = 0
                #exclude the hexagons at the vertices of the isocele triangle
                if (i!=0 or j!=0) and (i!=0 or j!=n) and (i!=n or j!=0):
                    hexagcenter = np.array([x+i*1/n+j*1/(2*n), y+j*sqrt(3)/(2*n)])
                    hexag = hexagon(hexagcenter[0],hexagcenter[1],th,scale,opt,fact=fact)
                    a = icoTriangs[k,0]
                    b = icoTriangs[k,1]
                    c = icoTriangs[k,2]
                    face = getProjectedFace(hexag,icoPoints,a,b,c)
                    center = getProjectedPt(hexagcenter,icoPoints,a,b,c)
                    xf,yf,zf = face[0,:],face[1,:],face[2,:]
                    vertsf=list(zip(xf,yf,zf))
                    # Dataframe
                    if return_df:
                        df=df.append({'idx':(k,i,j),
                                      'type':opt,
                                      'center':(center[0],center[1],center[2]),
                                      'vertices':vertsf},ignore_index=True)
                    
                    #plot
                    #ax.scatter(xf,yf,zf,s=1)
                    ax.scatter(center[0],center[1],center[2],marker='x',s=10,color=colors[k])
#                    ax.text(center[0]*1.01,center[1]*1.01,center[2]*1.01,'%s/%s' % (str(k),str(opt)), 
#                            size=10, zorder=1, color=colors[k],zdir=(0,0.5,0.5*np.sign(np.mean(zf)-0.5)))
#                    ax.text(center[0]*1.01,center[1]*1.01,center[2]*1.01,'%s/%s/%s' % (str(k),str(j),str(i)), 
#                            size=10, zorder=1, color=colors[k],zdir=(0,0.5,0.5*np.sign(np.mean(zf)-0.5)))

                    ax.add_collection3d(Poly3DCollection([vertsf], facecolors = colors[k], edgecolors='k', linewidths=1, alpha=0.9))

    #eofor
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.show()
    # return dataframe
    if return_df:
        return df


# In[12]:


#avec rearrangement des vertex des faces
Draw_Goldberg_Polyhedron(5,return_df=False, modif=True)


# In[13]:


def plot_icoVertices():
    icoPoints = getIcosaedreVertices()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(icoPoints[:,0],icoPoints[:,1],icoPoints[:,2], c='black')
    ax.set_xlabel(r'$X$', fontsize=20)
    ax.set_ylabel(r'$Y$', fontsize=20)
    ax.set_zlabel(r'$Z$', fontsize=20)
    ax.set_xlim3d([-1,1])
    ax.set_ylim3d([-1,1])
    ax.set_zlim3d([-1,1])

    plt.show()


# In[14]:


plot_icoVertices()


# In[15]:


# Import seaborn
import seaborn as sns


# In[16]:


def icoTriangsConfig(modif=False):
    """
    Draw an icosaedron with faces and vertices numering
    """
    
    #icosedre vertices
    icoPoints = getIcosaedreVertices()
        
    x,y = -1/2,0
    # ensemble de tous les triplets de points qui constituent les 20 triangles equilateraux
    icoTriangs = getIcoTriangs(modif)
    
    fig = plt.figure()
    fig.suptitle("icoTriangs" + ("-modif" if modif else "-orig"))

    ax = Axes3D(fig)

    nfaces=20
    sns.reset_orig()  # get default matplotlib styles back
    clrs = sns.color_palette('hls', n_colors=nfaces)  # a list of RGB tuples
#    ax.scatter(icoPoints[:,0],icoPoints[:,1],icoPoints[:,2], c='blue')
    for k in range(icoPoints.shape[0]):
        ax.text(icoPoints[k,0]*1.05,icoPoints[k,1]*1.05,icoPoints[k,2]*1.05,'%s' % str(k),size=15, zorder=10, color='blue')
    for k in range(nfaces):
        xf,yf,zf=icoPoints[icoTriangs[k],0],icoPoints[icoTriangs[k],1],icoPoints[icoTriangs[k],2]
        vertsf=list(zip(xf,yf,zf))
        ax.add_collection3d(Poly3DCollection([vertsf], facecolors = clrs[k], 
                                             edgecolors='k', linewidths=1, alpha=0.2))
        cx,cy,cz = np.mean(xf),np.mean(yf),np.mean(zf)
        ax.text(cx,cy,cz,'%s' % str(k),size=15, zorder=15, color='black')

        
    ax.set_xlabel(r'$X$', fontsize=20)
    ax.set_ylabel(r'$Y$', fontsize=20)
    ax.set_zlabel(r'$Z$', fontsize=20)
    ax.set_xlim3d([-1,1])
    ax.set_ylim3d([-1,1])
    ax.set_zlim3d([-1,1])

    plt.show()


# In[17]:


#Version de base
#icoTriangsConfig()


# In[18]:


#mofdified versipn
icoTriangsConfig(modif=True)


# In[19]:


def rot(phi, a,b,c):
    """ 
        rotation of angle phi (radians) arround axis (a,b,c) (unit norm)
        a=0.21
        b=0.43
        c=sqrt(1-a*a-b*b)
        phi=0.33
        r1=rot(phi,a,b,c)
        
        assert np.allclose(np.matmul(r1,r1),rot(2*phi,a,b,c)),\ 
                "bug: r1(phi).r1(phi) should be r1(2*phi)"
        assert np.allclose(np.matmul(r1,r1.T),rot(0.,1,0,0)),\
                "bug: r1.(r1.T) should be Identity"
        
    """
    cphi = cos(phi)
    sphi = sin(phi)
    return np.array([[cphi + a*a*(1-cphi),(1-cphi)*a*b - c*sphi, (1-cphi)*a*c + b*sphi],
             [(1-cphi)*a*b + c*sphi, cphi + b*b*(1-cphi), (1-cphi)*b*c - a*sphi],
             [(1-cphi)*a*c - b*sphi, (1-cphi)*b*c + a*sphi, cphi + c*c*(1-cphi)]])


# In[20]:


def getFace0TofaceIMtx():
    nfaces = 20
    icoFaceMtx = np.zeros((nfaces,3,3))
    
    # cosine directors of axis "0" starting at origin and passing by vertex 0
    a0 = 0.
    b0 = 0.
    c0 = 1.
    # cosine directors of axis "1" starting at origin and passing by vertex 1
    goldphi = (1.+sqrt(5.))/2.  # golden ratio (x^2=x+1)
    goldphi2 = goldphi * goldphi 
    c1 = goldphi/(1.+goldphi2)
    a1 = 2.*c1
    b1 = 0.
    assert np.isclose(a1*a1+b1*b1+c1*c1,1.0), "bug getFace0TofaceI axis 1 should be norm=1"
    # cosine directors of axis "4" starting at origin and passing by vertex 4
    a4 = - goldphi2/(1.+goldphi2)
    c4 = goldphi/(1.+goldphi2)
    b4 = - c4*sqrt(4.-goldphi2)
    assert np.isclose(a4*a4+b4*b4+c4*c4,1.0), "bug getFace0TofaceI axis 4 should be norm=1"

    theta5 = 2.*pi/5.
    
    #Face 0 is the reference 
    icoFaceMtx[0] = rot(0.,1.,0.,0.)    # identity
    
    # Faces of the upper cap : (vertex 0 is in common)
    # -----------------------
    # face "i" is obtained from face 0 by rotation +i*(2pi/5)
    # arround axis "0"    
    for i in [1,2,3,4]:
        icoFaceMtx[i] = rot(i*theta5, a0,b0,c0)
    
    # Faces of the middle: 
    # -----------------------
    # two types: 
    #    - type 1 obtained by rotation arround axe 0 of face 10
    #    - type 2 obtained by rotation arround axe 0 of face 15
    
    # Type 1
    # Face 10 is obtained face 0 by rotation -2pi/5 around axis 1
    icoFaceMtx[10] = rot(-theta5, a1,b1,c1)
    # Faces [11,12,13,14] obtained from face 10 by i=1,2,3,4 rotations of 2pi/5 
    # arround axe 0
    for i in [1,2,3,4]:
        rtmp = rot(i*theta5, a0,b0,c0)
        icoFaceMtx[10+i] = np.matmul(rtmp,icoFaceMtx[10]) 

    # Type 2
    # Face 15 is obtained face 0 by rotation -4pi/5 around axis 1
    icoFaceMtx[15] = rot(-2*theta5, a1,b1,c1)
    # Faces [16,17,18,19] obtained from face 15 by i=1,2,3,4 rotations of 2pi/5 
    # arround axe 0
    for i in [1,2,3,4]:
        rtmp = rot(i*theta5, a0,b0,c0)
        icoFaceMtx[15+i] = np.matmul(rtmp,icoFaceMtx[15]) 
    
    
    # Faces of the bottom cap (vertex 6 is in common)
    # -----------------------
    # first Face 5 is obtained from face 0 by rotation of 4pi/5 arround axis "4" 
    icoFaceMtx[5] = rot(2*theta5, a4,b4,c4)
    # then: faces [6,7,8,9] obtained from face 5 by i=1,2,3,4 rotations of +2pi/5 
    # arround axe 0
    for i in [1,2,3,4]:
        rtmp = rot(i*theta5, a0,b0,c0)
        icoFaceMtx[5+i] = np.matmul(rtmp,icoFaceMtx[5]) 

    
    #done
    return icoFaceMtx


# In[21]:


getFace0TofaceIMtx()[1]


# In[22]:


def getFaceIToface0Mtx():
    """
    This is the inverse of getFace0TofaceIMtx but we do not use any inversion mtx routine
    """
    
    nfaces = 20
    icoFaceMtx = np.zeros((nfaces,3,3))
    
    # cosine directors of axis "0" starting at origin and passing by vertex 0
    a0 = 0.
    b0 = 0.
    c0 = 1.
    # cosine directors of axis "1" starting at origin and passing by vertex 1
    goldphi = (1.+sqrt(5.))/2.  # golden ratio (x^2=x+1)
    goldphi2 = goldphi * goldphi 
    c1 = goldphi/(1.+goldphi2)
    a1 = 2.*c1
    b1 = 0.
    assert np.isclose(a1*a1+b1*b1+c1*c1,1.0), "bug getFace0TofaceI axis 1 should be norm=1"
    # cosine directors of axis "4" starting at origin and passing by vertex 4
    a4 = - goldphi2/(1.+goldphi2)
    c4 = goldphi/(1.+goldphi2)
    b4 = - c4*sqrt(4.-goldphi2)
    assert np.isclose(a4*a4+b4*b4+c4*c4,1.0), "bug getFace0TofaceI axis 4 should be norm=1"

    theta5 = 2.*pi/5.
    
    #Face 0 is the reference 
    icoFaceMtx[0] = rot(0.,1.,0.,0.)    # identity
    
    # Faces of the upper cap : (vertex 0 is in common)
    # -----------------------
    # face "i" is obtained from face 0 by rotation +i*(2pi/5)
    # arround axis "0"    
    for i in [1,2,3,4]:
        icoFaceMtx[i] = rot(-i*theta5, a0,b0,c0)
    
    # Faces of the middle: 
    # -----------------------
    # two types: 
    #    - type 1 obtained by rotation arround axe 0 of face 10
    #    - type 2 obtained by rotation arround axe 0 of face 15
    
    # Type 1
    # Face 10 is obtained face 0 by rotation -2pi/5 around axis 1
    icoFaceMtx[10] = rot(+theta5, a1,b1,c1)
    # Faces [11,12,13,14] obtained from face 10 by i=1,2,3,4 rotations of 2pi/5 
    # arround axe 0
    for i in [1,2,3,4]:
        rtmp = rot(-i*theta5, a0,b0,c0)
        icoFaceMtx[10+i] = np.matmul(icoFaceMtx[10],rtmp) 

    # Type 2
    # Face 15 is obtained face 0 by rotation -4pi/5 around axis 1
    icoFaceMtx[15] = rot(+2*theta5, a1,b1,c1)
    # Faces [16,17,18,19] obtained from face 15 by i=1,2,3,4 rotations of 2pi/5 
    # arround axe 0
    for i in [1,2,3,4]:
        rtmp = rot(-i*theta5, a0,b0,c0)
        icoFaceMtx[15+i] = np.matmul(icoFaceMtx[15],rtmp) 
    
    
    # Faces of the bottom cap (vertex 6 is in common)
    # -----------------------
    # first Face 5 is obtained from face 0 by rotation of 4pi/5 arround axis "4" 
    icoFaceMtx[5] = rot(-2*theta5, a4,b4,c4)
    # then: faces [6,7,8,9] obtained from face 5 by i=1,2,3,4 rotations of +2pi/5 
    # arround axe 0
    for i in [1,2,3,4]:
        rtmp = rot(-i*theta5, a0,b0,c0)
        icoFaceMtx[5+i] = np.matmul(icoFaceMtx[5],rtmp) 

    
    #done
    return icoFaceMtx


# In[ ]:





# In[23]:


"""
Verif getFace0TofaceI:
Start from face 0 vertices (0,1,2) and 
applying getFace0TofaceI[i] one should find vertices of face "i"
"""
nfaces=20
modif = True
#icosedre vertices
icoPoints = getIcosaedreVertices()
#triplet of vertices for each face (new schema)
icoTriangs = getIcoTriangs(modif)
# Rotation matrices to throw a point in Face 0 to a point in Face "i" (i:1,..,19)
faceMtx = getFace0TofaceIMtx()
# Throw face 0 vertices (0,1,2) by broadcasting to face i vertices and compare 
verticesI=np.einsum('ijk,kl->ijl', faceMtx,icoPoints[0:3].T)
for i in range(nfaces):
    print(f"face {i}: " + ("Ok" if np.allclose(verticesI[i].T, icoPoints[icoTriangs[i]]) else "Nok"))


# In[24]:


mtx02I = getFace0TofaceIMtx()
mtxI20 = getFaceIToface0Mtx()


# In[25]:


"""
Verif getFaceIToface0Mtx
"""
identity = np.identity(3)
for k in range(20):
    print(f"verif {k}:" + str(np.allclose(np.matmul(mtx02I[k],mtxI20[k]),np.identity(3))))


# In[26]:


def buildFace0(n=5):
    '''
        build face 0
         n : order of the pixelization
    '''
    # localisation bottom left vertices
    x,y = -1/2,0
    scale = 1/(n*sqrt(3))


    #factor enhancement of pentagon
    fact=0

##    df = pd.DataFrame(columns=['idx','type','center','vertices'])

    #icosedre vertices
    icoPoints = getIcosaedreVertices()
    #triplet of vertices for each face (new schema)
    #modif face vertices
    icoTriangs = getIcoTriangs(modif=True)

    faces   = []
    centers = []
    types   = []
    indexes = []

    for i in range(n+1):
        for j in range(n-i+1):
            if i==0:
                th = -2*pi/3
                if j==1:
                    opt = 3
                elif j==n-i-1:
                    opt = 4
                else:
                    opt = 2
            elif j==n-i: 
                th = 2*pi/3
                if i==1:
                    opt = 3
                elif i==n-1:
                    opt = 4
                else:
                    opt = 2
            elif (j==0 and i != 0):
                th = 0
                if i==1:
                    opt = 4
                elif i==n-1:
                    opt = 3
                else:
                    opt = 2
            else:
                opt = 1
                th = 0
            #exclude the hexagons at the vertices of the isocele triangle
            if (i!=0 or j!=0) and (i!=0 or j!=n) and (i!=n or j!=0):
                hexagcenter = np.array([x+i*1/n+j*1/(2*n), y+j*sqrt(3)/(2*n)])
                hexag = hexagon(hexagcenter[0],hexagcenter[1],th,scale,opt,fact=fact)
                a = icoTriangs[0,0]
                b = icoTriangs[0,1]
                c = icoTriangs[0,2]
                face   = getProjectedFace(hexag,icoPoints,a,b,c)
                center = getProjectedPt(hexagcenter,icoPoints,a,b,c)

                faces.append(face)
                centers.append(center)
                indexes.append((0,i,j))
                types.append(opt)
            
    return faces, centers, indexes, types
#######################################
#            xf,yf,zf = face[0,:],face[1,:],face[2,:]
#            vertsf = np.array(list(zip(xf,yf,zf)))
#            print("pts: ",vertsf)
#            print("...  ",face)
#           print("...  ",face.shape)
#            print("...  ",(icoPoints[0:3].T).shape)
#            verticesI=np.einsum('jk,kl->jl', faceMtx[1],face)
#            print("...  ",verticesI.shape)
#            #plot
#            ax.scatter(center[0],center[1],center[2],marker='x',s=10,color=colors[k])
#            ax.add_collection3d(Poly3DCollection([vertsf], 
#                                                facecolors = colors[k], edgecolors='k', linewidths=1, alpha=0.9)


# In[27]:


def plotFaceI(k=0, ax=None):
    '''
        k : index of Icosahedre face
    '''
    
    #Get tiles of Face 0
    faces0, centers0, indexes0, types0 = buildFace0()
    collectFaces0 = np.hstack(faces0)  #collect all tiles of face 0
    centers0 = np.array(centers0)
    
    
    #Get Face 0 -> Face k rotation matrix
    faceMtx = getFace0TofaceIMtx()
    
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel(r'$X$', fontsize=20)
        ax.set_ylabel(r'$Y$', fontsize=20)
        ax.set_zlabel(r'$Z$', fontsize=20)  
        
    nIcofaces = 20
    colors = cm.rainbow(np.linspace(0, 1, nIcofaces))

    #number of tiles per face wrt the order n
    nTiles = len(faces0) # (n+1)(n+2)/2 - 3

    centerfAll = np.einsum('jk,kl->jl', faceMtx[k],centers0.T).T
    
    #transform tuple to list
    idxfAll  = np.array(indexes0)
    typesAll = np.array(types0)
    
    
    # change first element of each index by the face identifier
    idxfAll[:,0] = k
    idxfAll = [tuple(x) for x in idxfAll.tolist()]

    for i in range(nTiles):
        vertices=np.einsum('jk,kl->jl', faceMtx[k],faces0[i])
        vertsf  = vertices.T
        centerf = centerfAll[i]
        idxf  = idxfAll[i]
        typef = typesAll[i]
    #    print("vertsf:  ", vertsf.shape)

        xf,yf,zf = vertsf[:,0],vertsf[:,1],vertsf[:,2]


        ax.scatter(centerf[0],centerf[1],centerf[2],marker='x',s=10,color='k')
        ax.add_collection3d(Poly3DCollection([list(zip(xf,yf,zf))], 
                                             facecolors = colors[k], 
                                             edgecolors='k', 
                                             linewidths=1, alpha=0.5))
        ax.text(centerf[0]*1.01,centerf[1]*1.01,centerf[2]*1.01,"{}".format('/'.join([str(x) for x in idxf])),size=10, zorder=1, color='k')

    if ax is None:
        ax.set_xlabel(r'$X$', fontsize=20)
        ax.set_ylabel(r'$Y$', fontsize=20)
        ax.set_zlabel(r'$Z$', fontsize=20)
        ax.set_xlim3d([-1,1])
        ax.set_ylim3d([-1,1])
        ax.set_zlim3d([-1,1])
        plt.show()


# In[28]:


#faces0, centers0, indexes0, types0 = buildFace0()


# In[29]:


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)  
for k in [0,1,4,10]:
    plotFaceI(k, ax=ax)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)
ax.set_xlim3d([-1,1])
ax.set_ylim3d([-1,1])
ax.set_zlim3d([-1,1])
plt.show()


# In[30]:


def pt3D2FaceId(pt,icoTriangCenters, zTh1, zTh2, full=True):
    # study combinatoric
    # 1) full: test 20 distance from current pt and the 20 face centers
    # 2) select faces based on up-cup/bottom-cup/the rest.
    if full:
        return np.argmax(np.einsum('jk,kl->jl', icoTriangCenters,pt[:,np.newaxis]))


# # angle2Face vectorized based on relative distance wrt face centers

# In[ ]:





# In[31]:


def getIcoTriangCenters():
    #icosedre vertices
    icoPoints = getIcosaedreVertices()
    #triplet of vertices for each face (new schema)
    #modif face vertices
    icoTriangs = getIcoTriangs(modif=True)
    #
    #Get the location of the barycenters of the icosaedron faces
    nfaces=20
    tmp = np.array([icoPoints[icoTriangs[k]] for k in range(nfaces)])
    icoTriangCenters = np.mean(tmp,axis=1,dtype=np.float64)
    # project on the unit-sphere
    norm=np.sqrt((icoTriangCenters*icoTriangCenters).sum(axis=1))

    return icoTriangCenters / norm[:,np.newaxis]


# In[ ]:





# In[32]:


def pt3DtoSpherical(xyz):
    norm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    theta = np.arccos(xyz[:,2]/norm)
    phi = np.arctan2(xyz[:,1], xyz[:,0])
    mask = phi<0
    phi[mask]+=2*pi
    return np.array([theta,phi]).T


# In[861]:





# In[33]:


# theta, phi angles of the 20 center of faces
icoTriangCenters = getIcoTriangCenters()
angleicoTriangCenters= pt3DtoSpherical(icoTriangCenters)


# In[34]:


# theta, phi angles the 12 vertices
icoPoints = getIcosaedreVertices()
icoVertices = icoPoints
angleicoVertices = pt3DtoSpherical(icoVertices)


# In[35]:


icoVertices


# In[36]:


angleicoVertices


# In[37]:


icoTriangCenters.shape


# In[38]:


angleicoTriangCenters.shape


# In[39]:


def angle2FaceId_fullVectorized(thetaPhi, icoTriangCenters):
    """ find the face of the different points 
        thetaPhi = Nx2  :  N x (theta, phi)
        icoTriangCenters = 20x3 :  20x(xc,yc,zc)
        use: scalaire product 
          pt = 3xN  : (x,y,z) x N
          (20,3) . (3,N) = (20,N)  
          argmax axis 0 => N values
    """
    #current pt
#    print(thetaPhi.shape)
    theta = thetaPhi[:,0]
    phi   = thetaPhi[:,1]
    pt = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    # test the 20 distnaces from current pt to the 20 face centers
#    print(icoTriangCenters.shape,pt.shape)
    return np.argmax(np.einsum('jk,kl->jl',icoTriangCenters,pt),axis=0)


# In[40]:


"""
 Verif1 : angle2FaceId_fullVectorized
 The icosahedres faces must be on their faces!
 N: the first N icoahedres faces to xcheck the einsum & argmax axis
""" 
N = 15
truth = np.arange(N)
assert np.allclose(angle2FaceId_fullVectorized(angleicoTriangCenters[:N,:], icoTriangCenters),
                   truth), "Bug: The icosahedres faces must be on their faces!"


# In[41]:


"""
Verif2 : angle2FaceId_fullVectorized 
draw random points on the sphere and plot location and face number
"""
Npts = 10  # ATTENTION Nptsx Npts points....
theta,phi = np.mgrid[0:pi:Npts*1j, 0:2*pi:Npts*1j]



someOnSpherePts = np.array([theta.reshape(-1,),phi.reshape(-1,)]).T

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

pts = np.array([x.reshape(-1,),y.reshape(-1,),z.reshape(-1,)]).T

faces = angle2FaceId_fullVectorized(someOnSpherePts, icoTriangCenters)



nIcofaces = 20
colors = cm.rainbow(np.linspace(0, 1, nIcofaces))

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)  

for k in range(Npts**2):
    ax.scatter(pts[k,0],pts[k,1],pts[k,2],marker='o',s=1,color=colors[faces[k]])



ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)
ax.set_xlim3d([-1.1,1.1])
ax.set_ylim3d([-1.1,1.1])
ax.set_zlim3d([-1.1,1.1])
plt.show()


# In[43]:


Npts = 100  # ATTENTION Nptsx Npts points....
theta,phi = np.mgrid[0:pi:Npts*1j, 0:2*pi:Npts*1j]
someOnSpherePts = np.array([theta.reshape(-1,),phi.reshape(-1,)]).T
print("Nbre de pts: ",someOnSpherePts.shape[0])


# In[78]:


get_ipython().run_cell_magic('timeit', '', '"""\nTiming angle2FaceId_fullVectorized\n"""\nangle2FaceId_fullVectorized(someOnSpherePts, icoTriangCenters)')


# # Vers le angle2pix

# # angle2Pix for a Point on a Face

# In[54]:


getIcosaedreVertices()


# In[55]:


icoTriangs = getIcoTriangs(modif=True)


# In[56]:


icoTriangs[0]


# In[57]:


#triplet of vertices of Face 0
vertices0 = getIcosaedreVertices()[icoTriangs[0]]


# In[58]:


vertices0


# In[59]:


#triplet of vertices of Face 1
vertices1 = getIcosaedreVertices()[icoTriangs[1]]


# In[60]:


vertices1


# In[61]:


angle2FaceId_fullVectorized(angleicoTriangCenters, icoTriangCenters)


# In[62]:


angleicoTriangCenters


# In[94]:


# 1 point dans la face 1
# si on prend le barycentre entre les sommets de 
a=0.1
b=0.7
pt1 = a * vertices1[0] + b * vertices1[1] + (1-a-b) * vertices1[2]
pt1 = pt1[np.newaxis,:]


# In[64]:


pt1.shape


# In[65]:


#matrice de Face I a Face 0
mtxI20 = getFaceIToface0Mtx()
mtx120 = mtxI20[1]


# In[82]:


#rotate to get the corresponding point on face0
pt0 = np.matmul(mtx120,pt1.T).T


# In[80]:


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)  

ax.scatter(vertices1[:,0],vertices1[:,1],vertices1[:,2],marker='o',s=1,color='k')



ax.scatter(pt1[:,0],pt1[:,1],pt1[:,2],marker='o',s=10,color='r')

ax.scatter(pt0[:,0],pt0[:,1],pt0[:,2],marker='o',s=10,color='blue')


ax.add_collection3d(Poly3DCollection([vertices1], facecolors = 'white', edgecolors='k', linewidths=1, alpha=0.5))
ax.add_collection3d(Poly3DCollection([vertices0], facecolors = 'lightblue', edgecolors='k', linewidths=1, alpha=0.5))


ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)
ax.set_xlim3d([-1.1,1.1])
ax.set_ylim3d([-1.1,1.1])
ax.set_zlim3d([-1.1,1.1])
plt.show()


# In[81]:


def getBarycentricCoord(pt,face):
    """
        pt (x,y,z) 
        face : index
        pt should be already ON the face
    """
    icoTriangs = getIcoTriangs(modif=True)
    #triplet of vertices of Face
    vertices = getIcosaedreVertices()[icoTriangs[face]]
    #
    vec0 = vertices[0][np.newaxis,:]-pt
    vec1 = vertices[1][np.newaxis,:]-pt
    vec2 = vertices[2][np.newaxis,:]-pt
    # 2 fois l'aire d'un des 20 triangles equilateraux 
    aire0 = sqrt(3)/2/sin(2*pi/5)**2
    #retreive barycentric coordinate
    a0 = np.cross(vec1,vec2)
    a0 = np.sqrt((a0*a0).sum(axis=1))/aire0
    a1 = np.cross(vec2,vec0)
    a1 = np.sqrt((a1*a1).sum(axis=1))/aire0
    return a0,a1


# # find barycentric coordinate of a 3D pt on the sphere projected onto the corresponding icosahedron triangle

# In[284]:


def testBaryCoordExt(a=0.1,b=0.7):
    """
    Test of getBarycentricCoordExtension method
    input: (a,b) barycentric coordinates of a pt on each icosahedron triangle
    process:
       from (a,b) find the pt on the trinagle surface
       project this point on to the sphere
       then call getBarycentricCoordExtension which perform the inverse 
         - retreive the pt on the triangle surface
         - retreive the barycentric coordinates
    """
    nFaces = 20
    assert a>=0 and a<=1 and b>=0 and b<=1 and a+b <=1, "a in [0,1], b in [0,1] and a+b in [0,1]"
    icoTriangs = getIcoTriangs(modif=True)
    for i in range(nFaces):
        #triangle i vertices
        vertices = getIcosaedreVertices()[icoTriangs[i]]
        #The pt at the surface  triangle as barycenter of the vertices
        ptOnFace = a * vertices[0] + b * vertices[1] + (1-a-b) * vertices[2]
        #put it on the sphere
        ptOnsphere = ptOnFace/np.sqrt(np.sum(ptOnFace*ptOnFace))
        #retreive the (a,b) coordinate
        an,bn = getBarycentricCoordExtension(ptOnsphere,icoTriangs,vertices,i)
        if np.isclose(an,a) and np.isclose(bn,b):
            print(f"test on face {i} Ok")
        else:
            print(f"test on face {i} NOK: {a} != {an} or {b} != {bn}")


# In[286]:


testBaryCoordExt()


# # Find hexagone index once the barycentric coordinates are found 

# In[302]:


#
#    x,y = -1/2,0
#    scale = 1/(n*sqrt(3)) 
#hexagcenter = np.array([x+i*1/n+j*1/(2*n), y+j*sqrt(3)/(2*n)])
# bottom left corner
xlc,ylc = -1/2,0
norder = 5
# (a,b) bary of pt
a=0.5
b=0.5
# (x,y) of the pt
xp = 0.5*(1+a)-b
yp = a*sqrt(3)/2
#
####jhexa = np.floor((2*n)/sqrt(3)*(yp-ylc))
####ihexa = np.floor(n*((xp-xlc)-(yp-yc)/sqrt(3)))
jhexa = np.floor(norder*a).astype(np.int64) 
ihexa = np.floor(norder*(1-b)).astype(np.int64) 


# In[303]:


ihexa,jhexa


# In[310]:


#tuiles de la face 0
faces0, centers0, indexes0, types0 = buildFace0(n=norder)


# In[311]:


indexes0


# In[309]:





# In[ ]:




