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


pd.__version__


# In[3]:


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


# In[4]:


def scalProd(p1,p2):
    '''
        input: p1 and p2 are the vetors of form [x0,x1,...,xn]'
        output: is the scalar product of p1 and p2.
    '''
    return sum([p1[i]*p2[i] for i in range(len(p1))])


# In[5]:


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


# In[6]:


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


# In[7]:


mapGridpoint2Sogere([0,0.5,0.5],[1,0,0],[0,1,0],[0,0,1])


# In[8]:


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


# In[9]:


hexagon(0,0,np.pi/6,0.5,1)


# In[10]:


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


# In[65]:


icoPoints = getIcosaedreVertices()


# In[11]:


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


# In[12]:


def getProjectedPt(p,icoPoints,u,v,w):
    """
    p: 2D point location
    """
    return mapGridpoint2Sogere(p,
                                icoPoints[u,:],
                                icoPoints[v,:],
                                icoPoints[w,:])


# In[13]:


def plot_icoVertices():
    s,c = 2/sqrt(5),1/sqrt(5)
    topPoints = [(0,0,1)] + [(s*cos(i*2*pi/5.), s*sin(i*2*pi/5.), c) for i in range(5)]
    bottomPoints = [(-x,y,-z) for (x,y,z) in topPoints]
    icoPoints = topPoints + bottomPoints
    icoPoints = np.array(icoPoints)
    
    print('plot_icoVertices:\n',icoPoints)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    topPoints=np.array(topPoints)
    bottomPoints=np.array(bottomPoints)
    ax.scatter(topPoints[:,0],topPoints[:,1],topPoints[:,2], c='blue')
    ax.scatter(bottomPoints[:,0],bottomPoints[:,1],bottomPoints[:,2], c='red')
    ax.set_xlabel(r'$X$', fontsize=20)
    ax.set_ylabel(r'$Y$', fontsize=20)
    ax.set_zlabel(r'$Z$', fontsize=20)
    plt.show()


# <img src="./icosaedre_vertices.png">

# In[14]:


def plot_ico3():
    phi = 0.5*(1+sqrt(5)) 

    topPoints =         [(phi,1,0)]+        [(phi,-1,0)]+        [(1,0,-phi)]+        [(0,phi,-1)]+        [(0,phi,1)]+        [(1,0,phi)]
    
    topPoints = np.array(topPoints)
    # rot clockwise arround Z to put point 1 in (1,0,0) position in the X,Y,Z frame
    sinth = 1/sqrt(1+phi**2)
    costh = phi*sinth
    scale = 1/sqrt(1+phi**2)
    rot_mat = scale*np.array([[costh,sinth,0],
                        [-sinth, costh,0],
                        [0,0,1]])
    
    for i in range(len(topPoints)):
        topPoints[i,:] = np.matmul(rot_mat,topPoints[i,:])

    # switch to frame (X',Y',Z')
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

    icoPoints=np.vstack((topPoints,bottomPoints))
            
    print('ico3:\n',icoPoints)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(topPoints[:,0],topPoints[:,1],topPoints[:,2],c='orange')
    ax.scatter(bottomPoints[:,0],bottomPoints[:,1],bottomPoints[:,2],c='cyan')
    ax.set_xlabel(r'$X$', fontsize=20)
    ax.set_ylabel(r'$Y$', fontsize=20)
    ax.set_zlabel(r'$Z$', fontsize=20)
    plt.show()


# In[15]:


def Draw_Goldberg_Polyhedron(n,fact=0,return_df=True):
        
    nfaces=20
    
    #DF structure
    if return_df:
        df = pd.DataFrame(columns=['idx','type','center','vertices'])
    
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel(r'$X$', fontsize=20)
    ax.set_ylabel(r'$Y$', fontsize=20)
    ax.set_zlabel(r'$Z$', fontsize=20)    
    colors = cm.rainbow(np.linspace(0, 1, nfaces))
    
    #icosedre vertices
    icoPoints = getIcosaedreVertices()
    
    
    x,y = -1/2,0
    # ensemble de tous les triplets de points qui constituent les 20 triangles equilateraux
    icoTriangs = [(0,i+1,(i+1)%5+1) for i in range(5)] +             [(6,i+7,(i+1)%5+7) for i in range(5)] +             [(i+1,(i+1)%5+1,(7-i)%5+7) for i in range(5)] +             [(i+1,(7-i)%5+7,(8-i)%5+7) for i in range(5)]
    icoTriangs=np.array(icoTriangs)

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


# In[85]:


icoTriangs = [(0,i+1,(i+1)%5+1) for i in range(5)] +             [(6,i+7,(i+1)%5+7) for i in range(5)] +             [(i+1,(i+1)%5+1,(7-i)%5+7) for i in range(5)] +             [(i+1,(7-i)%5+7,(8-i)%5+7) for i in range(5)]
icoTriangs=np.array(icoTriangs)
icoTriangs


# In[18]:


n=4
df0= Draw_Goldberg_Polyhedron(n,fact=0.0,return_df=True)


# In[17]:


len(df0)


# In[1737]:


#fact=0.1 doit etre Ok, 0.3 c'est pour debugger
Draw_Goldberg_Polyhedron(n,fact=0.3)


# In[19]:


def stat(n):
    typeOfHex=[]
    for i in range(n+1):
        for j in range(n-i+1):
            if i==0:
                opt = 2
                th = -2*pi/3
            elif j==n-i: 
                opt = 2
                th = 2*pi/3
            elif (j==0 and i != 0):
                opt = 2
                th = 0
            else:
                opt = 1
                th = 0

            if (i!=0 or j!=0) and (i!=0 or j!=n) and (i!=n or j!=0):
                typeOfHex.append(opt)
    _,nbre=np.unique(typeOfHex,return_counts=True)
    return nbre


# In[20]:


def func2(x):
    a=[i for i in x if x.count(i)==1]
    return a


# In[21]:


df0


# In[22]:


tol=10**(-6)
def myround(x,tol=10**(-6)):
    return tuple(np.round(np.array(x)/tol).astype(int))


# In[23]:


df0['center']=df0['center'].map(myround)
df0['vertices']=df0['vertices'].map(myround)


# In[24]:


df0


# In[ ]:





# In[25]:


g0=df0.groupby('center', as_index=False).agg({'idx':'sum','type':lambda x: list(x),'vertices':'sum'})


# In[26]:


g0


# In[27]:


g0['vertices']=g0['vertices'].map(lambda x: [list(y) for y in x])


# In[28]:


g0['vertices']=g0['vertices'].map(lambda x: tuple([tuple(y) for y in x]))


# In[29]:


g0['vertices']=g0['vertices'].map(func2)


# In[30]:


g0


# In[31]:


#En principe il doit n'y avoir que des hexagones => tmp=6
g0['Nvertices'] = g0['vertices'].map(lambda x:len(x))


# In[32]:


np.unique(g0['Nvertices'])


# In[33]:


#Des hexagones sont la fusion de 2 polygones appartenant à 2 faces
g0['nfaces'] = g0['idx'].map(lambda x:len(x)//3)


# In[34]:


np.unique(g0['nfaces'])


# In[35]:


g0['nfaces2'] = g0['type'].map(lambda x:len(x))


# In[36]:


np.unique(g0['nfaces']==g0['nfaces2'])


# In[37]:


np.unique(g0['nfaces'],return_counts=True)


# In[38]:


g0=g0.drop(columns=['nfaces2'])


# In[39]:


g0


# Il faut repérer les hexagones resulats de fusions de moities et pour lesquels la liste des points n'est pas connexe 

# In[40]:


def cmpdist(x):
    pt2,pt3,pt5=np.array(x[2]),np.array(x[3]),np.array(x[5])
    return ((pt2-pt3)**2).sum()<((pt2-pt5)**2).sum()


# In[41]:


g0['good']=g0['vertices'].map(cmpdist)


# In[42]:


np.unique(g0['good'],return_counts=True)


# In[43]:


g0


# In[44]:


mask = (g0['good'] == False)
g0_tbm = g0[mask]


# In[45]:


def swapt(x):
    a=x
    a[3],a[5]=a[5],a[3]
    return a


# In[46]:


g0.loc[mask, 'vertices']= g0_tbm['vertices'].map(swapt)    


# In[47]:


g0


# In[48]:


g0['good']=g0['vertices'].map(cmpdist)


# In[49]:


np.unique(g0['good'],return_counts=True)


# In[50]:


np.unique(g0['type'])


# In[51]:


def myfloat(x,tol=10**(-6)):
    return tuple(np.array(x)*tol)


# In[78]:


tol=10**(-6)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)    

nfaces=20

types = g0.loc[:,'type'].values
center = np.stack(g0.loc[:,'center'].values)*tol
ax.scatter(center[:,0],center[:,1],center[:,2],marker='x',s=10)

faces = np.stack(g0.loc[:,'vertices'].values)*tol
for f in range(faces.shape[0]):
    xf,yf,zf = faces[f,:,0],faces[f,:,1],faces[f,:,2]
    vertsf=list(zip(xf,yf,zf))
    if types[f]==[1]:      # hexagones inside icosaedre face
        col='yellow'
    elif types[f]==[2,2]:  # hexagones edges between 2 icosaedre faces
        col='green'
    else:                  # hexagones edges between 2 icosaedre faces neighboors of pentagons
        col='red'
    ax.add_collection3d(Poly3DCollection([vertsf], facecolors = col, edgecolors='k', linewidths=1, alpha=0.9))
    
plt.show()


# In[365]:


def xycent(xoff7,yoff7,i,j,n):
    return xoff7+i/n+j/(2*n), yoff7+j*sqrt(3)/n


# In[366]:


xoff7=-1/2
yoff7=0
n=4
scale = 1/(n*sqrt(3))
fact=0
tol=10**(-6)

pentaBuild=pd.DataFrame(columns=['idx','face','xyc','th','type'])


#Penta #0 : top
idx0 = (0,1,2,3,4)
for k in idx0:
    info = {
        'idx':idx0,
        'face':k,
        'xyc':xycent(xoff7,yoff7,1,0,n), #type 4
        'th': 0,
        'type': 4
    }
    pentaBuild=pentaBuild.append(info,ignore_index=True)

    

######
#Pentas of the upper ring
######
#Penta #1 :
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,0,1,n),(-2*pi/3,3))
#cases[1,0],cases[1,1],cases[1,2],cases[1,3] =  expand(xycent(xoff7,yoff7,1,n-1,n),(2*pi/3,3))
#cases[2,0],cases[2,1],cases[2,2],cases[2,3] =  expand(xycent(xoff7,yoff7,n-1,0,n),(0,3))
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,0,n-1,n),(-2*pi/3,4))
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,n-1,1,n),(2*pi/3,4))
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,1,0,n),(0,4))

idx0 = (0,1,11,16,10)
infos=[]
"""
infos.append({
        'idx':idx0,
        'face':0,
        'xyc':xycent(xoff7,yoff7,0,1,n),
        'th': -2*pi/3,
        'type': 3
    })
infos.append({
        'idx':idx0,
        'face':0,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th': 2*pi/3,
        'type': 3
    })

infos.append({
        'idx':idx0,
        'face':0,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th': 0,
        'type': 3
    })

infos.append({
        'idx':idx0,
        'face':0,
        'xyc':xycent(xoff7,yoff7,0,n-1,n),
        'th': -2*pi/3,
        'type': 4
    })
"""
infos.append({
        'idx':idx0,
        'face':0,
        'xyc':xycent(xoff7,yoff7,n-1,1,n),
        'th': 2*pi/3,
        'type': 4
    })



infos.append({
        'idx':idx0,
        'face':1,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th': 0,
        'type': 3
    })

infos.append({
        'idx':idx0,
        'face':11,
        'xyc':xycent(xoff7,yoff7,1,0,n),
        'th':0, #-2*pi/3
        'type':4
    })

infos.append({
        'idx':idx0,
        'face':16,
        'xyc':xycent(xoff7,yoff7,1,0,n),
        'th':0, #-2*pi/3
        'type':4
    })

infos.append({
        'idx':idx0,
        'face':10,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th': 0, #0
        'type':3
    })

for info in infos:
    pentaBuild=pentaBuild.append(info,ignore_index=True)

"""


#Penta #2 : 1,2,11,12,17
idx0 = (1,2,12,17,11)
infos=[]
infos.append({
        'idx':idx0,
        'face':1,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':2,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':12,
        'xyc':xycent(xoff7,yoff7,0,1,n),
        'th':-2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':17,
        'xyc':xycent(xoff7,yoff7,0,1,n),
        'th':-2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':11,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })

for info in infos:
    pentaBuild = pentaBuild.append(info,ignore_index=True)

    
#Penta #3 :
idx0 = (2,3,13,18,12)
infos=[]
infos.append({
        'idx':idx0,
        'face':2,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':3,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0,
    })
infos.append({
        'idx':idx0,
        'face':13,
        'xyc':xycent(xoff7,yoff7,0,1,n),
        'th':-2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':18,
        'xyc':xycent(xoff7,yoff7,0,1,n),
        'th':-2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':12,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
for info in infos:
    pentaBuild = pentaBuild.append(info,ignore_index=True)

       
#Penta #4 :
idx0 = (3,4,14,19,13)
infos=[]
infos.append({
        'idx':idx0,
        'face':3,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':4,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':14,
        'xyc':xycent(xoff7,yoff7,0,1,n),
        'th':-2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':19,
        'xyc':xycent(xoff7,yoff7,0,1,n),
        'th':-2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':13,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
for info in infos:
    pentaBuild = pentaBuild.append(info,ignore_index=True)


#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,0,1,n),(-2*pi/3,3))
#cases[1,0],cases[1,1],cases[1,2],cases[1,3] =  expand(xycent(xoff7,yoff7,1,n-1,n),(2*pi/3,3))
#cases[2,0],cases[2,1],cases[2,2],cases[2,3] =  expand(xycent(xoff7,yoff7,n-1,0,n),(0,3))
    
#Penta #5 : 
idx0 = (4,0,10,15,14)
infos=[]
infos.append({
        'idx':idx0,
        'face':4,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':0,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':10,
        'xyc':xycent(xoff7,yoff7,0,1,n),
        'th':-2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':15,
        'xyc':xycent(xoff7,yoff7,0,1,n),
        'th':-2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':14,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
for info in infos:
    pentaBuild = pentaBuild.append(info,ignore_index=True)

######
#Pentas of the lower ring
######

#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,0,1,n),(-2*pi/3,3))
#cases[1,0],cases[1,1],cases[1,2],cases[1,3] =  expand(xycent(xoff7,yoff7,1,n-1,n),(2*pi/3,3))
#cases[2,0],cases[2,1],cases[2,2],cases[2,3] =  expand(xycent(xoff7,yoff7,n-1,0,n),(0,3))
    

#Penta #6 :
idx0 = (6,7,15,10,16)
infos=[]
infos.append({
        'idx':idx0,
        'face':6,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':7,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':15,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':10,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':16,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
for info in infos:
    pentaBuild = pentaBuild.append(info,ignore_index=True)

#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,0,1,n),(-2*pi/3,3))
#cases[1,0],cases[1,1],cases[1,2],cases[1,3] =  expand(xycent(xoff7,yoff7,1,n-1,n),(2*pi/3,3))
#cases[2,0],cases[2,1],cases[2,2],cases[2,3] =  expand(xycent(xoff7,yoff7,n-1,0,n),(0,3))
    

#Penta #7 :
idx0 = (5,6,16,11,17)
infos=[]
infos.append({
        'idx':idx0,
        'face':5,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':6,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':16,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':11,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':17,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
for info in infos:
    pentaBuild = pentaBuild.append(info,ignore_index=True)


#Penta #8 
idx0 = (9,5,17,12,18)
infos=[]
infos.append({
        'idx':idx0,
        'face':9,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':5,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':17,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':12,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':18,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
for info in infos:
    pentaBuild = pentaBuild.append(info,ignore_index=True)


#Penta #9 : 8,9,19,18,13
idx0 = (8,9,18,13,19)
infos=[]
infos.append({
        'idx':idx0,
        'face':8,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':9,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':18,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':13,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':19,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
for info in infos:
    pentaBuild = pentaBuild.append(info,ignore_index=True)


#Penta #10
idx0 = (7,8,19,14,15)
infos=[]
infos.append({
        'idx':idx0,
        'face':7,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':8,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':19,
        'xyc':xycent(xoff7,yoff7,n-1,0,n),
        'th':0
    })
infos.append({
        'idx':idx0,
        'face':14,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
infos.append({
        'idx':idx0,
        'face':15,
        'xyc':xycent(xoff7,yoff7,1,n-1,n),
        'th':2*pi/3
    })
for info in infos:
    pentaBuild = pentaBuild.append(info,ignore_index=True)
"""

#Penta #11  : bottom
idx0 = (5,6,7,8,9)
for k in idx0:
    info = {
        'idx':idx0,
        'face':k,
        'xyc':xycent(xoff7,yoff7,1,0,n), #type 4
        'th':0, # -2*pi/3
        'type':4
    }
    pentaBuild=pentaBuild.append(info,ignore_index=True)



# In[367]:


pentaBuild = pentaBuild.drop('face',axis=1)


# In[368]:


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)    

pentas = np.unique(pentaBuild['idx'].values)

print("pentas:",pentas)
#colors = ['red','blue','green','darkred','deepskyblue','forestgreen',
#         'pink','purple','lime','chocolate','darkviolet','chartreuse']
 
colors = ["black","red","blue","green","deepskyblue"]
    
for i in range(len(pentas)):
    idx0 = pentas[i]
    dftmp=pentaBuild[pentaBuild['idx']==idx0]
    print(dftmp)
    pts3d = []
    for ik,k in enumerate(idx0):
        a = icoTriangs[k,0]
        b = icoTriangs[k,1]
        c = icoTriangs[k,2]
        th = dftmp['th'].values[ik]
        xc,yc=dftmp['xyc'].values[ik]
        case=dftmp['type'].values[ik]
        if case == 3:
            pt2d = np.array([sqrt(3)/2-fact,1/2+fact/sqrt(3)]) # type 3 
        else:
            pt2d = np.array([-sqrt(3)/2+fact,1/2+fact/sqrt(3)]) # type 4 

        rot_mat = scale * np.array([[np.cos(th), -np.sin(th)],
                                    [np.sin(th), np.cos(th)]])

        pt2d = np.matmul(rot_mat,pt2d)
        pt2d[0] += xc
        pt2d[1] += yc

        pt3d = getProjectedPt(pt2d,icoPoints,a,b,c)
        pts3d.append(pt3d)

    pts3d = np.array(list(pts3d))
    ax.scatter(pts3d[:,0],pts3d[:,1],pts3d[:,2],marker='o',s=15,color=colors)
    
    vertsf=list(zip(pts3d[:,0],pts3d[:,1],pts3d[:,2]))
    ax.add_collection3d(Poly3DCollection([vertsf], facecolors = 'purple', edgecolors='k', linewidths=1, alpha=0.2))
 
    
    
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.show()


# In[212]:


pentaBuild.head(10)


# Test de groupby idx de pentaBuild pour ensuite calculer le centre de chaque pentagone

# In[149]:


pentaBuild=pentaBuild.groupby('idx',as_index=False).agg(lambda x: x.tolist())


# In[150]:


pentaBuild


# In[151]:


pentaBuild['info']=[[(*a, b) for a, b in zip(x, y)] for x, y in zip(pentaBuild['xyc'],pentaBuild['th'])]
pentaBuild=pentaBuild.drop(['xyc','th'],axis=1)


# In[152]:


pentaBuild


# In[153]:


for row in pentaBuild.itertuples():
#    print(row.idx,row.info)
    idx0 = row.idx
    info0 = np.array(row.info)
    for ik,k in enumerate(idx0):
        xc,yc,th=info0[ik][0],info0[ik][1],info0[ik][2]
        #print("ik:",ik,xc,yc,th)


# In[154]:


pentaBuild


# In[156]:


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)    


#print("pentas:",pentas)
colors = ['red','blue','green','darkred','deepskyblue','forestgreen',
         'pink','purple','lime','chocolate','darkviolet','chartreuse']

for row in pentaBuild.itertuples():
    def make_pts3d(row):
        idx0 = row.idx
        info0 = np.array(row.info) 
        pts3d = []
        for ik,k in enumerate(idx0):
            a = icoTriangs[k,0]
            b = icoTriangs[k,1]
            c = icoTriangs[k,2]
            xc,yc,th=info0[ik][0],info0[ik][1],info0[ik][2]
            pt2d = np.array([sqrt(3)/2-fact,1/2+fact/sqrt(3)]) # type 3 
            rot_mat = scale * np.array([[np.cos(th), -np.sin(th)],
                                        [np.sin(th), np.cos(th)]])

            pt2d = np.matmul(rot_mat,pt2d)
            pt2d[0] += xc
            pt2d[1] += yc

            pt3d = myround(getProjectedPt(pt2d,icoPoints,a,b,c))
            pts3d.append(pt3d)

        pts3d = np.array(list(pts3d))*tol
        ax.scatter(pts3d[:,0],pts3d[:,1],pts3d[:,2],marker='o',s=15,color='k')

        vertsf=list(zip(pts3d[:,0],pts3d[:,1],pts3d[:,2]))
        ax.add_collection3ad(Poly3DCollection([vertsf], facecolors = 'purple', edgecolors='k', linewidths=1, alpha=0.2))
        return vertsf
    
    pentaBuild['pts3d'] =pentaBuild.apply(make_pts3d, axis=1)
    
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.show()


# In[1823]:


dfpenta= pentaBuild[['idx','pts3d']]


# In[1824]:


dfpenta


# In[1827]:


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)    
#print("pentas:",pentas)
colors = ['red','blue','green','darkred','deepskyblue','forestgreen',
         'pink','purple','lime','chocolate','darkviolet','chartreuse']
for row in dfpenta.itertuples():
    vertsf=row.pts3d
    ax.add_collection3d(Poly3DCollection([vertsf], facecolors = 'purple', edgecolors='k', linewidths=1, alpha=0.2))

ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)
    
plt.show()


# In[1848]:


dfpenta['xyzc']=dfpenta['pts3d']                    .map(lambda x: np.array(x).mean(axis=0))                    .map(lambda x: x/sqrt(sum(x*x)))


# In[1851]:


dfpenta=dfpenta.rename(columns={'pts3d':'vertices','xyzc':'center'})


# In[1852]:


dfpenta


# In[1853]:


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)    
#print("pentas:",pentas)
colors = ['red','blue','green','darkred','deepskyblue','forestgreen',
         'pink','purple','lime','chocolate','darkviolet','chartreuse']
for row in dfpenta.itertuples():
    vertsf=row.vertices
    ax.add_collection3d(Poly3DCollection([vertsf], facecolors = 'purple', edgecolors='k', linewidths=1, alpha=0.2))
    xyzc = row.center
    ax.scatter(xyzc[0],xyzc[1],xyzc[2],marker='o',s=15,color='k')

    
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)
    
plt.show()


# In[1858]:


dfhexa=g0.drop(['Nvertices','nfaces','good'],axis=1)


# In[1861]:


dfhexa=dfhexa[dfhexa.columns[[1,2,3,0]]]


# In[1862]:


dfhexa


# In[1864]:


dfhexa['type']=dfhexa['type'].map(lambda x: 1 if x==[1] else (2 if x==[2,2] else 3))


# In[1869]:


dfhexa['vertices']=dfhexa['vertices'].map(lambda x: np.array(x)*tol)


# In[1871]:


dfhexa['center']=dfhexa['center'].map(lambda x: np.array(x)*tol)


# In[1872]:


dfhexa


# In[1873]:


dfpenta['type']=0


# In[1875]:


dfpenta=dfpenta[dfpenta.columns[[0,3,1,2]]]


# In[1876]:


dfpenta


# In[1877]:


df_row_merged = pd.concat([dfpenta, dfhexa], ignore_index=True)


# In[1878]:


df_row_merged 


# In[1880]:


arr=[1,2,3,4,5,6]
def swaptmp(x):
    x[3],x[5]=x[5],x[3]
    return x
swaptmp(arr)


# In[1887]:


df_tmp=pd.DataFrame({'A':[[1,2,3,4,5,6],[1,2,3,4,5,6]],'good':[True,False]})


# In[1888]:


df_tmp


# In[1889]:


mask = (df_tmp['good'] == False)
df_tbm = df_tmp[mask]


# In[1890]:


df_tbm


# In[1893]:


df_tmp.loc[mask, 'A']= df_tbm['A'].map(swaptmp)


# In[1894]:


df_tmp


# In[265]:


def expand(x,y):
    return sum((x,y),())


# In[1220]:


fact=0
n=4
xoff7=-1/2
yoff7=0
scale = 1/(n*sqrt(3))


cases = np.zeros((1,4))
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,0,1,n),(-2*pi/3,3))
#cases[1,0],cases[1,1],cases[1,2],cases[1,3] =  expand(xycent(xoff7,yoff7,1,n-1,n),(2*pi/3,3))
#cases[2,0],cases[2,1],cases[2,2],cases[2,3] =  expand(xycent(xoff7,yoff7,n-1,0,n),(0,3))
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,0,n-1,n),(-2*pi/3,4))
#cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,n-1,1,n),(2*pi/3,4))
print('verif: ',expand(xycent(xoff7,yoff7,1,0,n),(0,4)))
cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,1,0,n),(0,4))


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)    

for k in [0,1,2,3,4]:
#for k in [5,6,7,8,9]:
    a = icoTriangs[k,0]
    b = icoTriangs[k,1]
    c = icoTriangs[k,2]
    print(k,a,b,c)
    pts3d = []
    col = []
    print("cases: ",cases)
    for cas in cases:
        th = cas[2]
        if cas[3] == 3:
            pt2d = np.array([sqrt(3)/2-fact,1/2+fact/sqrt(3)]) # type 3 
            col.append('k')
        else:
            pt2d = np.array([-sqrt(3)/2+fact,1/2+fact/sqrt(3)]) # type 4 
            col.append('r')
            
        rot_mat = scale * np.array([[np.cos(th), -np.sin(th)],
                                        [np.sin(th), np.cos(th)]])
        

        pt2d = np.matmul(rot_mat,pt2d)
        pt2d[0] += cas[0]
        pt2d[1] += cas[1]
        
        print(cas[0],cas[1],pt2d)
        
        pt3d = myround(getProjectedPt(pt2d,icoPoints,a,b,c))

        
        print(pt3d)
        
        pts3d.append(pt3d)

    
    pts3d = np.array(list(pts3d))*tol
    ax.scatter(pts3d[:,0],pts3d[:,1],pts3d[:,2],marker='o',s=15,color=col[:])

#    vertsf=list(zip(pts3d[:,0],pts3d[:,1],pts3d[:,2]))
#    ax.add_collection3d(Poly3DCollection([vertsf], facecolors = 'purple', edgecolors='k', linewidths=1, alpha=0.2))
 
    
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.show


# In[134]:


t0,t1,t2,t3=expand(xycent(-1/2,0,0,1,4),(0,4))


# In[135]:


print(t0,t1,t2,t3)


# In[133]:


cases


# In[131]:


def xycent(xoff7,yoff7,i,j,n):
    '''
            2D localisation of the center of a pentagon in the frame of a icosaedre face
    '''
    return xoff7+i*1/n+j*1/(2*n), yoff7+j*sqrt(3)/n


# In[136]:


n=4
xoff7=-1/2
yoff7=0
scale = 1/(n*sqrt(3))
print('verif: ',expand(xycent(xoff7,yoff7,1,0,n),(0,4)))
cases[0,0],cases[0,1],cases[0,2],cases[0,3] =  expand(xycent(xoff7,yoff7,1,0,n),(0,4))
print(cases)


# In[ ]:




