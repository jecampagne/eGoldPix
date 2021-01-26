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


# In[744]:


get_ipython().run_line_magic('pinfo', 'Timer')


# In[2]:


# timer.py

from contextlib import ContextDecorator
from dataclasses import dataclass, field
import time
from typing import Any, Callable, ClassVar, Dict, Optional

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""

    timers: ClassVar[Dict[str, float]] = {}
    counts: ClassVar[Dict[str, int]] = {}
    
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)
            self.counts.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time
            self.counts[self.name] += 1

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()


# In[ ]:





# In[3]:


class isocapix:
    def __init__(self,n,fact=0,tol=10**(-6)):
        '''
            n: order of the pixelization,
                # of pixels = 10*n^2+2 
                among them 12 are pentagon-like  and the rest are hexagon-like
            tol: a rounding factor to distinguish between 2 floats
        '''
        self.n    = n
        self.tol  = tol
        self.fact = fact
        
        self.xoff7 = -1/2
        self.yoff7 = 0

        #icosedron vertices
        self.icoPoints = self.getIcosaedreVertices()
        
        # number of isocaedron faces
        self.nIsofaces = 20
        # All triplet vertices of the 20  equilateral triangles
        self.icoTriangs = [(0,i+1,(i+1)%5+1) for i in range(5)] +             [(6,i+7,(i+1)%5+7) for i in range(5)] +             [(i+1,(i+1)%5+1,(7-i)%5+7) for i in range(5)] +             [(i+1,(7-i)%5+7,(8-i)%5+7) for i in range(5)]
        self.icoTriangs=np.array(self.icoTriangs)
        
        
        # scaling factor   
        self.scale = 1/(n*sqrt(3))
        
        # build the different pixels
        Timer.timers={}
        Timer.counts={}
        
        self.t1 = Timer("Hexa All", logger=None)
        self.tmapGridpoint2Sogere = Timer("mapGridpoint2Sogere",logger=None) 
        self.tslerp = Timer("slerp", logger=None)
        self.tbarycentricCoords = Timer("barycentricCoords", logger=None)
        
        self.t1.start()
        self.dfhexa = self.makeHexagon()
        self.t1.stop()


        totHexa_time = Timer.timers["Hexa All"]

        print(f"Hexa: {totHexa_time:0.2f} seconds")

        totHexa_mapGridpoint2Sogere = Timer.timers["mapGridpoint2Sogere"]
        print(f"==> mapGridpoint2Sogere: {totHexa_mapGridpoint2Sogere:0.2f} sec")
        
        totHexa_slerp = Timer.timers["slerp"]
        totHexa_slerp_cnt = Timer.counts["slerp"]
        print(f"==> slerp: {totHexa_slerp:0.2f} sec, cnt: {totHexa_slerp_cnt}")

        totHexa_barycentricCoords = Timer.timers["barycentricCoords"]
        print(f"==> barycentricCoords: {totHexa_barycentricCoords:0.2f} sec")
        self.t2 = Timer("Penta All", logger=None)
        self.t2.start()
        self.dfpenta  = self.makePentagon()
        self.t2.stop()
        totPenta_time = Timer.timers["Penta All"]
        print(f"Penta: {totPenta_time:0.2f} seconds")

        
        
        
        self.dfAll = pd.concat([self.dfpenta, self.dfhexa], ignore_index=True)


    #################################################################
    def barycentricCoords(self,p):
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
        self.tbarycentricCoords.start()

        x,y = p[0],p[1]
        # l3*sqrt(3)/2 = y
        l3 = y*2./sqrt(3.)
        # l1 + l2 + l3 = 1
        # 0.5*(l2 - l1) = x
        l2 = x + 0.5*(1 - l3)
        l1 = 1 - l2 - l3
        
        self.tbarycentricCoords.stop()

        
        return l1,l2,l3

    #################################################################
    def scalProd(self, p1,p2):
        '''
            input: p1 and p2 are the vetors of form [x0,x1,...,xn]'
            output: is the scalar product of p1 and p2.
            (nb. apply for list &  tuple & np.array)
        '''
        return sum([p1[i]*p2[i] for i in range(len(p1))])

    #################################################################
    def slerp(self,p0,p1,t):
        '''
            program outputs the spherical linear interpolation 
            of arc defined by p0, p1(around origin).  

            input: t=0 -> p0, t=1 -> p1. 
                    p0 and p1 are the vetors of form [x,y,z]

            output: interpolated coordinates.

            https://en.wikipedia.org/wiki/Slerp

        '''
###        assert abs(self.scalProd(p0,p0) - self.scalProd(p1,p1)) < self.tol

        self.tslerp.start()
    
        ang0Cos = self.scalProd(p0,p1)/self.scalProd(p0,p0)
        ang0Sin = sqrt(1 - ang0Cos*ang0Cos)
        ang0 = atan2(ang0Sin,ang0Cos)
        l0 = sin((1-t)*ang0)
        l1 = sin(t    *ang0)
        tmp= np.array([(l0*p0[i] + l1*p1[i])/ang0Sin for i in range(len(p0))])
        
        self.tslerp.stop()
        return tmp
    
    #################################################################
    def mapGridpoint2Sogere(self,p,s1,s2,s3):
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
                    
        self.tmapGridpoint2Sogere.start() 

        l1,l2,l3 = self.barycentricCoords(p)
        if abs(l3-1) < self.tol: return s3
        l2s = l2/(l1+l2)
        p12 = self.slerp(s1,s2,l2s)
        tmp= self.slerp(p12,s3,l3)
        
        self.tmapGridpoint2Sogere.stop() 
        return tmp

    #################################################################
    def hexagon(self,x,y,th,opt):
        '''
            this program creates the hexagon of given configuration and size.
            inputs: 
                - x and y are the rectangular coordinates of the center of the hexagon
                - th is the rotation angle measured anticlockwise positive 
                    from positive x axis
                - scale : is a contraction/dilation factor
                - opt: 1 full hexagon. 2 half hexagon
                - fact: a factor to shrink the hexagons neightbooring the pentagons  

                output: planar hexagon (complete/truncated) orientated 
            example:
                hexagon(0,0,np.pi/6,0.5,1)
        '''
        # rotation matrx with scale (th>0 the transformation is anti-clockwise)
        rot_mat = self.scale * np.array([[np.cos(th), -np.sin(th)],
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
            hex[0,:]= np.array([sqrt(3)/2-self.fact,sqrt(3)/2-self.fact,0,-sqrt(3)/2,-sqrt(3)/2]) # X-ccod
            hex[1,:]= np.array([0,1/2+self.fact/sqrt(3),1,0.5,0]) # Y-coord

        elif opt == 4:
            # point 3 et 4 sont modifiers par rapport au type 2
            hex = np.zeros((2,5))
            hex[0,:]= np.array([sqrt(3)/2,sqrt(3)/2,0,-sqrt(3)/2+self.fact,-sqrt(3)/2+self.fact]) # X-ccod
            hex[1,:]= np.array([0,0.5,1,1/2+self.fact/sqrt(3),0]) # Y-coord


        hex = np.matmul(rot_mat,hex)

        hex[0,:]= x+hex[0,:] 
        hex[1,:]= y+hex[1,:]

        return hex
    
    #################################################################
    def getIcosaedreVertices(self):
        """
            outputs location of the icosaedre vertices 3D points
        """
        #golden ratio
        phi = 0.5*(1+sqrt(5)) 

        topPoints =             [(phi,1,0)]+            [(phi,-1,0)]+            [(1,0,-phi)]+            [(0,phi,-1)]+            [(0,phi,1)]+            [(1,0,phi)]

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
    
    #################################################################
    def getProjectedFace(self,hexag,u,v,w):
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

            face[:,i] = self.mapGridpoint2Sogere(hexag[:,i],
                                            self.icoPoints[u,:],
                                            self.icoPoints[v,:],
                                            self.icoPoints[w,:])
    
        return face

    #################################################################
    def getProjectedPt(self,p,u,v,w):
        """
        p: 2D point location
        """
        return self.mapGridpoint2Sogere(p,
                                    self.icoPoints[u,:],
                                    self.icoPoints[v,:],
                                    self.icoPoints[w,:])
    
    #################################################################
    def rounding(self,x):
        """
            transform "x" (eg. list of floats) to tuple of integers up to a certain precision 
        """
        return tuple(np.round(np.array(x)/self.tol).astype(int))

    #################################################################
    def tofloat(self,x):
        """
            reverse of 'rounding' method up to the precision level
        """
        return tuple(np.array(x)*self.tol)

    #################################################################
    def keep_unique(self,x):
        a=[i for i in x if x.count(i)==1]
        return a
        
        
    #################################################################
    def cmpdist(self,x):
        """
            Context: After glueing half-hexagonal pixels the chain of the vertices
          x=(p0,p1,p2,p3,p4,p5) can be in a wrong sequence. 
           So this function compare dist(p2,p3) versus dist(p2,p5) to trigger a swapping operation 
           (see next function)
        """
        pt2,pt3,pt5=np.array(x[2]),np.array(x[3]),np.array(x[5])
        return ((pt2-pt3)**2).sum()<((pt2-pt5)**2).sum()
    
    #################################################################
    def swapt(self,x):
        x[3],x[5]=x[5],x[3]
        return x

    #################################################################
    # Main method to build hexagons
    #################################################################
    def makeHexagon(self):
        """
        return a dataframe with the hexagones
        """
        tmakeHexagon = Timer("makeHexagon", logger=None)
        tmakeHexagon.start()
        
        icoTriangs = self.icoTriangs
        faces = self.nIsofaces
        n = self.n
        
        listDF=[]
            
        #### df = pd.DataFrame(columns=['idx','type','center','vertices'])

        
        #exhaustive loop over the 20 faces of the icosaedre to build each hexagon
        thexa = Timer("hexagon", logger=None)
        tprojFace = Timer("getProjectedFace", logger=None)
        tprojPt   = Timer("getProjectedPt",logger=None)
        tdfappend = Timer("dfappend",logger=None)
        for k in range(faces):
                # (i,j) is a couple of indice that uniquely identify an hexagon
                # opt=1: full hexagone in the "middle" of the face
                # opt=2: hexagone at a edge between two faces but not affected by a possible deformation 
                # opt=3 or 4: hexagone like opt=2 but which neightboor of a pentagon so possibly
                #             affected by a deformation
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
                        #make the hexagon in the 2D frame of a generic equlateral trinagle
                        # orig hexagcenter = np.array([self.xoff7+i*1/n+j*1/(2*n), self.yoff7+j*sqrt(3)/(2*n)])
                        hexagcenter = np.array(self.xycent(self.xoff7,self.yoff7,i,j,n))
                        thexa.start()
                        hexag = self.hexagon(hexagcenter[0],hexagcenter[1],th,opt)
                        thexa.stop()
                        #project it on the sphere
                        a = self.icoTriangs[k,0]
                        b = self.icoTriangs[k,1]
                        c = self.icoTriangs[k,2]
                        #get list of the 3D vertices of the hexagon (possibly truncated and deformed)
                        tprojFace.start()
                        face = self.getProjectedFace(hexag,a,b,c)
                        tprojFace.stop()
                        xf,yf,zf = face[0,:],face[1,:],face[2,:]
                        vertsf=list(zip(xf,yf,zf))
                        #get the 3D position of the original center
                        #
                        # Todo: for deformed hexagon this should be changed later after the glue of half
                        #       hexagon shared by 2 faces
                        # 
                        tprojPt.start()
                        center = self.getProjectedPt(hexagcenter,a,b,c)
                        tprojPt.stop()
                        # Dataframe update
                        tdfappend.start()
                        listDF.append([(k,i,j),opt,(center[0],center[1],center[2]),vertsf])
                         
#                        df=df.append({'idx':(k,i,j),
#                                             'type':opt,
#                                              'center':(center[0],center[1],center[2]),
#                                              'vertices':vertsf},ignore_index=True)
                        tdfappend.stop()
                        
        
        #eo loop on faces
        
        tdfOperations= Timer("dfOpAll",logger=None)
        tdfOperations.start()

        df = pd.DataFrame(listDF,columns=['idx','type','center','vertices'])
        
        # transform the 3D 'float' positions into 3D 'integer' position up to a precision
        #     to be able to perform groupby etc. Nb. otherwise groupby with tuple of floats cannot
        #     manage to merge for instance two 3D points close from each other due to precision
        df['center']  =df['center'].map(self.rounding)
        df['vertices']=df['vertices'].map(self.rounding)
        # groupby center of the pixel polygon 
        df=df.groupby('center', as_index=False).agg({'idx':'sum','type':lambda x: list(x),'vertices':'sum'})
        # for each pixel polygon keep only the vertices that occure once. By this operation we kill
        # vertices that are shared by half-hexagons of two faces that are glued to make a single hexagon
        # at icosaedre edges.
        df['vertices']=df['vertices'].map(lambda x: [list(y) for y in x])                        .map(lambda x: tuple([tuple(y) for y in x]))                        .map(self.keep_unique)
        # After glueing half-hexagonal pixels the chain of the vertices
         #x=(p0,p1,p2,p3,p4,p5) can be in a wrong sequence. 
        df['good']=df['vertices'].map(self.cmpdist)
        mask = (df['good'] == False)
        df_tbm = df[mask]
        df.loc[mask, 'vertices']= df_tbm['vertices'].map(self.swapt)
        #clean
        df=df.drop(['good'],axis=1)
        #re organize the columns which as been changed after the groupby center
        df=df[df.columns[[1,2,3,0]]]

        #rephrase the typology of the hexagons
        # 1: hexagons in the middle of each icosaedre faces
        # 2: hexagons at edges of two icosaedre faces
        # 3: hexagons neightbooring a pentagon
        df['type']=df['type'].map(lambda x: 1 if x==[1] else (2 if x==[2,2] else 3))
 
        #rescale to get floating numbers for the vertices and the center
        df['vertices']=df['vertices'].map(lambda x: np.array(x)*self.tol)
        df['center']=df['center'].map(lambda x: np.array(x)*self.tol)
        
                
        tdfOperations.stop()

        tmakeHexagon.stop()

        
        #got it
        tot_time = Timer.timers["hexagon"]
        print(f"hexagon: {tot_time:0.2f} seconds")
        
        tot_time = Timer.timers["getProjectedFace"]
        print(f"projFace: {tot_time:0.2f} seconds")
        
        tot_time  = Timer.timers["getProjectedPt"]
        print(f"projPt: {tot_time:0.2f} seconds")
        
        tot_time= Timer.timers["dfappend"]
        print(f"DfAppend: {tot_time:0.2f} seconds")

        tot_time = Timer.timers["dfOpAll"]
        print(f"Df processing All: {tot_time:0.2f} seconds")
    
        tot_time = Timer.timers["makeHexagon"]
        print(f"(verif) makeHexagon: {tot_time:0.2f} seconds")
        
        return df
    
    #################################################################
    def xycent(self,xoff7,yoff7,i,j,n):
        '''
            2D localisation of the center of a pentagon in the frame of a icosaedre face
        '''
        return xoff7+i/n+j/(2*n), yoff7+j*sqrt(3)/(2*n)

        
    #################################################################
    # Main method to build pentagons
    #################################################################
    def makePentagon(self):
        """
         There are 12 only pentagons
        """
        
        pentaBuild=pd.DataFrame(columns=['idx','face','xyc','th'])
        
        #below idx0 is a tuple with the icosaedre face number (the order is important)
        #we build a DF of the vertices of each pentagon positionned in local 2D icosadre-face frame
        
        xoff7 = self.xoff7
        yoff7 = self.yoff7
        n     = self.n
                        
        #Penta #0 : top
        idx0 = (0,1,2,3,4)
        for k in idx0:
            info = {
                'idx':idx0,
                'face':k,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th':0 #'th':-2*pi/3
            }
            pentaBuild=pentaBuild.append(info,ignore_index=True)

        
        ######
        #Pentas of the upper ring
        ######
        #Penta #1 :
        idx0 = (0,1,11,16,10)
        infos=[]
        infos.append({
                'idx':idx0,
                'face':0,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th':-2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':1,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th':2*pi/3
            })

        infos.append({
                'idx':idx0,
                'face':11,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th': 0
            })

        infos.append({
                'idx':idx0,
                'face':16,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th': 0
            })

        infos.append({
                'idx':idx0,
                'face':10,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })

        for info in infos:
            pentaBuild=pentaBuild.append(info,ignore_index=True)

        #Penta #2
        idx0 = (1,2,12,17,11)
        infos=[]
        infos.append({
                'idx':idx0,
                'face':1,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th':-2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':2,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th':2*pi/3
            })

        infos.append({
                'idx':idx0,
                'face':12,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th': 0
            })

        infos.append({
                'idx':idx0,
                'face':17,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th': 0
            })

        infos.append({
                'idx':idx0,
                'face':11,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })

        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)

        #Penta #3 :
        idx0 = (2,3,13,18,12)
        infos=[]
        
        infos.append({
                'idx':idx0,
                'face':2,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th':-2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':3,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th':2*pi/3
            })

        infos.append({
                'idx':idx0,
                'face':13,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th': 0
            })

        infos.append({
                'idx':idx0,
                'face':18,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th': 0
            })

        infos.append({
                'idx':idx0,
                'face':12,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })

        
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)

        
        #Penta #4 :
        idx0 = (3,4,14,19,13)
        infos=[]

        infos.append({
                'idx':idx0,
                'face':3,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th':-2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':4,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th':2*pi/3
            })

        infos.append({
                'idx':idx0,
                'face':14,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th': 0
            })

        infos.append({
                'idx':idx0,
                'face':19,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th': 0
            })

        infos.append({
                'idx':idx0,
                'face':13,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)

        #Penta #5 : 
        idx0 = (4,0,10,15,14)
        infos=[]

        infos.append({
                'idx':idx0,
                'face':4,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th':-2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':0,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th':2*pi/3
            })

        infos.append({
                'idx':idx0,
                'face':10,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th': 0
            })

        infos.append({
                'idx':idx0,
                'face':15,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th': 0
            })

        infos.append({
                'idx':idx0,
                'face':14,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })

        
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)


        ######
        #Pentas of the lower ring
        ######
        """
        cases=np.array([
    [0,n-1,-2*pi/3,4],
    [n-1,1,2*pi/3,4],
    [1,0,0.,4]
])
        """

        #Penta #6 :
        idx0 = (6,7,15,10,16)
        infos=[]
        infos.append({
                'idx':idx0,
                'face':6,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th':-2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':7,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':15,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':10,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th': -2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':16,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th': -2*pi/3
            })
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)

        #Penta #7 :
        idx0 = (5,6,16,11,17)
        infos=[]

        infos.append({
                'idx':idx0,
                'face':6,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th':-2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':7,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':15,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':10,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th': -2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':16,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th': -2*pi/3
            })
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)


        #Penta #8 
        idx0 = (9,5,17,12,18)
        infos=[]

        infos.append({
                'idx':idx0,
                'face':6,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th':-2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':7,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':15,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':10,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th': -2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':16,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th': -2*pi/3
            })
        
        
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)

        #Penta #9 
        idx0 = (8,9,18,13,19)
        infos=[]
        infos.append({
                'idx':idx0,
                'face':6,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th':-2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':7,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':15,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':10,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th': -2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':16,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th': -2*pi/3
            })
        
        
        
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)


        
        #Penta #10
        idx0 = (7,8,19,14,15)
        infos=[]
        infos.append({
                'idx':idx0,
                'face':6,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th':-2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':7,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':15,
                'xyc':self.xycent(xoff7,yoff7,n-1,1,n),
                'th': 2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':10,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th': -2*pi/3
            })
        infos.append({
                'idx':idx0,
                'face':16,
                'xyc':self.xycent(xoff7,yoff7,0,n-1,n),
                'th': -2*pi/3
            })

        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)
        
        
        #Penta #11  : bottom
        idx0 = (5,6,7,8,9)
        for k in idx0:
            info = {
                'idx':idx0,
                'face':k,
                'xyc':self.xycent(xoff7,yoff7,1,0,n),
                'th':0 #-2*pi/3
            }
            pentaBuild=pentaBuild.append(info,ignore_index=True)

            
        #We can know drop face column
        pentaBuild = pentaBuild.drop('face',axis=1)
                
        #We group by idx and then get (x,y,th) triplet list
        pentaBuild=pentaBuild.groupby('idx',as_index=False).agg(lambda x: x.tolist())
        pentaBuild['info']=[[(*a, b) for a, b in zip(x, y)] for x, y in zip(pentaBuild['xyc'],pentaBuild['th'])]
    
        pentaBuild=pentaBuild.drop(['xyc','th'],axis=1)

        #print(pentaBuild.head(10))
        
        
        #get the 3D coordinates of the pentagon vertices
        for row in pentaBuild.itertuples():
            def make_pts3d(row):
                idx0 = row.idx
                info0 = np.array(row.info) 
                pts3d = []
                for ik,k in enumerate(idx0):
                    a = self.icoTriangs[k,0]
                    b = self.icoTriangs[k,1]
                    c = self.icoTriangs[k,2]
                    
                    #print(k,a,b,c)
                    
                    xc,yc,th=info0[ik][0],info0[ik][1],info0[ik][2]
                    ##pt2d = np.array([sqrt(3)/2-self.fact,1/2+self.fact/sqrt(3)]) # type 3 
                    pt2d = np.array([-sqrt(3)/2+self.fact,1/2+self.fact/sqrt(3)]) # type 4 
                    rot_mat = self.scale * np.array([[np.cos(th), -np.sin(th)],
                                                [np.sin(th), np.cos(th)]])
                    

                    pt2d = np.matmul(rot_mat,pt2d)
                    pt2d[0] += xc
                    pt2d[1] += yc


                    #pt3d = self.rounding(self.getProjectedPt(pt2d,a,b,c))
                    pt3d = self.getProjectedPt(pt2d,a,b,c)
                    
                    
                    pts3d.append(pt3d)

                pts3d = np.array(list(pts3d))#*self.tol
                vertsf=list(zip(pts3d[:,0],pts3d[:,1],pts3d[:,2]))
                return vertsf
        #
        pentaBuild['vertices']=pentaBuild.apply(make_pts3d, axis=1)
        #drop the intermediate "info"
        pentaBuild=pentaBuild.drop('info',axis=1)
        # compute pentagon barycenter and project onto the sphere
        pentaBuild['center']=pentaBuild['vertices']                    .map(lambda x: np.array(x).mean(axis=0))                    .map(lambda x: x/sqrt(sum(x*x)))
        
        #To uniformize with the DF of the hexagons
        pentaBuild['type']=0
        pentaBuild=pentaBuild[pentaBuild.columns[[0,3,1,2]]]
        
        
        
        # Got it!
        return pentaBuild
    


# In[ ]:





# In[824]:


mypix=isocapix(10,fact=0)


# In[299]:


mypix.dfpenta


# In[253]:


#mypix.dfhexa


# In[ ]:





# In[254]:



fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)    


types = mypix.dfhexa.loc[:,'type'].values
center = np.stack(mypix.dfhexa.loc[:,'center'].values)
idx = mypix.dfhexa.loc[:,'idx'].values
idx=[i[0] for i in idx]
ax.scatter(center[:,0],center[:,1],center[:,2],marker='x',s=10)
#for k in range(center.shape[0]):
#    ax.text(center[k,0]*1.1,center[k,1]*1.1,center[k,2]*1.1,'%s' % (str(idx[k])), 
#                            size=10, zorder=1, color='k')


faces = np.stack(mypix.dfhexa.loc[:,'vertices'].values)
for f in range(faces.shape[0]):
    xf,yf,zf = faces[f,:,0],faces[f,:,1],faces[f,:,2]
    vertsf=list(zip(xf,yf,zf))
    if types[f]==1:      # hexagones inside icosaedre face
        col='yellow'
    elif types[f]==2:  # hexagones edges between 2 icosaedre faces
        col='green'
    else:                  # hexagones edges between 2 icosaedre faces neighboors of pentagons
        col='red'
    ax.add_collection3d(Poly3DCollection([vertsf], facecolors = col, edgecolors='k', linewidths=1, alpha=0.9))

    
for row in mypix.dfpenta.itertuples():
    vertsf=row.vertices
    ax.add_collection3d(Poly3DCollection([vertsf], facecolors = 'purple', edgecolors='k', linewidths=1, alpha=1))
    xyzc = row.center
    ax.scatter(xyzc[0],xyzc[1],xyzc[2],marker='o',s=15,color='k')
    
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

    
plt.show()


# In[297]:


mypix.dfhexa


# Calcul des rayons Rin/Rout sur les pixels.

# In[300]:


dftmp = mypix.dfhexa.copy(deep=True)


# In[301]:


dftmp


# In[302]:


dftmp['c2']=dftmp['vertices'].map(lambda x: np.mean(x,axis=0)).map(lambda x : x/np.sqrt(np.sum(x*x)))


# In[306]:


dftmp['diff']=(dftmp['center']-dftmp['c2']).map(lambda x: np.mean(x))


# In[309]:


np.max(np.abs(dftmp['diff'].values))


# In[315]:


dftmp=dftmp.drop('diff',axis=1)


# In[316]:


dftmp


# In[419]:


def radout(x,c):
    return np.max(np.sqrt(np.sum((x-c)**2,axis=1)))


# In[420]:


dftmp['radout']=dftmp[['vertices','c2']].apply(lambda x: radout(x.vertices,x.c2), axis=1)


# In[421]:


dftmp['radout'].describe()


# In[414]:


def altitude(a,b,c):
    """
        Heron formula c is the base of a triangle and (a,b) are the lengthes of the other edges.
    """
    p=(a+b+c)/2.
    return 2.*np.sqrt(p*(p-a)*(p-b)*(p-c))/c
    


# In[433]:


def radint(x,c):
    # transform x:(x1,x2,...,xn) into (x1,x2,...,xn,x1)
    a = np.vstack((x,x[0]))
    # ||x2-x1||,||x3-x2||,...,||xn-xn-1||,||x1-xn||  (n terms)
    edgeLength = np.sqrt(np.sum(np.diff(a,axis=0)**2,axis=1))
    # ||c-x1||,||c-x2||,..., ||c-xn||      (n terms)
    radiiLength= np.sqrt(np.sum((x-c)**2,axis=1))    
    #
    alt=[]
    for i in range(len(radiiLength)-1):
        alt.append(altitude(radiiLength[i],radiiLength[i+1],edgeLength[i]))
    return np.min(alt)


# In[434]:


x=dftmp['vertices'][0]


# In[429]:


c=dftmp['c2'][0]


# In[430]:


radint(x,c)


# In[435]:


dftmp['radint']=dftmp[['vertices','c2']].apply(lambda x: radint(x.vertices,x.c2), axis=1)


# In[437]:


dftmp['radint'].describe()


# In[438]:


dftmp['radout'].describe()


# In[452]:


dftmp['ratio_norm']=sqrt(3)/2*dftmp['radout']/dftmp['radint']


# In[453]:


dftmp['ratio_norm'].describe()


# les pentagones..

# In[442]:


dftmp = mypix.dfpenta.copy(deep=True)


# In[443]:


dftmp


# In[444]:


dftmp['c2']=dftmp['vertices'].map(lambda x: np.mean(x,axis=0)).map(lambda x : x/np.sqrt(np.sum(x*x)))


# In[445]:


dftmp


# In[446]:


dftmp['radout']=dftmp[['vertices','c2']].apply(lambda x: radout(x.vertices,x.c2), axis=1)


# In[447]:


dftmp['radout'].describe()


# In[448]:


dftmp['radint']=dftmp[['vertices','c2']].apply(lambda x: radint(x.vertices,x.c2), axis=1)


# In[449]:


dftmp['radint'].describe()


# In[455]:


dftmp['ratio_norm']=sqrt(25+10*sqrt(5))/sqrt(50+10*sqrt(5))*dftmp['radout']/dftmp['radint']


# In[456]:


dftmp['ratio_norm'].describe()


# In[458]:


2*sqrt(3)* (0.142**2)


# In[460]:


25/sqrt(25+10*sqrt(5)) * (0.125**2)


# In[461]:


0.05676113500041882/0.06985014496763767


# In[462]:


(sqrt(25+10*sqrt(5))/4) / (3/2*sqrt(3))


# In[492]:


def radout(x,c):
    return np.max(np.sqrt(np.sum((x-c)**2,axis=1)))

def altitude(a,b,c):
    """
        Heron formula c is the base of a triangle and (a,b) are the lengthes of the other edges.
    """
    p=(a+b+c)/2.
    return 2.*np.sqrt(p*(p-a)*(p-b)*(p-c))/c

def radint(x,c):
    # transform x:(x1,x2,...,xn) into (x1,x2,...,xn,x1)
    a = np.vstack((x,x[0]))
    # ||x2-x1||,||x3-x2||,...,||xn-xn-1||,||x1-xn||  (n terms)
    edgeLength = np.sqrt(np.sum(np.diff(a,axis=0)**2,axis=1))
    # ||c-x1||,||c-x2||,..., ||c-xn||      (n terms)
    radiiLength= np.sqrt(np.sum((x-c)**2,axis=1))    
    #
    alt=[]
    for i in range(len(radiiLength)-1):
        alt.append(altitude(radiiLength[i],radiiLength[i+1],edgeLength[i]))
    return np.min(alt)

def process(df,Rin_norm=None,Rout_norm=None):
    df['c2']=df['vertices'].map(lambda x: np.mean(x,axis=0)).map(lambda x : x/np.sqrt(np.sum(x*x)))
    df['radint']=df[['vertices','c2']].apply(lambda x: radint(x.vertices,x.c2), axis=1)
    if Rin_norm is not None:
         df['radint'] =  df['radint']/Rin_norm
    df['radout']=df[['vertices','c2']].apply(lambda x: radout(x.vertices,x.c2), axis=1)
    if Rout_norm is not None:
        df['radout']=df['radout']/Rout_norm 
    df.drop('c2',axis=1)
    return df
    
def analyse(mypix):
    npix = 10*(mypix.n)**2 + 2
    Rin_norm = sqrt(2*pi/(sqrt(3)*npix))
    Rout_norm = sqrt(8*pi/(3*sqrt(3)*npix))
    print(f'npix={npix}, Rin_norm={Rin_norm}, Rout_norm={Rout_norm}')
    #hexagones
    print("Process hexagones")
    df=process(mypix.dfhexa.copy(),Rin_norm=Rin_norm,Rout_norm=Rout_norm)
    print('hexa : Rin\n',df['radint'].describe())
    print('hexa : Rout\n',df['radout'].describe())
    df.drop(['radint','radout'],axis=1)
    #pentagones
    print("Process pentagones")
    df=process(mypix.dfpenta.copy(),Rin_norm=Rin_norm,Rout_norm=Rout_norm)
    print('penta: Rin\n',df['radint'].describe())
    print('penta: Rout\n',df['radout'].describe())
    df.drop(['radint','radout'],axis=1)
    


# In[ ]:





# In[493]:


analyse(mypix)


# In[476]:


mypix=isocapix(100,fact=0)


# In[494]:


mypix100fact0 = mypix


# Test maintenant Goldberg modifiee mais d'abord Goldberg n=20 poiur reference 

# In[499]:


mypix20fact0 =isocapix(20,fact=0)


# In[500]:


analyse(mypix20fact0)


# In[501]:


mypix20fact0d1 =isocapix(20,fact=0.1)


# In[502]:


analyse(mypix20fact0d1)


# In[503]:


mypix20fact0d05 =isocapix(20,fact=0.05)


# In[504]:


analyse(mypix20fact0d05)


# In[505]:


mypix20fact0d025 =isocapix(20,fact=0.025)
analyse(mypix20fact0d025)


# In[507]:


get_ipython().run_cell_magic('time', '', 'mypix20fact0d04 =isocapix(20,fact=0.04)')


# In[509]:


get_ipython().run_cell_magic('time', '', 'analyse(mypix20fact0d04)')


# In[510]:


get_ipython().run_cell_magic('time', '', 'mypix100fact0d05 =isocapix(100,fact=0.05)')


# In[511]:


get_ipython().run_cell_magic('time', '', 'analyse(mypix100fact0d05)')


# In[515]:


mypix100fact0 = mypix20fact0


# In[516]:


newpix20fact0 = isocapix(20,fact=0)


# In[519]:


newpix5fact0 = isocapix(5,fact=0)


# In[654]:


plotpix(newpix5fact0,face=1,alpha=0.5)


# In[6]:


def plotpix(mypix,
              faces=None,
              plotcenter=False,
              plotidx=False,
              cols=['deepskyblue', 'yellow', 'lime','red'],
              linewidth=0.1,
              alpha=0.1
            ):

    """
    cols: 
     pentagones
     hexagones inside icosaedre face
     hexagones edges between 2 icosaedre faces
     hexagones edges between 2 icosaedre faces neighboors of pentagons
     ex. B&W setting    ['black', 'gray', 'silver','white']
        Colored setting ['deepskyblue', 'yellow', 'lime','red']
    """
    def extract(atype,idx,face):
        if (atype == 0) and not (face in idx): # penta
            return False
        elif (atype == 1) and (idx[0] != face): # hexa inside
            return False
        elif (atype > 1) and (idx[0] != face) and (idx[3] != face): #hexa edge
            return False
        else:
            return True
        return True

    df = mypix.dfAll
    if faces is not None:
        df = pd.DataFrame()
        for face in faces:
            df1 = mypix.dfAll[mypix.dfAll.apply(lambda x: extract(x.type,x.idx,face=face), axis=1)]
            df = pd.concat([df,df1], ignore_index=True)

    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel(r'$X$', fontsize=20)
    ax.set_ylabel(r'$Y$', fontsize=20)
    ax.set_zlabel(r'$Z$', fontsize=20)    
    
    for row in df.itertuples():
        
        atype  = row.type
        vertsf = row.vertices
        ax.add_collection3d(Poly3DCollection([vertsf], 
                                             facecolors = cols[atype], 
                                             edgecolors='k', 
                                             linewidths=linewidth, alpha=alpha))

        if plotcenter:
            xyzc = row.center
            ax.scatter(xyzc[0],xyzc[1],xyzc[2],marker='o',s=15,color='k')
        
        if (faces is not None) and plotidx:
            xyzc = row.center
            idx  = row.idx
            ax.text(xyzc[0],xyzc[1],xyzc[2],','.join(map(str, idx)), 
                                    size=7, zorder=10000, color='k')

    
    ax.set_xlim3d([-1,1])
    ax.set_ylim3d([-1,1])
    ax.set_zlim3d([-1,1])
    # Make panes transparent
    ax.xaxis.pane.fill = False # Left pane
    ax.yaxis.pane.fill = False # Right pane
    ax.zaxis.pane.fill = False
    # Remove grid lines
    ax.grid(False)


    plt.show()


# In[676]:


plotpix(newpix5fact0,faces=[0,10],alpha=0.5,plotidx=True)


# In[809]:


get_ipython().run_line_magic('timeit', 'mapGridpoint2Sogere([0,0.5,0.5],[1,0,0],[0,1,0],[0,0,1])')


# In[813]:


get_ipython().run_line_magic('timeit', 'slerp([1,0,0],[0,1,0],0.1)')


# In[796]:


get_ipython().run_line_magic('timeit', 'barycentricCoords([1,2])')


# In[801]:





# In[816]:


tmp = newpix5fact0.icoTriangs.copy()


# In[817]:


tmp


# In[819]:


tmp[10]


# In[821]:


tmp[10]=np.array([9,2,1])


# In[822]:


tmp


# In[4]:


mypix=isocapix(5,fact=0)


# In[7]:


plotpix(mypix)


# In[8]:


mypix.icoTriangs


# In[ ]:




