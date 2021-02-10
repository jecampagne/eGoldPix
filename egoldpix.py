#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

from math import sin,cos,acos,sqrt,pi, atan2

from scipy import stats

import pandas as pd
from iteration_utilities import flatten
get_ipython().run_line_magic('matplotlib', '')


# In[287]:


class egoldpix:
    def __init__(self,n=5,fact=0,tol=10**(-6)):
        '''
            n: order of the pixelization,
                # of pixels = 10*n^2+2 
                among them 12 are pentagon-like  and the rest are hexagon-like
            tol: a rounding factor to distinguish between 2 floats
            fact: to modify pentagon & neightboor hexagons shape
        '''
    
        self.n    = n
        self.tol  = tol
        self.fact = fact
        

        # number of isocaedron faces
        self.nIsofaces = 20

        #icosedron vertices
        self.icoPoints = self.getIcosaedreVertices()
        
        # All triplet vertices of the 20  equilateral triangles
        self.icoTriangs= self.getIcoTriangs(modif=True)
    
        #vertices for Face 0
        self.icoTriangs0  = self.icoTriangs[0]
        self.icoVertices0 = self.icoPoints[self.icoTriangs0]
    
    
        #center of the isocaedron faces
        self.icoTriangCenters = self.getIcoTriangCenters()
    
        # scaling factor   
        self.scale = 1/(n*sqrt(3))

        #offsets of the bottom left cornet of a icosahedron face for barycentric coordinate
        self.xoff7 = -1/2
        self.yoff7 = 0
        
        # 2 fois l'aire d'un des 20 triangles equilateraux 
        aire0 = sqrt(3)/2/sin(2*pi/5)**2
        self.aire0inv = 1/aire0
        
        #2*pi/3
        self.twopi3 = 2*pi/3

        #1/norder and related factors
        self.oneOvn = 1/self.n
        self.oneOv2n = 1/(2*self.n)
        self.sqrt3Ov2n = sqrt(3)/(2*self.n)
        
        #Get Face 0 -> Face I rotation matrix
        self.face0toIMtx = self.getFace0TofaceIMtx()

        #Get Face I -> Face 0 rotation matrix
        self.faceIto0Mtx = self.getFaceIToface0Mtx()
        
    #################################################################    
    ##
    ## For demo
    #################################################################    
    def buildFace0(self,nmax=10):
        '''
            For demo only, order of the pixelization should be low <=10
            build face 0
            order of the pixelization
        '''
        assert self.n<=nmax, "for demo only: order of pixelisation <= "+str(nmax)
        
        n=self.n
        
        faces   = []
        centers = []
        types   = []
        indexes = []

        for i in range(n+1):
            for j in range(n-i+1):
                opt, th = self.getHexInfos(i,j)
                #exclude the hexagons at the vertices of the isocele triangle
                if (i!=0 or j!=0) and (i!=0 or j!=n) and (i!=n or j!=0):
                    hexagcenter = self.getHexagoneCenterOnFace(i,j)
                    hexag = self.hexagon(hexagcenter[0],hexagcenter[1],th,opt)
                    a = self.icoTriangs0[0]
                    b = self.icoTriangs0[1] 
                    c = self.icoTriangs0[2] 
                    face   = self.getProjectedFace(hexag,a,b,c)
                    #ici on refait a la main getHexagoneCenterOnSphere
                    center = self.getProjectedPt(hexagcenter,a,b,c)

                    faces.append(face)
                    centers.append(center)
                    indexes.append((0,i,j))
                    types.append(opt)

        return faces, centers, indexes, types
    
    #################################################################    
    def plotFaceI(self, k=0, ax=None,nmax=10):
        '''
            For demo only, order of the pixelization should be low <=10
            
            plot the different tiles of a Icosahedron face
            k : index of Icosahedron face
            
            use Face 0 and rotations
        '''
        assert self.n<=nmax, "for demo only: order of pixelisation <= "+str(nmax)

        #Get tiles of Face 0
        faces0, centers0, indexes0, types0 = self.buildFace0()
        collectFaces0 = np.hstack(faces0)  #collect all tiles of face 0
        centers0 = np.array(centers0)


        #Get Face 0 -> Face k rotation matrix
        faceMtx = self.face0toIMtx

        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel(r'$X$', fontsize=20)
            ax.set_ylabel(r'$Y$', fontsize=20)
            ax.set_zlabel(r'$Z$', fontsize=20)  

        nIcofaces =  self.nIsofaces
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

        # change of frame
        # X' = -Y, Y'=-Z, Z'=X
        tmp = np.zeros_like(topPoints)
        for i in range(topPoints.shape[0]):
            tmp[i,0] = -topPoints[i,1]
            tmp[i,1] = -topPoints[i,2]
            tmp[i,2] =  topPoints[i,0]
        topPoints = tmp

        # bottom points of the icosahedron
        bottomPoints = np.zeros_like(topPoints)
        for i in range(bottomPoints.shape[0]):
            bottomPoints[i,0] = -topPoints[i,0]
            bottomPoints[i,1] =  topPoints[i,1]
            bottomPoints[i,2] = -topPoints[i,2]

        # icosahron vertices
        icoPoints=np.vstack((topPoints,bottomPoints))

        #return
        return icoPoints

    #################################################################
    def getIcoTriangs(self, modif=True):
        """
            20 faces : 3 vertex per face 
            12 vertex: each vertex is a number 0,...,11
            order : (0,1,2) for face 0 and then following some rotations
                we get the triplet for each face
        """
        nfaces =  self.nIsofaces
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
            icoTriangs = [(0,i+1,(i+1)%5+1) for i in range(5)] +                 [(6,i+7,(i+1)%5+7) for i in range(5)] +                 [(i+1,(i+1)%5+1,(7-i)%5+7) for i in range(5)] +                 [(i+1,(7-i)%5+7,(8-i)%5+7) for i in range(5)]
            icoTriangs=np.array(icoTriangs)

        return icoTriangs

    #################################################################
    def getIcoTriangCenters(self):
        """
            center of icosahedron faces projected on the sphere
        """
        #Get the location of the barycenters of the icosaedron faces
        nfaces=self.nIsofaces
        tmp = np.array([self.icoPoints[self.icoTriangs[k]] for k in range(nfaces)])
        icoTriangCenters = np.mean(tmp,axis=1,dtype=np.float64)
        # project on the unit-sphere
        norm=np.sqrt((icoTriangCenters*icoTriangCenters).sum(axis=1))

        return icoTriangCenters / norm[:,np.newaxis]

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

        x,y = p[0],p[1]
        # l3*sqrt(3)/2 = y
        l3 = y*2./sqrt(3.)
        # l1 + l2 + l3 = 1
        # 0.5*(l2 - l1) = x
        l2 = x + 0.5*(1 - l3)
        l1 = 1 - l2 - l3
        
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
            outputs the spherical linear interpolation of arc defined by p0, p1(around origin).  

            input: t=0 -> p0, t=1 -> p1. 
                    p0 and p1 are the vetors of form [x,y,z]

            output: interpolated coordinates.

            https://en.wikipedia.org/wiki/Slerp

        '''
###        assert abs(self.scalProd(p0,p0) - self.scalProd(p1,p1)) < self.tol
        ang0Cos = self.scalProd(p0,p1)/self.scalProd(p0,p0)
        ang0Sin = sqrt(1 - ang0Cos*ang0Cos)
        ang0 = atan2(ang0Sin,ang0Cos)
        l0 = sin((1-t)*ang0)
        l1 = sin(t    *ang0)
        tmp= np.array([(l0*p0[i] + l1*p1[i])/ang0Sin for i in range(len(p0))])
        
        return tmp
    
    #################################################################
    def mapGridpoint2Sphere(self,p,s1,s2,s3):
        '''
            inputs:
                - 'p' is the coordinate array of the planer verticies of the closed 
                    shape to be projected in the form [x,y,z]'.
                - 's1','s2' and 's3' are the vectors defining plane of the coordinates 
                    
            output: is the coordinate array of the projected face on the unit sphere.

            ex. mapGidpoint2Sphere([0,0.5,0.5],[1,0,0]',[0,1,0]',[0,0,1]')
        '''
        l1,l2,l3 = self.barycentricCoords(p)
        if abs(l3-1) < self.tol: return s3
        l2s = l2/(l1+l2)
        p12 = self.slerp(s1,s2,l2s)
        tmp = self.slerp(p12,s3,l3)
        
        return tmp

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
            face[:,i] = self.mapGridpoint2Sphere(hexag[:,i],
                                            self.icoPoints[u,:],
                                            self.icoPoints[v,:],
                                            self.icoPoints[w,:])
    
        return face
    
    #################################################################
    def getProjectedPt(self,p,u,v,w):
        """
        p: 2D point location
        """
        return self.mapGridpoint2Sphere(p,
                                    self.icoPoints[u,:],
                                    self.icoPoints[v,:],
                                    self.icoPoints[w,:])

    
    #################################################################
    def getHexInfos(self,i,j):
        """
            from (i,j) index of an hexagon get its oriantation and shape option
        """
        n = self.n 
        
        if i==0:
            th = -self.twopi3
            if j==1:
                opt = 3
            elif j==n-i-1:
                opt = 4
            else:
                opt = 2
        elif j==n-i: 
            th = self.twopi3
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

        return opt,th
    

    #################################################################
    def getHexagoneCenterOnFace(self,i,j):
        """
            On Face 0 of the icosahedron
            from (i,j) index of a hexagon gets its center on the Face
        """
        #        hexagcenter = np.array([self.xoff7+i*1/self.n+j*1/(2*self.n), 
        #                                self.yoff7+j*sqrt(3)/(2*self.n)])

        return np.array([self.xoff7 + i*self.oneOvn + j*self.oneOv2n, 
                         self.yoff7 + j*self.sqrt3Ov2n])

        
    #################################################################
    def getHexagoneCenterOnSphere(self,i,j):
        """
            On Face 0 of the icosahedron
            from (i,j) index of a hexagon gets its center projected On the sphere
        """
        
        hexagcenter = self.getHexagoneCenterOnFace(i,j)
        return self.getProjectedPt(hexagcenter,
                                   self.icoTriangs0[0],
                                   self.icoTriangs0[1],
                                   self.icoTriangs0[2])

    #################################################################
    def hexagon(self,x,y,th,opt):
        '''
            this program creates the hexagon of given configuration and size.
            inputs: 
                - x and y are the rectangular coordinates of the center of the hexagon
                - th is the rotation angle measured anticlockwise positive 
                    from positive x axis
                - opt: 1 full hexagon. 2 half hexagon, etc

                output: planar hexagon (complete/truncated) orientated 
            example:
                hexagon(0,0,np.pi/6,0.5,1)
        '''
        
        fact = self.fact
        
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
    
    #################################################################
    def rot(self, phi, a,b,c):
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

    #################################################################
    def getFace0TofaceIMtx(self):
        """
            rotation matrix from Face 0 to Face I of the icosahedron
        """
        
        nfaces = self.nIsofaces
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
        icoFaceMtx[0] = self.rot(0.,1.,0.,0.)    # identity

        # Faces of the upper cap : (vertex 0 is in common)
        # -----------------------
        # face "i" is obtained from face 0 by rotation +i*(2pi/5)
        # arround axis "0"    
        for i in [1,2,3,4]:
            icoFaceMtx[i] = self.rot(i*theta5, a0,b0,c0)

        # Faces of the middle: 
        # -----------------------
        # two types: 
        #    - type 1 obtained by rotation arround axe 0 of face 10
        #    - type 2 obtained by rotation arround axe 0 of face 15

        # Type 1
        # Face 10 is obtained face 0 by rotation -2pi/5 around axis 1
        icoFaceMtx[10] = self.rot(-theta5, a1,b1,c1)
        # Faces [11,12,13,14] obtained from face 10 by i=1,2,3,4 rotations of 2pi/5 
        # arround axe 0
        for i in [1,2,3,4]:
            rtmp = self.rot(i*theta5, a0,b0,c0)
            icoFaceMtx[10+i] = np.matmul(rtmp,icoFaceMtx[10]) 

        # Type 2
        # Face 15 is obtained face 0 by rotation -4pi/5 around axis 1
        icoFaceMtx[15] = self.rot(-2*theta5, a1,b1,c1)
        # Faces [16,17,18,19] obtained from face 15 by i=1,2,3,4 rotations of 2pi/5 
        # arround axe 0
        for i in [1,2,3,4]:
            rtmp = self.rot(i*theta5, a0,b0,c0)
            icoFaceMtx[15+i] = np.matmul(rtmp,icoFaceMtx[15]) 


        # Faces of the bottom cap (vertex 6 is in common)
        # -----------------------
        # first Face 5 is obtained from face 0 by rotation of 4pi/5 arround axis "4" 
        icoFaceMtx[5] = self.rot(2*theta5, a4,b4,c4)
        # then: faces [6,7,8,9] obtained from face 5 by i=1,2,3,4 rotations of +2pi/5 
        # arround axe 0
        for i in [1,2,3,4]:
            rtmp = self.rot(i*theta5, a0,b0,c0)
            icoFaceMtx[5+i] = np.matmul(rtmp,icoFaceMtx[5]) 


        #done
        return icoFaceMtx
    
    #################################################################
    def getFaceIToface0Mtx(self):
        """
        This is the inverse of getFace0TofaceIMtx but we do not use any inversion mtx routine
        """

        nfaces = self.nIsofaces
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
        icoFaceMtx[0] = self.rot(0.,1.,0.,0.)    # identity

        # Faces of the upper cap : (vertex 0 is in common)
        # -----------------------
        # face "i" is obtained from face 0 by rotation +i*(2pi/5)
        # arround axis "0"    
        for i in [1,2,3,4]:
            icoFaceMtx[i] = self.rot(-i*theta5, a0,b0,c0)

        # Faces of the middle: 
        # -----------------------
        # two types: 
        #    - type 1 obtained by rotation arround axe 0 of face 10
        #    - type 2 obtained by rotation arround axe 0 of face 15

        # Type 1
        # Face 10 is obtained face 0 by rotation -2pi/5 around axis 1
        icoFaceMtx[10] = self.rot(+theta5, a1,b1,c1)
        # Faces [11,12,13,14] obtained from face 10 by i=1,2,3,4 rotations of 2pi/5 
        # arround axe 0
        for i in [1,2,3,4]:
            rtmp = self.rot(-i*theta5, a0,b0,c0)
            icoFaceMtx[10+i] = np.matmul(icoFaceMtx[10],rtmp) 

        # Type 2
        # Face 15 is obtained face 0 by rotation -4pi/5 around axis 1
        icoFaceMtx[15] = self.rot(+2*theta5, a1,b1,c1)
        # Faces [16,17,18,19] obtained from face 15 by i=1,2,3,4 rotations of 2pi/5 
        # arround axe 0
        for i in [1,2,3,4]:
            rtmp = self.rot(-i*theta5, a0,b0,c0)
            icoFaceMtx[15+i] = np.matmul(icoFaceMtx[15],rtmp) 


        # Faces of the bottom cap (vertex 6 is in common)
        # -----------------------
        # first Face 5 is obtained from face 0 by rotation of 4pi/5 arround axis "4" 
        icoFaceMtx[5] = self.rot(-2*theta5, a4,b4,c4)
        # then: faces [6,7,8,9] obtained from face 5 by i=1,2,3,4 rotations of +2pi/5 
        # arround axe 0
        for i in [1,2,3,4]:
            rtmp = self.rot(-i*theta5, a0,b0,c0)
            icoFaceMtx[5+i] = np.matmul(icoFaceMtx[5],rtmp) 


        #done
        return icoFaceMtx
    
    #################################################################
    def pt2FaceId(self,pt):
        """ 
         find the face of a set of N points
            use: scalaire product 
              pt = 3xN  : (x,y,z) x N
              (20,3) . (3,N) = (20,N)  
              argmax axis 0 => N values
        """
        # test the 20 distnaces from current pt to the 20 face centers
        return np.argmax(np.einsum('jk,kl->jl',self.icoTriangCenters,pt),axis=0)

    #################################################################
    def angle2FaceId(self,thetaPhi):
        """ 
            find the face of a set of N points
            thetaPhi = Nx2  :  N x (theta, phi)
            use: scalaire product 
              pt = 3xN  : (x,y,z) x N
              (20,3) . (3,N) = (20,N)  
              argmax axis 0 => N values
        """
        #current pt
        theta = thetaPhi[:,0]
        phi   = thetaPhi[:,1]
        pt = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        return self.pt2FaceId(pt)
        
    
    #################################################################    
    def getBarycentricCoord(self,pt,faceId):
        """
            Input:
                pt (x,y,z) pt should be already ON the face of the icosahedron (ie not On sphere)
                faceId : index of face
            Output:
                barycentric coordinates

            Todo: pe que c'est deja vectorizable 
            au lieu de return a0[0],a1[0]
            [(a,b) in zip(a0,a1)] ?? a suivre
        """
        vertices = self.icoPoints[self.icoTriangs[faceId]] 
        #
        vec0 = vertices[0][np.newaxis,:]-pt
        vec1 = vertices[1][np.newaxis,:]-pt
        vec2 = vertices[2][np.newaxis,:]-pt
        #retreive barycentric coordinate
        a0 = np.cross(vec1,vec2)
        a0 = np.sqrt((a0*a0).sum(axis=1))*self.aire0inv
        a1 = np.cross(vec2,vec0)
        a1 = np.sqrt((a1*a1).sum(axis=1))*self.aire0inv
        return a0[0],a1[0]

    #################################################################
    def getBarycentricCoordExtension(self,pt,faceId):
        """
            Input:
                pt (x,y,z) On the sphere
                faceId : index of face
            Output:
                barycentric coordinates
    
            use of getBarycentricCoord for pt on the icosahedron planar equilateral triangle
        """
        vertices = self.icoPoints[self.icoTriangs[faceId]] 

        
        
        u = vertices[1][np.newaxis,:]-vertices[0][np.newaxis,:]
        
        print("getBarycentri pt shape: ",pt.shape)
        print("getBarycentri  u shape: ",u.shape)
        
        
        v = vertices[2][np.newaxis,:]-vertices[0][np.newaxis,:]
        w = pt - vertices[0][np.newaxis,:]
        # S=[u,v,OPshere]
        smtx = np.vstack((u,v,pt)).T
        null = np.zeros_like(u)
        mtx1 = np.vstack((u,v,null)).T
        smtxinv = np.linalg.inv(smtx)
        mtx2 = mtx1 @ smtxinv
        #OPproj on planar traingle = OV0 + S diag(1,1,0) S^-1 V0Ps
        pproj = np.dot(mtx2,w.T).T+vertices[0][np.newaxis,:]
        return self.getBarycentricCoord(pproj,faceId)
    
    
    #################################################################    
    def findNeightboorsHexagCenter(self,a,b,full=False):
        """
         implicitly on the Face 0 of the icosahedron
         (a,b) : barycentric coordinates of the point

            ignoring edge-effect the 6 neightboors arround (i,j) hexagons are
                 (i,j-1) (i-1,j)
            (i+1,j-1) (i,j) (i-1,j+1)
                  (i+1,j) (i,j+1)            

            use full=True to debug if necessary
            
            Todo: voir si on peut eviter la loop (ic,jc) surtout si on inclus les coins
                par vectorisation
            
        """
        c = 1-a-b        # the third barycentric coordinate
        bscaled = self.n *b
        cscaled = self.n *c

        # (i,j) index target    
        i = int(round(bscaled))
        j = int(round(cscaled))

        #Choose indexes of the hexagones to test for closest approach
        if full:
            indexes = [(i,j),(i,j-1),(i-1,j),(i-1,j+1),(i,j+1),(i+1,j),(i+1,j-1)]
        else:
            if bscaled >= i:
                if cscaled >= j:
                    indexes = [(i,j),(i,j+1),(i+1,j)]
                else:
                    indexes = [(i,j),(i,j-1),(i+1,j),(i+1,j-1)]
            else:
                if cscaled >= j:
                    indexes = [(i,j),(i-1,j),(i-1,j+1),(i,j+1)]
                else:
                    indexes = [(i,j),(i,j-1),(i-1,j)]

        centers = []
        
        for (ic,jc) in indexes:
            ## On peut inclure les coins... en commentant les 2 lignes ci-dessous
            ##        if (ic==0 and jc==0) or (ic==0 and jc==n) or (ic==n and jc==0):
            ##            continue

            center = self.getHexagoneCenterOnSphere(ic,jc)
            #save
            centers.append(center)
            
            
        return np.array(centers), np.array(indexes)

    #################################################################    
    def findClosest(self,pt,pts):
        """
            find the closest point to "p" in the list "pts"
            p: (x,y,z)
            pts : a list of (xi,yi,zi)
        """
        
        #print("findClosest, pt  shape ",pt.shape) #3,1
        #print("findClosest, pts shape ",pts.shape) #3,3
        
        return np.argmax(np.einsum('jk,kl->jl',pts,pt),axis=0)
    
    
    #################################################################    
    def pt2pix(self,pt):
        """
            test pt -> pixel identifier 
            input : pt (x,y,z)
            output: faceId, (i,j)-face0
        """
        # determine the face Id
        iFace = self.pt2FaceId(pt)[0]  # Todo  vectorization
        # rotate the point to face 0
        pt0 = np.dot(self.faceIto0Mtx[iFace],pt)
        assert self.pt2FaceId(pt0)[0] == 0, "pt2pix rotation Face I->0 pb"
        #get barycentric coord (a,b) of pt0 
        # print("pt0 shape: ",pt0.shape) # (3,1)
        a,b = self.getBarycentricCoordExtension(pt0.T,faceId=0)
        #centres of target tiles 
        centernbs,indexes = self.findNeightboorsHexagCenter(a,b)
        #find the closest one 
        iloc = self.findClosest(pt0,centernbs)[0]  # Todo vectorization
        
        #the closest tile index
        idxClosest    = indexes[iloc]

        # juste pour le debug ici car on ne veut que l'index
        #        centerClosest = centernbs[iloc]         
        #        print("pt0: ",pt0)
        #        print("centerClosest: ",centerClosest)
        #        print("idxClosest: ",idxClosest)
        
        return iFace, idxClosest
    
    #################################################################    
    def pix2pt(self,iFace,ijdx):
        """
          from face index and (i,j)-index on face 0 retreive the tile center
          input: 
              iFace: on of the 20 faces of the icosahedron
              ijdx: (i,j)-index of the tile on face 0 
        """
        #find center of the tile on Face 0 
        center0 = self.getHexagoneCenterOnSphere(ijdx[0],ijdx[1])
        #Rotate from Face0 to Face faceId
        center = np.dot(self.face0toIMtx[iFace],center0)
        return center

    #################################################################    
    def pix2TileVertices(self,iFace,ijdx):
        """
          from face index and (i,j)-index on face 0 retreive the tile vertices
          input: 
              iFace: on of the 20 faces of the icosahedron
              ijdx: (i,j)-index of the tile on face 0 
        """
        #Get the tile vertices on Face 0
        i = ijdx[0]
        j = ijdx[1]
        #get shape option and orientation of the tile
        opt, th = self.getHexInfos(i,j)
        #coordinates of the tile center on Face 0 frame
        hexagcenter = self.getHexagoneCenterOnFace(i,j)
        #the vertices coordinates on Face 0 frame
        verticesOnFace  = self.hexagon(hexagcenter[0],hexagcenter[1],th,opt)
        #the vertices coordinates projected on Sphere (Face 0)
        verticesOnSphere= self.getProjectedFace(verticesOnFace,
                                            self.icoTriangs0[0],
                                            self.icoTriangs0[1],
                                            self.icoTriangs0[2])
        print("Mtx: shape ",self.face0toIMtx[iFace].shape)
        print("verticesOnSphere: shape ",verticesOnSphere.shape)
            


# In[289]:


mypix = egoldpix(n=10)


# In[290]:


#mypix.plotFaceI()


# In[291]:


# theta, phi angles of the 20 center of faces
icoTriangCenters = mypix.icoTriangCenters


# In[292]:


#icoTriangCenters.shape


# In[293]:


#mypix.pt2FaceId(icoTriangCenters.T)


# In[294]:


#mypix.pt2pix(icoTriangCenters.T)


# In[295]:


tmppt = icoTriangCenters[2].reshape(3,1)


# In[296]:


#tmppt = np.array([-0.62321876, -0.03638397,  0.78120073]).reshape(3,1)


# In[297]:


iFace, ijdx = mypix.pt2pix(tmppt)


# In[298]:


mypix.pix2pt(iFace, ijdx)


# In[299]:


mypix.pix2TileVertices(iFace, ijdx)


# # Avec Theta,Phi

# In[ ]:


#        theta = thetaPhi[:,0]
#        phi   = thetaPhi[:,1]
#        pt = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])


# In[25]:


def pt3DtoSpherical(xyz):
    norm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    theta = np.arccos(xyz[:,2]/norm)
    phi = np.arctan2(xyz[:,1], xyz[:,0])
    mask = phi<0
    phi[mask]+=2*pi
    return np.array([theta,phi]).T


# In[26]:


# theta, phi angles of the 20 center of faces
icoTriangCenters = mypix.icoTriangCenters
angleicoTriangCenters= pt3DtoSpherical(icoTriangCenters)


# In[27]:


angleicoTriangCenters


# In[28]:


mypix.angle2FaceId(angleicoTriangCenters)


# In[72]:


a=np.array([1]).squeeze()
print(mypix.faceIto0Mtx[a].shape)


# In[ ]:




