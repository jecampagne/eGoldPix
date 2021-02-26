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


# In[239]:


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
        
        #Get Face neighbors array to solve tile index ambiguity
        self.faceNeighbors = self.getFaceNeighbors()
        
        #Get pendagon DataFrame
        self.pentaDF = self.makePentagon()
        
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
        icoTriangs = np.zeros((self.nIsofaces,3),dtype=int)
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

        return icoTriangs
    #################################################################
    def getFaceNeighbors(self):
        """
            Set icosahedron face neighbors: bottom, left, right
            according to face orientation given by getIcoTriangs 
        """
        nbarr = np.zeros((self.nIsofaces,3),dtype=np.int)
        #up cap
        nbarr[0] = [10,4,1]
        nbarr[1] = [11,0,2]
        nbarr[2] = [12,1,3]
        nbarr[3] = [13,2,4]
        nbarr[4] = [14,3,5]
        #bottom cap
        nbarr[5] = [9,17,6]
        nbarr[6] = [5,18,7]
        nbarr[7] = [6,19,8]
        nbarr[8] = [7,15,9]
        nbarr[9] = [8,16,5]
        # v isocele triangle central band
        nbarr[10] = [15,0,16]
        nbarr[11] = [16,1,17]
        nbarr[12] = [17,2,18]
        nbarr[13] = [18,3,19]
        nbarr[14] = [19,4,15]
        # ^ isocele triangle central band
        nbarr[15] = [14,10,8]
        nbarr[16] = [10,11,9]
        nbarr[17] = [11,12,5]
        nbarr[18] = [12,13,6]
        nbarr[19] = [13,14,7]
        return nbarr

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
            
            Fev2021: it is used also for the pentagons
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
            Version for drawning only hexagon of given configuration and size.
            see hexagonV2
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
        
        print("findNeightboorsHexagCenter; i,j target: ",i,j)

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
        print("findNeightboorsHexagCenter; indexes: ",indexes)
        for (ic,jc) in indexes:
            if ic+jc > self.n: continue
            
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
            output: faceId, (i,j)-face0 coding
        """
        # determine the face Id
        iFace = self.pt2FaceId(pt)[0]
        # rotate the point to face 0
        pt0 = np.dot(self.faceIto0Mtx[iFace],pt)
        #####Pas necessaire et peut produire une exception a cause d'arrondis
        ######assert self.pt2FaceId(pt0)[0] == 0, "pt2pix rotation Face I->0 pb"+str(self.pt2FaceId(pt0)[0])
        #get barycentric coord (a,b) of pt0 
        # print("pt0 shape: ",pt0.shape) # (3,1)
        a,b = self.getBarycentricCoordExtension(pt0.T,faceId=0)
        #centres of target tiles 
        centernbs,indexes = self.findNeightboorsHexagCenter(a,b)
        #find the closest one 
        iloc = self.findClosest(pt0,centernbs)[0]
        
        #the closest tile index
        idxClosest    = indexes[iloc]

        # juste pour le debug ici car on ne veut que l'index
        #        centerClosest = centernbs[iloc]         
        #        print("pt0: ",pt0)
        #        print("centerClosest: ",centerClosest)
        #        print("idxClosest: ",idxClosest)
        
        #return    
        return self.codeTileIndex(iFace, idxClosest)
    
    #################################################################    
    def codeTileIndex(self, iFace, ijdx):
        """
            according to i,j-index choose to code
            o [[face, i,j]] as default
            o [[face1,i1,j1],[face2,i2,j2]] for tiles at edges between two icosahedron faces
            o for pentagon NOT YET IMPLEMENTED
        """
        i0 = ijdx[0]
        j0 = ijdx[1]

        tileIdx = [[iFace, i0, j0]] #default
        
        #pentagons fist: top, bottom left, bottom right
        if (i0==0 and j0==0) or (i0==self.n and j0==0) or (i0==0 and j0==self.n):
            a = self.pentaDF.loc[self.pentaDF['idx']                .map(lambda x: next((i for i,v in enumerate(x) if v == tileIdx[0]),-1)) != -1]
            if a.empty:
                    print("codeTileIndex: penta bug: iface,i,j",tileIdx[0])
            tileIdx = a.iloc[0]['idx']
                
        elif j0==0 and (i0 != 0 or i0 != self.n):
            #tile edge between Face I and Left Face
            # index correspondance (Face,i0,0) <-> (Left-Face,0,i0)
            # like face 4 =Left-Face for face 0
            i4 = 0
            j4 = i0
            iFaceLeft = self.faceNeighbors[iFace][1]
            #order the two triplets according to the face Id
            if  iFace < iFaceLeft:
                tileIdx.append([iFaceLeft, i4, j4])
            else:
                tileIdx.insert(0,[iFaceLeft, i4, j4])

        elif i0==0 and j0 != self.n:
            #tile edge between Face I and Right Face
            # index correspondance (Face,0,j0) <-> (Right-Face,j0,0)
            # like face 1 =Right-Face for face 0
            i1 = j0
            j1 = 0
            iFaceRight = self.faceNeighbors[iFace][2]
            if iFace < iFaceRight:
                tileIdx.append([iFaceRight, i1, j1])
            else:
                tileIdx.insert(0,[iFaceRight, i1, j1])

        elif i0+j0==self.n:
            #tile edge between Face I and Bottom Face 
            # index correspondance (Face,i0,j0) avec i0+j0=n <-> (Bottom-Face,i0,0)
            # like face 10 =Bottom-Face for face 0
            i10 = i0
            j10 = 0
            iFaceBottom = self.faceNeighbors[iFace][0]
            if iFace < iFaceBottom:
                tileIdx.append([iFaceBottom, i10, j10])
            else:
                tileIdx.insert(0,[iFaceBottom, i10, j10])

        return tileIdx

    #################################################################    
    def decodeTileIndex(self,tileIdx):
        """
            according to codeTileIndex convention, extract the first triplet
            o ok for non-edge hexagonal tiles
            o ok for edge hexagonal tiles
            o for pentagon NOT YET IMPLEMENTED
        """
        # for non-edge hexagonal tiles & for edge hexagonal tiles take the first triplet
        
        return tileIdx[0]
    #################################################################    
    def pix2pt(self,tileIdx):
        """
            from tile index  retreive the tile center
        """
        iFace, i,j = self.decodeTileIndex(tileIdx)
        #get location on the sphere as if it is a Face 0 tile   
        center0 = self.getHexagoneCenterOnSphere(i,j)
        # Rotate from Face0 to Face iFace
        center = np.dot(self.face0toIMtx[iFace],center0)
        return center

    #################################################################    
    def pix2pt_DEPRECATED(self,iFace,ijdx):
        """
          from face index and (i,j)-index on face 0 retreive the tile center
          input: 
              iFace: on of the 20 faces of the icosahedron
              ijdx: (i,j)-index of the tile on face 0 
        """
        print("DEPRECATED NO MORE IN USE")
        #find center of the tile on Face 0 
        center0 = self.getHexagoneCenterOnSphere(ijdx[0],ijdx[1])
        #Rotate from Face0 to Face iFace
        center = np.dot(self.face0toIMtx[iFace],center0)
        return center

    #################################################################
    def hexagonV2(self,x,y,th,opt):
        '''
            Modified version of hexagon method but particularized for Face 0
            
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
                 mais on supprime les points 0 et 4 dont y=0
            ''' 
            hex = np.zeros((2,3))
            hex[0,:]= np.array([sqrt(3)/2,0,-sqrt(3)/2]) # X-ccod
            hex[1,:]= np.array([0.5,1,0.5]) # Y-coord

        elif opt == 3:
            # point 0 et 1 sont modifiers par rapport au type 2
            hex = np.zeros((2,5))
            hex[0,:]= np.array([sqrt(3)/2-fact,0,-sqrt(3)/2]) # X-ccod
            hex[1,:]= np.array([1/2+fact/sqrt(3),1,0.5]) # Y-coord

        elif opt == 4:
            # point 3 et 4 sont modifiers par rapport au type 2
            hex = np.zeros((2,5))
            hex[0,:]= np.array([sqrt(3)/2,0,-sqrt(3)/2+fact]) # X-ccod
            hex[1,:]= np.array([0.5,1,1/2+fact/sqrt(3)]) # Y-coord


        hex = np.matmul(rot_mat,hex)

        hex[0,:]= x+hex[0,:] 
        hex[1,:]= y+hex[1,:]

        return hex

    #################################################################    
    def xycent_DEPRECATED(self,xoff7,yoff7,i,j,n):
        '''
            2D localisation of the center of a pentagon in the frame of a icosaedre face
            DEPRECATED: use getHexagoneCenterOnFace(self,i,j)
        '''
        return xoff7+i/n+j/(2*n), yoff7+j*sqrt(3)/(2*n)
    #################################################################    
    def makePentagon(self):
        """
         There are 12 only pentagons so we can be more explicit 
         Orientation 'th' coherent with getHexInfo
         Face-vertices relation from getIcoTriangs
         the (i,j,th) triplet follow type 4 hexagons numbering/orientation 
         'ij' gives for each face the location (i,j) corresponding the position in getIcoTriangs
              position 0 => (0,0)
              position 1 => (n,0)
              position 2 => (0,n)
        """
        
        pentaBuild=pd.DataFrame(columns=['idx','ij','xyc','th'])
        
        #below idx0 is a tuple with the icosaedre face number 
        #  ===> the order in the tuple idx0 is important
        #we build a DF of the vertices of each pentagon positionned in local 2D icosadre-face frame
        
        xoff7 = self.xoff7
        yoff7 = self.yoff7
        n     = self.n
                        
        #Penta #0 : top
        
        # ------ nb idx0 should be a tuple for hastable
        idx0 = (0,1,2,3,4)
        for k in idx0:
            info = {
                'idx':idx0,
                'ij' : (0,0),
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th':0
            }
            pentaBuild=pentaBuild.append(info,ignore_index=True)

        
        ######
        #Pentas of the upper ring
        ######
        #Penta #1 : 
        idx0 = (4,0,10,15,14)
        infos=[]

        infos.append({
                'idx':idx0,
                'ij' : (0,n),
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th':-self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th':self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (n,0),                            
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })

        
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)


        #Penta #2 :
        idx0 = (0,1,11,16,10)
        infos=[]
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th':-self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th':self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })

        for info in infos:
            pentaBuild=pentaBuild.append(info,ignore_index=True)
        
            
            
        #Penta #3
        idx0 = (1,2,12,17,11)
        infos=[]
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th':-self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th':self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })

        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)
                    
            
        #Penta #4 :
        idx0 = (2,3,13,18,12)
        infos=[]
        
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th':-self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th':self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })

        
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)


        
        #Penta #5 :
        idx0 = (3,4,14,19,13)
        infos=[]

        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th':-self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th':self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })

        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })

        
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)
            

        ######
        #Pentas of the lower ring
        ######

        #Penta #7 :
        idx0 = (5,6,18,12,17)
        infos=[]
        
        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })
        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)

            
        #Penta #8:
        idx0 = (9,5,17,11,16)
        infos=[]

        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })
        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)


        #Penta #9
        idx0 = (8,9,16,10,15)
        infos=[]

        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })
        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })
        
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)
            

        
        #Penta #10
        idx0 = (7,8,15,14,19)
        infos=[]

        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })
        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })

        
        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)
        
 
        #Penta #11
        idx0 = (6,7,19,13,18)

        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })
        infos.append({
                'idx':idx0,
                'ij' : (n,0),                
                'xyc':self.getHexagoneCenterOnFace(n-1,1),
                'th': self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            })
        infos.append({
                'idx':idx0,
                'ij' : (0,0),                
                'xyc':self.getHexagoneCenterOnFace(1,0),
                'th': 0
            })

        for info in infos:
            pentaBuild = pentaBuild.append(info,ignore_index=True)

       
        #Penta #6  : bottom
        idx0 = (5,6,7,8,9)
        for k in idx0:
            info = {
                'idx':idx0,
                'ij' : (0,n),                
                'xyc':self.getHexagoneCenterOnFace(0,n-1),
                'th': -self.twopi3
            }
            pentaBuild=pentaBuild.append(info,ignore_index=True)

                            
        #We group by idx and then get (x,y,th) triplet list
        pentaBuild=pentaBuild.groupby('idx',as_index=False).agg(lambda x: x.tolist())
        pentaBuild['info']=[[(*a, b) for a, b in zip(x, y)] for x, y in zip(pentaBuild['xyc'],pentaBuild['th'])]
    
        pentaBuild=pentaBuild.drop(['xyc','th'],axis=1)

                
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
        
        # Rebuild indexation (face, i,j)
        pentaBuild['newIdx']=[[(a, *b) for a, b in zip(x, y)] for x, y in zip(pentaBuild['idx'],pentaBuild['ij'])]
        pentaBuild=pentaBuild.drop('idx',axis=1).drop('ij',axis=1).rename(columns={'newIdx':'idx'})
        #transform tuple into lis
        pentaBuild['idx']= pentaBuild['idx'].map(lambda x:[list(l) for l in x])
        pentaBuild['vertices'] = pentaBuild['vertices'].map(lambda x:[list(l) for l in x])
        pentaBuild=pentaBuild[pentaBuild.columns[[2,0,1]]]
        
        #To uniformize with the DF of the hexagons
        ###pentaBuild['type']=0
        #pentaBuild=pentaBuild[pentaBuild.columns[[0,1,2]]]
                
        # Got it!
        return pentaBuild

    #################################################################    
    def pix2TileVertices(self,tileIdx):
        """
          from face index and (i,j)-index on face 0 retreive the tile vertices
          input: 
              tileIdx : see codeTileIndex
                  iFace: on of the 20 faces of the icosahedron
                  (i0,j0)-index of the tile on face 0 
        """
        #Get the tile vertices on Face 0
        iFace, i0,j0 = self.decodeTileIndex(tileIdx)
        #get shape option and orientation of the tile
        opt, th = self.getHexInfos(i0,j0)
        #coordinates of the tile center on Face 0 frame
        hexagcenter = self.getHexagoneCenterOnFace(i0,j0)
        verticesOnFace  = self.hexagonV2(hexagcenter[0],hexagcenter[1],th,opt)
        #the vertices coordinates projected on Sphere (Face 0)
        verticesOnSphere0= self.getProjectedFace(verticesOnFace,
                                            self.icoTriangs0[0],
                                            self.icoTriangs0[1],
                                            self.icoTriangs0[2])
        
        if i0==0 and j0==0:
            #top pentagon
            print("top penta not yet implemented")

        elif i0==self.n and j0==0:
            #bottom left
            print("bottom left penta not yet implemented")

        elif i0==0 and j0==self.n:
            #bottom right
            print("bottom right penta not yet implemented")
            
        elif j0==0 and (i0 != 0 or i0 != self.n):
            #tile edge between Face 0 and Face 4
            # index correspondance (0,i0,0) <-> (4,0,i0)
            i4 = 0
            j4 = i0
            opt4, th4 = self.getHexInfos(i4,j4)
            hexagcenter4 = self.getHexagoneCenterOnFace(i4,j4)
            verticesOnFace4 = self.hexagonV2(hexagcenter4[0],hexagcenter4[1],th4,opt4)            
            verticesOnSphere4= self.getProjectedFace(verticesOnFace4,
                                            self.icoTriangs[4][0],
                                            self.icoTriangs[4][1],
                                            self.icoTriangs[4][2])

            #merge vertices on Face 0 with vertices on Face 4
            verticesOnSphere0 = np.hstack((verticesOnSphere0,verticesOnSphere4))

        elif i0==0 and j0 != self.n:
            #tile edge between Face 0 and Face 1
            # index correspondance (0,0,j0) <-> (1,j0,0)
            i1 = j0
            j1 = 0
            opt1, th1 = self.getHexInfos(i1,j1)
            hexagcenter1 = self.getHexagoneCenterOnFace(i1,j1)
            verticesOnFace1 = self.hexagonV2(hexagcenter1[0],hexagcenter1[1],th1,opt1)            
            verticesOnSphere1= self.getProjectedFace(verticesOnFace1,
                                            self.icoTriangs[1][0],
                                            self.icoTriangs[1][1],
                                            self.icoTriangs[1][2])

            #merge vertices on Face 0 with vertices on Face 1
            verticesOnSphere0 = np.hstack((verticesOnSphere0,verticesOnSphere1))
   
        elif i0+j0==self.n:
            #tile edge between Face 0 and Face 10
            # index correspondance (0,i0,j0) avec i0+j0=n <-> (10,i0,0)
            i10 = i0
            j10 = 0
            opt10, th10 = self.getHexInfos(i10,j10)
            hexagcenter10 = self.getHexagoneCenterOnFace(i10,j10)
            verticesOnFace10 = self.hexagonV2(hexagcenter10[0],hexagcenter10[1],th10,opt10)            
            verticesOnSphere10= self.getProjectedFace(verticesOnFace10,
                                            self.icoTriangs[10][0],
                                            self.icoTriangs[10][1],
                                            self.icoTriangs[10][2])

            #merge vertices on Face 0 with vertices on Face 1
            verticesOnSphere0 = np.hstack((verticesOnSphere0,verticesOnSphere10))
   
        
        
        #Rotate from Face 0 to Face iFace
        verticesOnSphere = np.dot(self.face0toIMtx[iFace],verticesOnSphere0)
        
        return verticesOnSphere


# In[240]:


mypix = egoldpix(n=6)


# In[241]:


mypix.pentaDF


# In[174]:


# theta, phi angles of the 20 center of faces
icoTriangCenters = mypix.icoTriangCenters


# In[175]:


#test face 12 center 
tmppt = icoTriangCenters[12].reshape(3,1)


# In[724]:


#test face 6 center 
#tmppt = icoTriangCenters[6].reshape(3,1)


# In[725]:


#point entre face 0 et 4 du cote 0
#tmppt1 = icoTriangCenters[0].reshape(3,1)
#tmppt2 = icoTriangCenters[4].reshape(3,1)
#tmppt = 0.5*(tmppt1+tmppt2)
#tmppt = tmppt/np.sqrt(np.sum(tmppt*tmppt))
#mtx = mypix.rot((2.*pi/5.)/30, 0,0,1)
#tmppt = np.dot(mtx,tmppt)


# In[726]:


#point entre face 0 et 1 du cote 0
#tmppt1 = icoTriangCenters[0].reshape(3,1)
#tmppt2 = icoTriangCenters[1].reshape(3,1)
#tmppt = 0.5*(tmppt1+tmppt2)
#tmppt = tmppt/np.sqrt(np.sum(tmppt*tmppt))
#mtx = mypix.rot((-2.*pi/5.)/30, 0,0,1)
#tmppt = np.dot(mtx,tmppt)


# In[727]:


#point entre face 0 et 10 du cote 0
#tmppt1 = icoTriangCenters[0].reshape(3,1)
#tmppt2 = icoTriangCenters[10].reshape(3,1)
#tmppt = 0.5*(tmppt1+tmppt2)
#tmppt = tmppt/np.sqrt(np.sum(tmppt*tmppt))
#goldphi = (1.+sqrt(5.))/2.  # golden ratio (x^2=x+1)
#goldphi2 = goldphi * goldphi 
#c1 = goldphi/(1.+goldphi2)
#a1 = 2.*c1
#b1 = 0.
#mtx = mypix.rot((2.*pi/5.)/30, a1,b1,c1)
#tmppt = np.dot(mtx,tmppt)


# In[176]:


#point entre face 10 et 15
tmppt1 = icoTriangCenters[10].reshape(3,1)
tmppt2 = icoTriangCenters[15].reshape(3,1)
tmppt = 0.5*(tmppt1+tmppt2)
tmppt = tmppt/np.sqrt(np.sum(tmppt*tmppt))


# In[729]:


#point entre face 18 et 6
#tmppt1 = icoTriangCenters[18].reshape(3,1)
#tmppt2 = icoTriangCenters[6].reshape(3,1)
#tmppt = 0.5*(tmppt1+tmppt2)
#tmppt = tmppt/np.sqrt(np.sum(tmppt*tmppt))


# In[780]:


#mypix.icoPoints[mypix.icoTriangs[0]]


# In[781]:


#tmppt = mypix.icoVertices0[2].reshape(3,1)


# In[782]:


#tmppt


# In[783]:


#mypix.icoPoints[mypix.icoTriangs[11]]


# In[177]:


tmppt


# In[178]:


pixId = mypix.pt2pix(tmppt)


# In[179]:


pixId


# In[180]:


center = mypix.pix2pt(pixId)


# In[181]:


center


# In[182]:


vertices = mypix.pix2TileVertices(pixId)


# In[183]:


vertices


# In[ ]:





# In[184]:


zoom=False
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)  

colors = cm.rainbow(np.linspace(0, 1, mypix.nIsofaces))

ax.scatter(tmppt[0],tmppt[1],tmppt[2],marker='o',s=10,color='blue')


ax.scatter(center[0],center[1],center[2],marker='x',s=10,color='k')
xf,yf,zf = vertices[0,:],vertices[1,:],vertices[2,:]
ax.add_collection3d(Poly3DCollection([list(zip(xf,yf,zf))], 
                    facecolors = colors[iFace], 
                    edgecolors='k', 
                    linewidths=1, alpha=0.5))

ax.text(center[0]*1.01,
        center[1]*1.01,
        center[2]*1.01,"{}".format('/'.join([str(x) for x in mypix.decodeTileIndex(pixId)])),
        size=10, zorder=1, color='k')

ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)
if zoom:
    ax.set_xlim3d([np.min(vertices[0,:]),np.max(vertices[0,:])])
    ax.set_ylim3d([np.min(vertices[1,:]),np.max(vertices[1,:])])
    ax.set_zlim3d([np.min(vertices[2,:]),np.max(vertices[2,:])])
else:
    ax.set_xlim3d([-1,1])
    ax.set_ylim3d([-1,1])
    ax.set_zlim3d([-1,1])
    
plt.show()


# In[185]:


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20) 
mypix.plotFaceI(k=10,ax=ax)
#mypix.plotFaceI(k=4)
#mypix.plotFaceI(k=1)
mypix.plotFaceI(k=15,ax=ax)
ax.set_xlim3d([-1,1])
ax.set_ylim3d([-1,1])
ax.set_zlim3d([-1,1])
plt.show()


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


# In[759]:


atmp = np.zeros((3,2),dtype=np.int)


# In[760]:


atmp.shape


# In[761]:


atmp[0] = [1,2]


# In[762]:


atmp


# # Penta

# In[186]:


mypix.pentaDF.iloc[0].vertices


# In[187]:


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)  
colors = cm.rainbow(np.linspace(0, 1, mypix.nIsofaces))


for row in mypix.pentaDF.itertuples():
        
    vertsf = row.vertices
    ax.add_collection3d(Poly3DCollection([vertsf], 
                                             facecolors = 'deepskyblue', 
                                             edgecolors='k', 
                                             linewidths=1, alpha=0.5))

    xyzc = row.center
    ax.scatter(xyzc[0],xyzc[1],xyzc[2],marker='o',s=15,color='k')



ax.set_xlabel(r'$X$', fontsize=20)
ax.set_ylabel(r'$Y$', fontsize=20)
ax.set_zlabel(r'$Z$', fontsize=20)
ax.set_xlim3d([-1,1])
ax.set_ylim3d([-1,1])
ax.set_zlim3d([-1,1])
    
plt.show()


# In[242]:


# Les sommets de l'icosaedre sur la sphere
icoPoints = mypix.getIcosaedreVertices()


# In[243]:


# sommet #2 + extra => remis sur la sphere
tmppt = icoPoints[2] + [0.01, -0.01,-0.01]
tmppt = tmppt/np.sqrt(np.sum(tmppt*tmppt))
tmppt = tmppt.reshape(3,1)


# In[244]:


tmppt


# In[245]:


pixId = mypix.pt2pix(tmppt)
pixId


# In[246]:


center = mypix.pix2pt(pixId)


# In[247]:


center


# In[248]:


icoPoints[2]


# In[ ]:




