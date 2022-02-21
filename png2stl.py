import os
import sys
import argparse
import cv2
import numpy as np
import stl
from stl import mesh


def parse_args():
    parser = argparse.ArgumentParser(description='convert image to stl height map')
    parser.add_argument('--image',type=str, help='input image file path', required=True)
    parser.add_argument('--outstl',type=str, help='output model file path', default='out.stl')
    parser.add_argument('--size', type=int, nargs=2, default=(40,40),
                        help='dimension of the model width x height in mm')
    parser.add_argument('--height', type=float, default=8, help='height of the model in mm')
    parser.add_argument('--pic_height', type=float, default=2, help='height of the height-field / engraving')
    parser.add_argument('--engrave', action='store_true')
    parser.add_argument('--nonrect', action='store_true', help='If set, zero value generate holes, can be used for non-rectangular shapes')
    args = parser.parse_args()
    return args



def quadsplit(quad):
    """Split ccw quad to two triangles"""
    v1, v2, v3, v4 = quad
    return [
        [v1, v2, v3],
        [v3, v4, v1]]


def flip(t):
    """
    Flip a triangle t = [v1, v2, v3]
    """
    v1, v2, v3 = t
    t = [v1, v3, v2]
    return t


class Im2stl:
    """
        Args:
          im - input image
          size - output width, height in mm
          heightfield_height - max "z" value for the color white (255)
          shape_height - "z" value for the color black (0)
          engrave - if set, carve the heightfield from the shape


        1. find the boundaries in the image
        2. mark the boundary edges
        3. Generate height field for the top & bottom
        4. Generate the boundary
    """
    def __init__(self, im: np.ndarray, size: tuple, heightfield_height: int, shape_height: int, engrave: bool = False):
        self.im = im
        self.size = size
        self.heightfield_height = heightfield_height
        self.shape_height = shape_height
        self.engrave = engrave

        self.height0 = im.min()
        self.height1 = im.max()
        self.dheight = self.height1 - self.height0

        # output vertices and faces
        self.V = []
        self.F = []

    def get_stl(self):
        self.update_contours()
        self.generate_top()
        self.generate_sides()
        self.to_numpy()
        return self.V, self.F


    def update_contours(self):
        _, imbw = cv2.threshold(self.im.copy(), 1, 255, cv2.THRESH_BINARY)
#        cv2.imshow("bw", imbw)
#        cv2.waitKey(1)
        contours, hierarchy = cv2.findContours(imbw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contourimage = np.zeros(imbw.shape)
        print(f"Coontours: {len(contours[0])}")
#        cv2.drawContours(contourimage, contours, -1, 255, lineType=cv2.LINE_8)
#        cv2.imshow("cont", contourimage)
#        cv2.waitKey(-1)
        
#        self.contourimage = contourimage
        self.contours = contours
        self.hierarchy = hierarchy

    def generate_top(self):
        im = self.im
        w, h = self.size
        height0 = self.height0
        dheight = self.dheight
        ih, iw = im.shape
        shape_height = self.shape_height
        heightfield_height = self.heightfield_height

        print(f"Image resolution: {iw} x {ih}")
        print(f"object height: {shape_height}")
        print(f"engrave: {self.engrave}, h: {heightfield_height}")
        
        def c2i(u, v):
            """coordinate to index"""
            return v * iw + u
        
        X, Y = np.meshgrid(range(iw), range(ih))
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        nVertices = X.shape[0]
        Z = np.zeros(nVertices)
        Xlow = X.copy()
        Ylow = Y.copy()
        Zlow = Z.copy()
        bottom_start = len(X)
        self.bottom_start = bottom_start
        hfactor = heightfield_height / dheight
        for i, (x, y) in enumerate(zip(X, Y)):
            Z[i] = (im[y, x] - height0) * hfactor
            
        if self.engrave:
            Z = shape_height - Z
        else:
            Z = shape_height + Z
        

        def is_in_shape(t):
            result = True
            for vi in t:
                if im[Y[vi], X[vi]] == 0:
                    result = False
            return result
            
        faces = []
        for v in range(ih - 1):
            for u in range(iw - 1):
                triangles = quadsplit([c2i(u, v), c2i(u + 1, v), c2i(u + 1, v + 1), c2i(u, v + 1)])
                triangles = list(filter(is_in_shape, triangles))
                triangles2 = []
                for t in triangles:
                    triangles2.append([ti + bottom_start for ti in t])
                triangles2 = [flip(t) for t in triangles2]
                if len(triangles2) > 0:
                    triangles.extend(triangles2)
                if len(triangles) > 0:
                    faces.extend(triangles)

        X = np.concatenate([X, Xlow])
        Y = np.concatenate([Y, Ylow])
        Z = np.concatenate([Z, Zlow])
        
                # faces.append([c2i(u, v), c2i(u + 1, v + 1), c2i(u + 1, v)])
                # faces.append([c2i(u, v), c2i(u, v + 1), c2i(u + 1, v + 1)])

        # Close the shape
        # X1 = [0, iw, 0, iw]
        # Y1 = [0,  0, ih, ih]
        # Z1 = [0,  0,  0,  0]
        # F1 = []
        # F1.extend(quadsplit([nVertices, nVertices + 2 , nVertices + 3, nVertices + 1]))
        # F1.extend(quadsplit([nVertices, nVertices + 1 , iw - 1, 0]))
        # F1.extend(quadsplit([nVertices + 2, nVertices, 0, nVertices - iw]))
        # F1.extend(quadsplit([nVertices + 2, nVertices - iw, nVertices - 1, nVertices + 3]))
        # F1.extend(quadsplit([nVertices + 1, nVertices + 3, nVertices - 1, iw - 1]))
        
        # X = np.concatenate([X, np.array(X1)])
        # Y = np.concatenate([Y, np.array(Y1)])
        # Z = np.concatenate([Z, np.array(Z1)])
        
        X = X * w / iw
        Y = Y * h / ih
        # faces.extend(F1)

        self.V = np.vstack([X, Y, Z]).transpose()
        self.F = faces

    def c2i(self, x: int, y: int, bottom: bool = False):
        """
        Convert x,y coordinate to index in self.V
        Args:
           x,y
           bottom - True if we need the bottom vertex
        """
        ih, iw = self.im.shape
        idx = y * iw + x
        if bottom:
            idx += self.bottom_start
        return idx

    def generate_sides(self):
        sfaces = []
        # Support only outer contour for now
        contours = self.contours[0]
        N = len(contours)
        for i in range(N):
            i2 = (i + 1) % N
            x1, y1 = contours[i][0]
            x2, y2 = contours[i2][0]
            v1idx = self.c2i(x1, y1, False)
            v2idx = self.c2i(x2, y2, False)
            v3idx = self.c2i(x2, y2, True)
            v4idx = self.c2i(x1, y1, True)
            
            triangles = quadsplit([v1idx, v2idx, v3idx, v4idx])
            sfaces.extend(triangles)
        self.F.extend(sfaces)

    def to_numpy(self):
        self.F = np.array(self.F, dtype=np.int32)
        

    
def im2stl(im: np.ndarray, size: tuple, heightfield_height: int, shape_height: int, engrave: bool = False):
    w, h = size
    height0 = im.min()
    height1 = im.max()
    dheight = height1 - height0
    ih, iw = im.shape

    print(f"Image resolution: {iw} x {ih}")
    print(f"object height: {shape_height}")
    print(f"engrave: {engrave}, h: {heightfield_height}")

    def c2i(u, v):
        """coordinate to index"""
        return v * iw + u

    X, Y = np.meshgrid(range(iw), range(ih))
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    nVertices = X.shape[0]
    Z = np.zeros(nVertices)
    hfactor = heightfield_height / dheight
    for i, (x, y) in enumerate(zip(X, Y)):
        Z[i] = (im[y, x] - height0) * hfactor

    if engrave:
        Z = shape_height - Z
    else:
        Z = shape_height + Z
        

    faces = []
    for v in range(ih - 1):
        for u in range(iw - 1):
            faces.extend(quadsplit([c2i(u, v), c2i(u + 1, v), c2i(u + 1, v + 1), c2i(u, v + 1)]))
            # faces.append([c2i(u, v), c2i(u + 1, v + 1), c2i(u + 1, v)])
            # faces.append([c2i(u, v), c2i(u, v + 1), c2i(u + 1, v + 1)])

    # Close the shape
    X1 = [0, iw, 0, iw]
    Y1 = [0,  0, ih, ih]
    Z1 = [0,  0,  0,  0]
    F1 = []
    F1.extend(quadsplit([nVertices, nVertices + 2 , nVertices + 3, nVertices + 1]))
    F1.extend(quadsplit([nVertices, nVertices + 1 , iw - 1, 0]))
    F1.extend(quadsplit([nVertices + 2, nVertices, 0, nVertices - iw]))
    F1.extend(quadsplit([nVertices + 2, nVertices - iw, nVertices - 1, nVertices + 3]))
    F1.extend(quadsplit([nVertices + 1, nVertices + 3, nVertices - 1, iw - 1]))

    X = np.concatenate([X, np.array(X1)])
    Y = np.concatenate([Y, np.array(Y1)])
    Z = np.concatenate([Z, np.array(Z1)])

    X = X * w / iw
    Y = Y * h / ih
    faces.extend(F1)

    F = np.array(faces)
    V = np.vstack([X, Y, Z]).transpose()

    return V, F


def save_stl(outpath, V, F):
    shape = mesh.Mesh(np.zeros(F.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(F):
        for j in range(3):
            shape.vectors[i][j] = V[f[j], :]
    shape.save(outpath)
#    shape.save(outpath, mode=stl.Mode.ASCII)


def main():
    args = parse_args()
    im = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    if args.nonrect:
        converter = Im2stl(im, args.size, args.pic_height, args.height, args.engrave)
        V, F = converter.get_stl()
    else:
        V, F = im2stl(im, args.size, args.pic_height, args.height, args.engrave)

    save_stl(args.outstl, V, F)


if __name__ == '__main__':
    main()
