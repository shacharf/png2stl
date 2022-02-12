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
    parser.add_argument('--height', type=int, default=8, help='height of the model in mm')
    parser.add_argument('--pic_height', type=int, default=2, help='height of the height-field / engraving')
    parser.add_argument('--engrave', action='store_true')
    args = parser.parse_args()
    return args


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

    def quadsplit(quad):
        """Split ccw quad to two triangles"""
        v1, v2, v3, v4 = quad
        return [
            [v1, v2, v3],
            [v3, v4, v1]]
    
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
    V, F = im2stl(im, args.size, args.pic_height, args.height, args.engrave)
    save_stl(args.outstl, V, F)


if __name__ == '__main__':
    main()
