import matplotlib.pyplot as plt
import numpy as np
import mayavi
from mayavi import mlab
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib
# source:
# https://3dprint.com/index.php?gf-download=2020%2F03%2Fmaterials-12-04134.pdf&form-id=35&field-id=8&hash=2b1d844c66cbda26a2b642db84918a78a59d48c9b2e79e1f728ec7ad4ce648e8
#

def test2d():

    def f(X, Y):
        Z = np.sin(X) * np.cos(Y)
        return Z

    X = np.linspace(-18.0, 18.0, 800 )
    Y = np.linspace(-18.0, 18.0, 800 )
    Z = np.linspace(-18.0, 18.0, 800 )
    x,y = np.meshgrid(X, Y)
    z = f(x, y)
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(x, y, z)
    plt.subplot(2,1,2)
    plt.imshow(z, cmap='seismic')
    plt.show()

def writeobj(fname, verts, faces, normals):
    with open(fname, "w") as f:
        for item in verts:
            f.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

        for item in normals:
            f.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

        for item in faces:
            f.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0]+1, item[1]+1, item[2]+1))


def test3d():

    def f(X, Y, Z):
        S = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
        return S

    R = 100  # resolution
    D = 5
    r = D/2.0   # radius
    Cnt = 5
    sample_radius = np.pi * Cnt
    sr = sample_radius
    X = np.linspace(-sr, sr, R )
    Y = np.linspace(-sr, sr, R )
    Z = np.linspace(-sr, sr, R )
    x,y,z = np.meshgrid(X, Y, Z)
    s = f(x, y,z)
    s = s * (r/sr)
    s[0,:,:] = 0
    s[-1,:,:] = 0
    s[:,0,:] = 0
    s[:,-1,:] = 0
    s[:,:,0] = 0
    s[:,:,-1] = 0
    verts, faces, normals, values = measure.marching_cubes_lewiner(s, 0)
    if False:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

    # ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    # ax.set_ylabel("y-axis: b = 10")
    # ax.set_zlabel("z-axis: c = 16")
    #
    # ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
    # ax.set_ylim(0, 20)  # b = 10
    # ax.set_zlim(0, 32)  # c = 16

    # plt.tight_layout()
    # plt.show()
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow(s[0], cmap='seismic')
    plt.subplot(2,2,2)
    plt.imshow(s[3], cmap='seismic')
    plt.subplot(2,2,3)
    plt.imshow(s[8], cmap='seismic')
    plt.subplot(2,2,4)
    plt.imshow(s[15], cmap='seismic')
    plt.show()

    writeobj("c:/temp/a2.obj", verts, faces, normals)
    print("Done")


def gen3d(resolution, diameter, count):
    """
    :param resolution:
    :param diameter:
    :param count: number of building-blocks / axis
    :return:
    """

    def f(X, Y, Z):
        S = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
        return S

    print("sampling voxel grid")
    R = resolution
    D = diameter
    r = D/2.0           # radius
    bval = 0.0          # boundary value
    Cnt = count
    sample_radius = np.pi * Cnt
    sr = sample_radius
    X = np.linspace(-sr, sr, R )
    Y = np.linspace(-sr, sr, R )
    Z = np.linspace(-sr, sr, R )
    x,y,z = np.meshgrid(X, Y, Z)
    s = f(x, y, z)
    s[0,:,:] = bval
    s[-1,:,:] = bval
    s[:,0,:] = bval
    s[:,-1,:] = bval
    s[:,:,0] = bval
    s[:,:,-1] = bval
    print("Generating surface")
    verts, faces, normals, values = measure.marching_cubes_lewiner(s, bval)
    verts = verts * diameter / resolution
    return verts, faces, normals, values

def test_gencube():
    verts, faces, normals, values = gen3d(100, 20, 2.5)
    print("Saving...")
    writeobj("c:/temp/a3.obj", verts, faces, normals)
    print("Done")

if __name__ == '__main__':
    # test3d()
    test_gencube()