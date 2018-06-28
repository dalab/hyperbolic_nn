# Generate the 3d MLR figure from our paper.

import numpy as np
import numpy.linalg as LA
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3


def mob_add(u, v, c):
    numerator = (1.0 + 2.0 * c * np.dot(u,v) + c * LA.norm(v)**2) * u + (1.0 - c * LA.norm(u)**2) * v
    denominator = 1.0 + 2.0 * c * np.dot(u,v) + c**2 * LA.norm(v)**2 * LA.norm(u)**2
    return numerator / denominator


# fig = plt.figure()
#
# ax = plt.axes(projection='3d')
#
# def f(x, y):
#     return (1. - (x ** 2 + y ** 2)) * random.choice([-1,1])

sphere = np.random.random((3, 5000)) - 0.5
sphere = sphere / np.linalg.norm(sphere, axis=0)

X = sphere[0]
Y = sphere[1]
Z = sphere[2]


fig=p.figure()
ax = p3.Axes3D(fig)
# plot3D requires a 1D array for x, y, and z
# ravel() converts the 100x100 array into a 1x10000 array
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter3D(X,Y,Z, c='y', s = .1)


pp = np.array([0., 0., 1.])
pp = pp/ np.linalg.norm(pp) * 0.5

a = np.random.random(3) - 0.5
a = a / np.linalg.norm(a)

# a_line = np.zeros((200, 3))
# for j in range(200):
#     a_line[j] = a * np.random.random()
# ax.scatter3D(a_line[:,0], a_line[:,1], a_line[:,2], c='g', s = 5)


aa_line = np.zeros((200, 3))
for j in range(200):
    aa_line[j] = - a * 2. * (np.random.random() - 0.5) + pp
ax.scatter3D(aa_line[:,0], aa_line[:,1], aa_line[:,2], c='r', s = 10)


ax.scatter3D(pp[0], pp[1], pp[2], c='r', s = 500)


num = 3000
r = np.zeros((num + 300, 3))
for i in range(num):
    solved = False
    if i % 100 == 0:
        print(i)
    while not solved:
        xx = np.random.random(3) - 0.5
        xx = xx / np.linalg.norm(xx)
        rescale = np.random.random()
        rescale = 1 - (1 - rescale) ** 2
        v = mob_add(-pp, xx * rescale, 1.)

        if abs(v.dot(a)) < 1e-2:
            r[i] = xx * rescale # mob_add(pp, xx * rescale, 1.)
            solved = True


for i in range(300):
    solved = False
    if i % 100 == 0:
        print(i)
    while not solved:
        xx = np.random.random(3) - 0.5
        xx = xx / np.linalg.norm(xx)
        v = mob_add(-pp, xx, 1.)

        if abs(v.dot(a)) < 1e-2:
            r[i + num] = xx
            solved = True


ax.scatter3D(r[:num,0], r[:num,1], r[:num,2], c='g', s = 5, edgecolor='g')
ax.scatter3D(r[num:,0], r[num:,1], r[num:,2], c='g', s = 5, edgecolor='g')

ax.set_xbound(-1., 1.)
ax.set_ybound(-1., 1.)
ax.set_zbound(-1., 1.)


fig.add_axes(ax)


p.show()
