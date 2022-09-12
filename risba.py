# plot image for transformation of coordinate systems

# izboljšave: eliminate hardcoded stuff, naredi sliko tako, da samo radij vpišeš in se avtomatsko popravijo koordinate vseh stvari na sliki
# čeprav tukaj ni tako nujno, saj je samo ena slika


import matplotlib.pyplot as plt
import numpy as np
import math

r = 5

def circle_points(r):
    koti = []
    for i in range(10000):
        koti.append(i/10000 * 2*math.pi)

    xi = []
    yi = []
    for kot in koti:
        xi.append(r*math.cos(kot))
        yi.append(r*math.sin(kot))
    xi = np.array(xi)
    yi = np.array(yi)
    return xi, yi



def draw_image():
    xi, yi = circle_points(r)
    xpoints = np.array([0, 4])
    ypoints = np.array([0, 3])
    xpoints_hor = np.array([0, r])
    ypoints_hor = np.array([0, 0])
    xpoints_w = np.array([0, 0])
    ypoints_w = np.array([0, 4])
    xpoints_pod = np.array([4, 4])
    ypoints_pod = np.array([0, 6])
    xpoints_xos = np.array([1.5, 6])
    ypoints_xos = np.array([6.33, 0.33])
    xpoints_zos = np.array([4, 5.95])
    ypoints_zos = np.array([3, 4.52])
    plt.scatter(xi, yi, s=1)
    plt.plot(xpoints, ypoints)
    plt.plot(xpoints_hor, ypoints_hor, linestyle='dashed', alpha=0.3, color='black')
    plt.plot(xpoints_pod, ypoints_pod, alpha=0.3, linestyle='dashed', color='black')
    xpravokotno = np.array([3.7, 4])
    ypravokotno = np.array([0.3, 0.3])
    xpravokotno1 = np.array([3.7, 3.7])
    ypravokotno1 = np.array([0, 0.3])
    plt.plot(xpravokotno, ypravokotno, alpha=0.3, color='black')
    plt.plot(xpravokotno1, ypravokotno1, alpha=0.3, color='black')
    xpravokotno2 = np.array([4.3, 4.48])
    ypravokotno2 = np.array([3.2, 2.93])
    xpravokotno3 = np.array([4.48, 4.2])
    ypravokotno3 = np.array([2.93, 2.75])
    plt.plot(xpravokotno2, ypravokotno2, alpha=0.3, color='black')
    plt.plot(xpravokotno3, ypravokotno3, alpha=0.3, color='black')
    """ plt.plot(xpoints_xos, ypoints_xos, alpha=0.3, linestyle='dashed', color='black')
    plt.plot(xpoints_zos, ypoints_zos, alpha=0.3, linestyle='dashed', color='black') """
    plt.text(1.73, 0.58, s=r'$\gamma$', fontsize=8)
    plt.text(3.4, 4.5, s=r'$\gamma$', fontsize=8)
    plt.text(4.36, 4.5, s=r'$\frac{\pi}{2} - \gamma$', fontsize=8)
    plt.text(-0.8, 1.7, s=r'$\vec{\omega}$')
    plt.text(0.8, 6, s='x')
    plt.text(6.5, 4, s='z')
    plt.arrow(0, 0, 0, 2, head_width=0.2)
    plt.arrow(4, 3, 4/2*1.1, 3/2*1.1, head_width=0.2, linestyle='-', alpha=0.3, color='black')
    plt.arrow(6, 0.33, -3*1.4, 4*1.4, head_width=0.2, linestyle='-', alpha=0.3, color='black')
    plt.xlim([-7, 7])
    plt.ylim([-7, 7])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gcf().set_size_inches(6, 6)
    plt.axis('off')
    plt.title("Projectile's coordinate system")
    plt.show(block=False)
    plt.savefig('koordinatni_sistem.png')
    plt.pause(8)
    plt.close()










draw_image()