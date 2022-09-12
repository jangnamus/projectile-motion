import math
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.lines as mlines




# funkcije, ki se začnejo s 'show', ne vračajo vrednosti, ampak prikažejo graf
# globalne spremenljivke: g, gostota, m, S, v0, x0, y0, z0, st_meritev, dt, c_max, pavza, w, gama
# če ne bi uporabljal globalnih spremenljivk, bi imele funkcije zelo veliko parametrov





# zračni upor
# vrne domet
def zracni_upor(x, z, v0, phi):
    vx0 = v0 * math.cos(math.radians(phi))
    vz0 = v0 * math.sin(math.radians(phi))
    vx = vx0
    vz = vz0

    while z > 0:
        v = math.sqrt(vx**2 + vz**2)
        Fx = - (1/2)*gostota*S*v*vx
        Fz = - m*g - (1/2)*gostota*S*v*vz

        vx = vx + Fx/m * dt
        vz = vz + Fz/m * dt

        x = x + vx * dt
        z = z + vz * dt

    return x





# brez zračnega upora - numerično
# vrne domet
def brez_upora_num(x, z, v0, phi):
    vx0 = v0 * math.cos(math.radians(phi))
    vz0 = v0 * math.sin(math.radians(phi))
    vx = vx0
    vz = vz0

    while z > 0:
        vz = vz - g * dt

        x = x + vx * dt
        z = z + vz * dt

    return x









# brez zračnega upora - točno
# vrne domet
def brez_upora_tocno(v0, phi):
    x = (v0**2) / g * math.sin(2*math.radians(phi))
    return x




# internal function
# vrne nekaj parov kotov s podobnim dometom v brezvetrju
def najdi_podobne_zadetke(st_podobnih, najvecji_kot, koti, dometi):
    razlike = dict()
    nekaj_parov = []
    manjsi_koti = koti[:najvecji_kot]
    vecji_koti = koti[najvecji_kot + 1:]
    for kot1 in manjsi_koti:
        for kot2 in vecji_koti:
            razlike[(kot1, kot2)] = abs(dometi[kot1] - dometi[kot2])
    urejen = sorted(razlike.items(), key=lambda x: x[1])
    for i in range(st_podobnih):
        nekaj_parov.append(urejen[i][0])
    return nekaj_parov








# najde dva kota s podobnim dometom
# vrne domete, kote in dva kota s podobnim dometom
def find_two_angles():
    dometi = []
    koti = []
    for kot in range(91):
        koti.append(kot)

    print('kot       zračni upor                 brez upora numerično                 brez upora točno')
    for kot in koti:
        r1 = zracni_upor(x0, z0, v0, kot)
        r2 = brez_upora_num(x0, z0, v0, kot)
        r3 = brez_upora_tocno(v0, kot)
        dometi.append(r1)

        print(kot, "      ", r1, '           ', r2, '            ', r3)


    indeks_najvecjega = dometi.index(max(dometi))

    pari_najblizjih = najdi_podobne_zadetke(6, indeks_najvecjega, koti, dometi)

    #se enkrat naredi: pogoj: da sta vsaj 10 stopinj narazen in da sta najblizja izmed takih po dometu.

    najmanjse_odstopanje = 10000    # najmanjse odstopanje od optimalne razlike kotov s podobnima dometoma
    optimalna_razlika = 40
    for par in pari_najblizjih:
        a = abs(abs(par[1] - par[0]) - optimalna_razlika)
        if a < najmanjse_odstopanje:
            najmanjse_odstopanje = a
            pravi_par = par

    return dometi, koti, pravi_par










# prikaže domet v odvisnosti od kotov
def show_dometi_koti(dometi, koti, pravi_par):

    najvecji = max(dometi)
    indeks_najvecjega = dometi.index(max(dometi))
    pravi_domet = dometi[pravi_par[0]]

    koti = np.array(koti)
    dometi = np.array(dometi)
    xpoints = np.array([0, 90])
    ypoints = np.array([pravi_domet, pravi_domet]) 
    yerr = 1000
    plt.scatter(koti, dometi, s=10, label='range')
    plt.plot((indeks_najvecjega, indeks_najvecjega), (0, najvecji), linestyle='dashed', alpha=0.5, label='maximum range launch angle')
    #plt.plot(xpoints, ypoints, color='red', linestyle='dashed', alpha=0.4, label='kota s podobnima dometoma')
    #ali pa to za fitanje: plt.errorbar(koti, dometi, marker='s', ms = 3, mfc='r', ecolor='g', label='meritve')
    plt.plot(koti, dometi, 'r-', alpha=0.8, linewidth=1, label='curve fit')

    plt.xlabel(r'$koti$' + ' ' + r'$[\degree]$')
    plt.ylabel(r'$domet$' + ' ' + r'$[m]$')
    plt.title('Range vs. launch angle')
    #plt.text(pravi_par[0], pravi_domet - pravi_domet/10, s=f"kota: {pravi_par}")
    plt.text(x=indeks_najvecjega + 3, y = najvecji/10, s=r'$\phi_{max} = $' + f'{str(indeks_najvecjega)}' + r'$\degree$')
    plt.legend(fontsize=9, loc='upper right')
    plt.xlabel(r'$\phi$' + ' ' + r'$[\degree]$')
    plt.ylabel(r'$x$' + ' ' + r'[m]')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.gca().set_ylim(top=najvecji+2000)
    plt.gca().set_xlim(right=100)
    plt.show(block=False) 
    plt.savefig('domet_kot.png')
    plt.pause(pavza)
    plt.close()



#----------------------------------------------------------------------------------------------------------------------


# Tir leta






# internal function
# vrne tir 2d, brez upora (x, z koordinate)
def tir_dvad_brezupora(x, z, v0, phi):
    x_coordinates = []
    z_coordinates = []

    vx0 = v0 * math.cos(math.radians(phi))
    vz0 = v0 * math.sin(math.radians(phi))
    vx = vx0
    vz = vz0

    while z > 0:
        vz = vz - g * dt

        x = x + vx * dt
        z = z + vz * dt

        x_coordinates.append(x)
        z_coordinates.append(z)

    x_coordinates = np.array(x_coordinates)
    z_coordinates = np.array(z_coordinates)

    return x_coordinates, z_coordinates









# internal function
# vrne tir 2d, samo upor (x, z koordinate)
def tir_leta_dvad(x, z, v0, phi):
    x_coordinates = []
    z_coordinates = []

    vx = v0 * math.cos(math.radians(phi))
    vz = v0 * math.sin(math.radians(phi))

    while z > 0:
        v = math.sqrt(vx**2 + vz**2)
        Fx = - (1/2)*gostota*S*v*vx
        Fz = - m*g - (1/2)*gostota*S*v*vz

        vx = vx + Fx/m * dt
        vz = vz + Fz/m * dt

        x = x + vx * dt
        z = z + vz * dt

        x_coordinates.append(x)
        z_coordinates.append(z)

    final_angle = round(math.degrees(math.atan(abs(vz/vx))))

    x_coordinates = np.array(x_coordinates)
    z_coordinates = np.array(z_coordinates)

    return x_coordinates, z_coordinates













# internal function
# vrne tir 3d, z vetrom (x, y, z koordinate)
def tir_leta_trid(x, y, z, v0, phi, veter_vektor):

    vx = v0 * math.cos(math.radians(phi))
    vz = v0 * math.sin(math.radians(phi))
    vy = 0 
    cx = veter_vektor.get_x()
    cy = veter_vektor.get_y()

    xos = []
    yos = []
    zos = []

    while z > 0:
        v = math.sqrt((vx-cx)**2 + (vy-cy)**2 + (vz)**2)
        Fx = - (1/2)*gostota*S*v*(vx-cx)
        Fy = - (1/2)*gostota*S*v*(vy-cy)
        Fz = - m*g - (1/2)*gostota*S*v*vz

        vx = vx + (Fx/m)*dt
        vy = vy + (Fy/m)*dt
        vz = vz + (Fz/m)*dt

        x = x + vx*dt
        y = y + vy*dt
        z = z + vz*dt

        xos.append(x)
        yos.append(y)
        zos.append(z)

    xos = np.array(xos)
    yos = np.array(yos)
    zos = np.array(zos)

    return xos, yos, zos

















# prikaže tir 2d, samo upor
def show_tir_leta_dvad(x, z, v0, phi):

    x_coordinates, z_coordinates = tir_leta_dvad(x, z, v0, phi)
    plt.title(f'2d trajectory {str(phi)}' + r'$\degree$')
    plt.text(100, 0, f'{str(phi)}' + r'$\degree$')
    #plt.text(x-100, 0, f'{str(final_angle)}' + r'$\degree$')
    plt.scatter(x_coordinates, z_coordinates)
    plt.show(block=False)
    plt.pause(pavza)
    plt.close()














# prikaže tir 3d, z vetrom
def show_tir_leta_trid(x, y, z, v0, phi, vetri):

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.set_title('3D trajectories')

    for veter_vektor in vetri:
        xos, yos, zos = tir_leta_trid(x, y, z, v0, phi, veter_vektor)
        ax.plot3D(xos, yos, zos)
    
    ax.plot([0, xos.max()], [0, 0], color='black', alpha=0.5, linestyle='dashed')
    plt.scatter(0, 0, s=12, c='black')

    #ax.dist=15
    ax.view_init(elev=35, azim=315)
    #ax.text(0, -300, 5000, f'veter {veter_vektor}', color='blue')
    ax.set_xlabel('x [m]', labelpad=15)
    ax.set_ylabel('y [m]', labelpad=15)
    ax.set_zlabel('z [m]', labelpad=15)
    plt.show(block=False)
    plt.savefig('vizualizacija3d.png')
    plt.pause(pavza)
    plt.close()




















# prikaže primerjavo metov z in brez zračnega upora
def show_primerjava_zbrezupora(kot):
    x, z = tir_dvad_brezupora(x0, z0, v0, kot)
    x1, z1 = tir_leta_dvad(x0, z0, v0, kot)
    plt.scatter(x, z, s=1, label='without air resistance', c='blue')
    plt.scatter(x1, z1, s=1, label='with air resistance', c='orange')
    plt.title(f'Trajectory with and without air resistance (launch angle {kot}' + r'$\degree$' + ')')
    plt.xlabel(r'x' + ' ' + r'[m]')
    plt.ylabel(r'z' + ' ' + r'[m]')
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=z.max() + z.max()/4)
    orange_circle = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                            markersize=6, alpha=1, label='with air resistance')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=6, alpha=1, label='without air resistance')
    plt.legend(handles=[blue_circle, orange_circle], fontsize=8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show(block=False) 
    plt.savefig('primerjava_zbrez.png')
    plt.pause(pavza)
    plt.close()











# prikaže 2d tire več metov pod različnimi koti v brezvetrju
def show_vec_kotov(koti):
    circles = []
    for kot in koti:
        x, z = tir_leta_dvad(x0, z0, v0, kot)
        p = plt.scatter(x, z, s=1, label=f'{kot}'+r'$\degree$')                                # https://stackoverflow.com/questions/47908429/get-color-of-a-scatter-point
        circle = mlines.Line2D([], [], color=p.get_facecolors()[0].tolist(), marker='o', linestyle='None',
                            markersize=6, alpha=1, label=f'{kot}'+r'$\degree$')
        circles.append(circle)
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=6, alpha=1, label=f'{15}'+r'$\degree$')
    orange_circle = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                            markersize=6, alpha=1, label=f'{30}'+r'$\degree$')
    green_circle = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                            markersize=6, alpha=1, label=f'{45}'+r'$\degree$')
    red_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                            markersize=6, alpha=1, label=f'{60}'+r'$\degree$')
    purple_circle = mlines.Line2D([], [], color='purple', marker='o', linestyle='None',
                            markersize=6, alpha=1, label=f'{75}'+r'$\degree$')
    plt.legend(handles=circles, fontsize=8)
    plt.title('Trajectories of the projectile in the presence of air resistance')
    plt.xlabel(r'x' + ' ' + r'[m]')
    plt.ylabel(r'z' + ' ' + r'[m]')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show(block=False) 
    plt.savefig('vec_kotov.png')
    plt.pause(pavza)
    plt.close()



#----------------------------------------------------------------------------------------------------------------



















# Veter

class Vektor():
    def __init__(self, x, y, theta) -> None:
        self.x = x
        self.y = y
        self.theta = theta
    
    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_theta(self):
        return self.theta

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def __str__(self) -> str:
        return 'x: ' + str(round(self.x, 2)) + ', y: ' + str(round(self.y, 2))








# ustvari st_vetrov različnih vetrov in jih vrne
def naredi_vetre(max_speed, st_vetrov):
    normirani_rji = []
    ploskovno_porazdeljeni_rji = []
    koncni_rji = []
    thetas = []

    for i in range(st_vetrov):
        normirani_rji.append(random.random())

    for i in range(st_vetrov):
        angle_random = random.randint(0, 628)  # 628 = 2*pi * 100
        thetas.append(angle_random / 100)

    for r in normirani_rji:
        ploskovno_porazdeljeni_rji.append(math.sqrt(r))

    for r in ploskovno_porazdeljeni_rji:
        koncni_rji.append(r * max_speed)

    vectors = []
    x_koordinate = []
    y_koordinate = []

    for i in range(st_vetrov):
        x = koncni_rji[i] * math.cos(thetas[i])
        y = koncni_rji[i] * math.sin(thetas[i])
        theta = thetas[i]
        x_koordinate.append(x)
        y_koordinate.append(y)

        vektor = Vektor(x, y, theta)
        vectors.append(vektor)

    return vectors




















# prikaže enakomerno porazdelitev izžrebanih vetrov po površini kroga
def show_distribution(vektorji):
    x_koordinate = []
    y_koordinate = []
    for vektor in vektorji:
        cx = vektor.get_x()
        cy = vektor.get_y()
        x_koordinate.append(cx)
        y_koordinate.append(cy)
    x_koordinate = np.array(x_koordinate)
    y_koordinate = np.array(y_koordinate)

    plt.title('Uniform distribution of points within a circle')
    plt.scatter(x_koordinate, y_koordinate, s=1)
    plt.gca().set_aspect('equal', adjustable='box')   # da imata osi enak scale
    plt.xticks(range(-8,10,2) )   # Put x axis ticks every 2 units.
    plt.yticks(range(-8,10,2) )
    plt.xlabel(r'$c_{x}$' + ' ' + r'$[\frac{m}{s}]$')
    plt.ylabel(r'$c_{y}$' + ' ' + r'$[\frac{m}{s}]$')
    plt.show(block=False)
    plt.savefig('enakomerna_porazdelitev.png')
    plt.pause(pavza)
    plt.close()




# prikaže enakomerno porazdelitev izžrebanih vetrov po površini kroga, 12 skupin vsaka različna barva
def show_groups_distribution(vektorji):
    v1 = []
    v2 = []
    v3 = []
    v4 = []
    v5 = []
    v6 = []
    v7 = []
    v8 = []
    v9 = []
    v10 = []
    v11 = []
    v12 = []

    skupine_vektorjev = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12]

    for vektor in vektorji:
        if vektor.length() < c_max/(math.sqrt(2)):
            if math.radians(0) <= vektor.get_theta() < math.radians(60):
                v1.append(vektor)
            if math.radians(60) <= vektor.get_theta() < math.radians(120):
                v2.append(vektor)
            if math.radians(120) <= vektor.get_theta() < math.radians(180):
                v3.append(vektor)
            if math.radians(180) <= vektor.get_theta() < math.radians(240):
                v4.append(vektor)
            if math.radians(240) <= vektor.get_theta() < math.radians(300):
                v5.append(vektor)
            if math.radians(300) <= vektor.get_theta() < math.radians(360):
                v6.append(vektor)
        else:
            if math.radians(0) <= vektor.get_theta() < math.radians(60):
                v7.append(vektor)
            if math.radians(60) <= vektor.get_theta() < math.radians(120):
                v8.append(vektor)
            if math.radians(120) <= vektor.get_theta() < math.radians(180):
                v9.append(vektor)
            if math.radians(180) <= vektor.get_theta() < math.radians(240):
                v10.append(vektor)
            if math.radians(240) <= vektor.get_theta() < math.radians(300):
                v11.append(vektor)
            if math.radians(300) <= vektor.get_theta() < math.radians(360):
                v12.append(vektor)

    colors = ['blue', 'black', 'purple', 'red', 'yellow', 'green', 'orange', 'pink', 'goldenrod', 'darkcyan', 'navy', 'maroon']
    for skupina in skupine_vektorjev:
        i = skupine_vektorjev.index(skupina)
        x_koord = []
        y_koord = []
        for vektor in skupina:
            x = vektor.get_x()
            y = vektor.get_y()
            x_koord.append(x)
            y_koord.append(y)
        x_koord = np.array(x_koord)
        y_koord = np.array(y_koord)
        plt.scatter(x_koord, y_koord, color=colors[i], s=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Uniform distribution of points within a circle by groups')
    plt.show(block=False)
    plt.pause(10)
    plt.close()









































#----------------------------------------------------------------------------------------------------------------

# Raztresenost zadetkov v vetru



# vrne (x,y) koordinate zadetka v vetru
def z_vetrom(x, y, z, v0, phi, veter_vektor):
    vx = v0 * math.cos(math.radians(phi))
    vz = v0 * math.sin(math.radians(phi))
    vy = 0

    cx = veter_vektor.get_x()
    cy = veter_vektor.get_y()

    while z > 0:
        v = math.sqrt((vx-cx)**2 + (vy-cy)**2 + (vz)**2)
        Fx = - (1/2)*gostota*S*v*(vx-cx)
        Fy = - (1/2)*gostota*S*v*(vy-cy)
        Fz = - m*g - (1/2)*gostota*S*v*vz

        vx = vx + (Fx/m)*dt
        vy = vy + (Fy/m)*dt
        vz = vz + (Fz/m)*dt

        x = x + vx*dt
        y = y + vy*dt
        z = z + vz*dt

    return x, y





# internal function
# vrne zadetke metov v različnih vetrovnih pogojih
def dispersion_wind(alfa, beta, vektorji):
    xi_v_vetru_alfa = []
    yi_v_vetru_alfa = []
    xi_v_vetru_beta = []
    yi_v_vetru_beta = []


    for veter_vektor in vektorji:
        x, y = z_vetrom(x0, y0, z0, v0, alfa, veter_vektor)
        xi_v_vetru_alfa.append(x)
        yi_v_vetru_alfa.append(y)

    for veter_vektor in vektorji:
        x, y = z_vetrom(x0, y0, z0, v0, beta, veter_vektor)
        xi_v_vetru_beta.append(x)
        yi_v_vetru_beta.append(y)

    xi_v_vetru_alfa = np.array(xi_v_vetru_alfa)
    yi_v_vetru_alfa = np.array(yi_v_vetru_alfa)
    xi_v_vetru_beta = np.array(xi_v_vetru_beta)
    yi_v_vetru_beta = np.array(yi_v_vetru_beta)
    
    return xi_v_vetru_alfa, yi_v_vetru_alfa, xi_v_vetru_beta, yi_v_vetru_beta







# prikaže raztresenost zadetkov v različnih vetrovnih pogojih
def show_dispersion_wind(alfa, beta, vektorji):
    domet_v_brezvetrju_alfa = zracni_upor(x0, z0, v0, alfa)
    domet_v_brezvetrju_beta = zracni_upor(x0, z0, v0, beta)
    print(f"v brezvetrju alfa {alfa}:", domet_v_brezvetrju_alfa)
    print(f"v brezvetrju beta {beta}: ", domet_v_brezvetrju_beta)

    points = dispersion_wind(alfa, beta, vektorji)

    xpoint = np.array([domet_v_brezvetrju_alfa])
    xpoint1 = np.array([domet_v_brezvetrju_beta])
    ypoint = np.array([0])
    xi_v_vetru_alfa = points[0]
    yi_v_vetru_alfa = points[1]
    xi_v_vetru_beta = points[2]
    yi_v_vetru_beta = points[3]

    tezisce_x_alfa = round(np.average(xi_v_vetru_alfa), 3)
    tezisce_y_alfa = round(np.average(yi_v_vetru_alfa), 3)
    tezisce_x_beta = round(np.average(xi_v_vetru_beta), 3)
    tezisce_y_beta = round(np.average(yi_v_vetru_beta), 3)
    print(xi_v_vetru_alfa.min(), xi_v_vetru_alfa.max(), yi_v_vetru_alfa.min(), yi_v_vetru_alfa.max())

    #print(xi_v_vetru_alfa[:30], yi_v_vetru_alfa[:30])
    plt.scatter(xi_v_vetru_alfa, yi_v_vetru_alfa, color='blue', s=1, label=r'$\phi = $' + f'{alfa}' + r'$\degree$')
    plt.scatter(xi_v_vetru_beta, yi_v_vetru_beta, color='orange', s=1, label=r'$\phi = $' + f'{beta}' + r'$\degree$')
    #plt.scatter(xpoint, ypoint, color='red', s=50)
    plt.scatter(xpoint1, ypoint, color='black', s=50, label='domet v brezvetrju')
    plt.axhline(y=tezisce_y_alfa, color='black', alpha=0.3, linestyle='dashed')
    """ plt.axvline(x=tezisce_x_alfa, color='red', alpha=0.5, linestyle='dashed')
    plt.axhline(y=tezisce_y_beta, color='black', alpha=0.5, linestyle='dashed')
    plt.axvline(x=tezisce_x_beta, color='red', alpha=0.5, linestyle='dashed') """
    plt.title('Scatter of landing points in different windy conditions')
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([xi_v_vetru_beta.min() - 60, xi_v_vetru_beta.max() + 60])
    plt.ylim([yi_v_vetru_beta.min() - 100, yi_v_vetru_beta.max() + 150])
    black_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=8, label='no-wind range')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                markersize=8, label=r'$\phi = $' + f'{alfa}' + r'$\degree$')
    orange_circle = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                            markersize=8, label=r'$\phi = $' + f'{beta}' + r'$\degree$')
    plt.legend(handles=[black_circle, blue_circle, orange_circle], fontsize=8)
    plt.xlabel(r'$x$' + ' ' + r'[m]')
    plt.ylabel(r'$y$' + ' ' + r'[m]')
    plt.show(block=False)
    plt.savefig('raztresenost_veter.png')
    plt.pause(pavza)
    plt.close()







# prikaže raztresenost zadetkov v različnih vetrovnih pogojih po skupinah
def show_groups_distribution_wind(skupine_vektorjev, stevila, colors):

    skupine_dometov = []
    for skupina in skupine_vektorjev:
        dometi_te_skupine = []
        for veter in skupina:
            x = z_vetrom(0, 0, 0.001, v0, alfa, veter)
            dometi_te_skupine.append(x)
        skupine_dometov.append(dometi_te_skupine)


    for dometi_skupine in skupine_dometov:
        i = skupine_dometov.index(dometi_skupine)
        dometi_skupine = np.array(dometi_skupine)
        plt.scatter(stevila, dometi_skupine, color=colors[i], s=1)

    plt.show(block=False)
    plt.pause(pavza)
    plt.close()























#----------------------------------------------------------------------------------------------------------------

# Coriolis


# vrne (x,y) koordinate zadetka v vetru ob upoštevanju coriolisa
def z_vetrom_coriolis(x, y, z, v0, phi, veter_vektor, w, gama):
    vx = v0 * math.cos(math.radians(phi))
    vz = v0 * math.sin(math.radians(phi))
    vy = 0
    t = 0

    cx = veter_vektor.get_x()
    cy = veter_vektor.get_y()

    while z > 0:
        a_cor = vektorski_produkt([vx, vy, vz], [w*math.cos(gama), 0, w*math.sin(gama)]) 

        ax = a_cor[0]
        ay = a_cor[1]
        az = a_cor[2]

        v = round(math.sqrt((vx-cx)**2 + (vy-cy)**2 + (vz)**2), 5)

        Fx = - (1/2)*gostota*S*v*(vx-cx) + 2*m*ax
        Fy = - (1/2)*gostota*S*v*(vy-cy) + 2*m*ay
        Fz = - m*g - (1/2)*gostota*S*v*vz + 2*m*az

        vx = vx + (Fx/m)*dt
        vy = vy + (Fy/m)*dt
        vz = vz + (Fz/m)*dt

        x = x + vx*dt
        y = y + vy*dt
        z = z + vz*dt
        t += dt

    return x, y, t










def vektorski_produkt(a, b):
    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]
    return (x, y, z)










# internal function
# vrne zadetke metov v različnih vetrovnih pogojih ob upoštevanju coriolisa
def dispersion_coriolis(alfa, beta, vektorji):
    xi_s_coriem_alfa = []
    yi_s_coriem_alfa = []
    xi_s_coriem_beta = []
    yi_s_coriem_beta = []

    casi = []

    for veter_vektor in vektorji:
        x, y, t = z_vetrom_coriolis(x0, y0, z0, v0, alfa, veter_vektor, w, gama)
        xi_s_coriem_alfa.append(x)
        yi_s_coriem_alfa.append(y)

    for veter_vektor in vektorji:
        x, y, t = z_vetrom_coriolis(x0, y0, z0, v0, beta, veter_vektor, w, gama)
        xi_s_coriem_beta.append(x)
        yi_s_coriem_beta.append(y)
        casi.append(t)

    print('povprečen čas letenja: ', round(sum(casi)/ len(casi), 2))

    xi_s_coriem_alfa = np.array(xi_s_coriem_alfa)
    yi_s_coriem_alfa = np.array(yi_s_coriem_alfa)
    xi_s_coriem_beta = np.array(xi_s_coriem_beta)
    yi_s_coriem_beta = np.array(yi_s_coriem_beta)

    return xi_s_coriem_alfa, yi_s_coriem_alfa, xi_s_coriem_beta, yi_s_coriem_beta











# prikaže raztresenost zadetkov v različnih vetrovnih pogojih ob upoštevanju coriolisa
def show_distribution_coriolis(alfa, beta, vektorji):

    domet_v_brezvetrju_alfa = zracni_upor(x0, z0, v0, alfa)
    domet_v_brezvetrju_beta = zracni_upor(x0, z0, v0, beta)

    xpoint = np.array([domet_v_brezvetrju_alfa])
    xpoint1 = np.array([domet_v_brezvetrju_beta])
    ypoint = np.array([0])

    points = dispersion_coriolis(alfa, beta, vektorji)
    points_just_wind = dispersion_wind(alfa, beta, vektorji)
    

    xi_v_vetru_alfa, yi_v_vetru_alfa, xi_v_vetru_beta, yi_v_vetru_beta = points
    xi_s_coriem_alfa, yi_s_coriem_alfa, xi_s_coriem_beta, yi_s_coriem_beta = points_just_wind


    #plt.scatter(xpoint, ypoint, color='red', s=50)
    plt.title('Scatter of landing points in different windy conditions with Coriolis')
    plt.scatter(xi_v_vetru_beta, yi_v_vetru_beta, color='orange', alpha=0.15, s=1, label=r'$\phi = $' + f'{beta}' + r'$\degree$' + ', without Coriolis')
    plt.scatter(xi_s_coriem_beta, yi_s_coriem_beta, color="green", s=1, label=r'$\phi = $' + f'{beta}' + r'$\degree$' + ', with Coriolis')
    plt.scatter(xi_v_vetru_alfa, yi_v_vetru_alfa, color='blue', alpha=0.15, s=1, label=r'$\phi = $' + f'{alfa}' + r'$\degree$' + ', without Coriolis')
    plt.scatter(xi_s_coriem_alfa, yi_s_coriem_alfa, color="deeppink", s=1, label=r'$\phi = $' + f'{alfa}' + r'$\degree$' + ', with Coriolisom')
    plt.scatter(xpoint1, ypoint, color='black', s=50, label='no-wind range')

    tezisce_x_cor_alfa = round(np.average(xi_s_coriem_alfa), 3)
    tezisce_y_cor_alfa = round(np.average(yi_s_coriem_alfa), 3)
    tezisce_x_cor_beta = round(np.average(xi_s_coriem_beta), 3)
    tezisce_y_cor_beta = round(np.average(yi_s_coriem_beta), 3)
    """ plt.axhline(y=tezisce_y_alfa, color='black', alpha=0.5, linestyle='dashed')
    plt.axvline(x=tezisce_x_alfa, color='red', alpha=0.5, linestyle='dashed')
    plt.axhline(y=tezisce_y_beta, color='black', alpha=0.5, linestyle='dashed')
    plt.axvline(x=tezisce_x_beta, color='red', alpha=0.5, linestyle='dashed')
    plt.axhline(y=tezisce_y_cor_alfa, color='green', alpha=0.5, linestyle='dashed')
    plt.axvline(x=tezisce_x_cor_alfa, color='yellow', alpha=0.5, linestyle='dashed')
    plt.axhline(y=tezisce_y_cor_beta, color='green', alpha=0.5, linestyle='dashed')
    plt.axvline(x=tezisce_x_cor_beta, color='yellow', alpha=0.5, linestyle='dashed') """
    plt.axhline(y=tezisce_y_cor_alfa, color='black', alpha=0.3, linestyle='dashed')
    plt.axhline(y=tezisce_y_cor_beta, color='black', alpha=0.3, linestyle='dashed')
    plt.xlim([xi_s_coriem_beta.min() - 100, xi_v_vetru_beta.max() + 100])
    plt.ylim([yi_s_coriem_beta.min() - 100, yi_v_vetru_beta.max() + 120])
    black_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=6, label='no-wind range')
    pink_circle = mlines.Line2D([], [], color='deeppink', marker='o', linestyle='None',
                            markersize=6, label=r'$\phi = $' + f'{alfa}' + r'$\degree$' + ', with Coriolis')
    green_circle = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                            markersize=6, label=r'$\phi = $' + f'{beta}' + r'$\degree$' + ', with Coriolis')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                markersize=6, alpha=0.2, label=r'$\phi = $' + f'{alfa}' + r'$\degree$' + ', without Coriolis')
    orange_circle = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                            markersize=7, alpha=0.2, label=r'$\phi = $' + f'{beta}' + r'$\degree$' + ', without Coriolis')
    print(f"novo težišče kot {alfa}: ", tezisce_y_cor_alfa)
    print(f"novo težišče kot {beta}: ", tezisce_y_cor_beta)
    plt.legend(handles=[black_circle, pink_circle, green_circle, blue_circle, orange_circle], fontsize=6, loc='upper right')
    plt.xlabel(r'$x$' + ' ' + r'[m]')
    plt.ylabel(r'$y$' + ' ' + r'[m]')
    plt.show(block=False)
    plt.savefig('raztresenost_coriolis.png')
    plt.pause(pavza)
    plt.close()


    print(xi_v_vetru_alfa.max(), xi_v_vetru_alfa.min())
    print(xi_s_coriem_alfa.max(), xi_s_coriem_alfa.min())





































#---------------------------------------------------------------------------------------------------------------------------------------------------
# Function calls (main)


# globalne spremenljivke
# original: m=40kg, S=0.0045m2, v0=500m/s
g = 10
gostota = 1
m = 40
S = 0.0045
v0 = 500

x0 = 0
y0 = 0
z0 = 0.00000001

st_meritev = 5000
dt = 0.01
c_max = 8
pavza = 9

w = 7.272e-5
gama = 46.54    #zemljepisna širina v stopinjah



def main():
    start = time.time()

    dometi, koti, pravi_par = find_two_angles()
    alfa = pravi_par[0]
    beta = pravi_par[1]


    # za to funkcijo spremenim vrednosti S iz 0.0045 na 0.05
    #show_primerjava_zbrezupora(52)

    angles = [15, 30, 45, 60, 75]
    show_vec_kotov(angles)

    show_dometi_koti(dometi, koti, pravi_par)


    vetrcki = naredi_vetre(c_max, st_meritev)

    show_distribution(vetrcki)
    show_groups_distribution(vetrcki)


    show_dispersion_wind(alfa, beta, vetrcki)
    show_distribution_coriolis(alfa, beta, vetrcki)


    stop = time.time()
    print("čas računanja: ", stop - start)












#-----------------------------------------------------------------------------------
# run
main()

#-------------------------------------------------------------------------------------------------
 
# neke specificne zadeve za neke specificne analize oz grafe

""" nicelni_veter = Vektor(0, 0, 0)
vetrcki = naredi_vetre(c_max, st_meritev)


izbrani_vetri = [nicelni_veter]
mam_poz = False
mam_neg = False
for veter in vetrcki:
    if mam_poz and mam_neg:
        break
    if abs(veter.get_x()) < 1:
        if veter.get_y() > 5 and not mam_poz:
            izbrani_vetri.append(veter)
            mam_poz = True
        elif veter.get_y() < -5 and not mam_neg:
            izbrani_vetri.append(veter)
            mam_neg = True



show_tir_leta_dvad(x0, z0, v0, 50)
show_tir_leta_trid(x0, y0, z0, v0, 50, izbrani_vetri)  """


"""





#-----------------------------------------------------------------------------------------------------------
# izdelava vrtečega se grafa
import imageio




def vrteci_tir_leta_trid(x, y, z, v0, phi, vetri):            
    for i in range(360):
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.set_title('3D trajectories')

        for veter_vektor in vetri:
            xos, yos, zos = tir_leta_trid(x, y, z, v0, phi, veter_vektor)
            ax.plot3D(xos, yos, zos)
        
        ax.plot([0, xos.max()], [0, 0], color='black', alpha=0.5, linestyle='dashed')
        plt.scatter(0, 0, s=12, c='black')

        #ax.dist=15
        ax.view_init(elev=35, azim=315 + i/2)
        #ax.text(0, -300, 5000, f'veter {veter_vektor}', color='blue')
        ax.set_xlabel('x [m]', labelpad=15)
        ax.set_ylabel('y [m]', labelpad=15)
        ax.set_zlabel('z [m]', labelpad=15)
        plt.savefig('viz' + f'{i}' + '.png')
        if i % 20 == 0:
            print(i)
        plt.close() 




def create_vrteci_graf(x, y, z, v0, phi, vetri):
    #vrteci_tir_leta_trid(x0, y0, z0, v0, 50, izbrani_vetri)

    slike = []
    with imageio.get_writer('grangif.gif', mode='I') as writer:
        for i in range(90):
            image = imageio.v2.imread('viz' + f'{i}' + '.png')
            writer.append_data(image)
            slike.append(image)

        for j in range(90):
            i = 89-j
            image = imageio.v2.imread('viz' + f'{i}' + '.png')
            writer.append_data(image)
            slike.append(image)
        
    #imageio.mimsave('grangif.gif', slike, format='GIF', duration=1)   # če želiš manipulirati hitrost prehoda med slikami


create_vrteci_graf(x0, y0, z0, v0, 50, izbrani_vetri)

"""