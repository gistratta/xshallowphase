
"""
    Author: Giulia Stratta
This script
 1. Read data from X-ray afterglow luminosity light curve
 2. Define model: ejecta energy variation with time assuming energy injection (from Simone Notebook)
 3. Fit model to the data and compute best fit param and cov. matrix
 4. Plot best fit model on data

 Per lanciare lo script: 
    ipython --pylab
    import script
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpmath import *
import scipy.special


"""
 1. READ LIGHT CURVES
"""

#filename='grb120102flux.qdp'
#data=pd.read_csv(filename,comment='!', sep='\t', na_values='NO',header=None,skiprows=13,skip_blank_lines=True)

filename='050319_lumk.qdp'
data=pd.read_csv(filename,comment='!', sep=' ', header=None,skiprows=4,skip_blank_lines=True)

# trasforma i dati in un ndarray
data_array=data.values

# elimina le righe con NAN alla prima colonna
table=data_array[np.logical_not(np.isnan(data_array[:,0])),:]

time=table[:,0]
dtimep=table[:,1]
dtimem=table[:,2]
lum=table[:,3]*0.001
dlum=table[:,4]*0.001

"""
 PLOT LIGHT CURVE
"""

# Se uso il Notebook
#%matplotlib inline

#plt.loglog(time,lum,'.')
#plt.xlabel('time from trigger [s]')
#plt.ylabel('Luminosity x 10^51 [erg cm^-2 s^-1]')
#plt.ylabel('0.3-10 keV flux [erg cm^-2 s^-1]')
#plt.show()


"""
 2. DEFINE MODEL'S PARAMETERS
 
 Simone's NOTE:
 
 spini = initial spin period [in millisecond],
 B = magnetic dipole field [in units of 10^(14)G],
 r0 = typical NS radius [in units of 10 km],
 Ine = moment of inertia of the NS [in units of 10^(45)].
 
 In principle, all of these have values that are not well determined.
 HOWEVER:
 A) r0 and Ine have only minor uncertainties. The NS radius is assumed here to be 12 km, hence the radius is always written as 1.2.
 The coefficient 0.35 in the moment of inertia is somewhat depenedent on the NS EOS. 
 Both the coefficient 1.2 and 0.35 can only vary in a restricted range, and are considered as given constants here.
 B) A different argument holds for spini AND B. These are actual free parameters, that should be determined by the fits.
 In this first part I fix their values in order to produce plots of the relevant quantities,
 that will be needed to compare between the "old", magnetic dipole formula,
 and the "new" formula, proposed by Contopoulos Spitkovsky 2006).
 (*ADDITIONAL UNITS: tsdi is in units of seconds; Ein,the initial spin energy, is in units of 10^(51) ergs.)
    
"""

# set decimal places
mp.dps = 25 
# Setting the mp.pretty option will use the str()-style output for repr() as well
mp.pretty = True

msun = 1.989                        # units of 10^28 kg
c = 2.99792458                      # units of 10^5 km/s
r0 = 1.2                            # units of 10 km
Ine = 0.35*msun*1.4*(r0)**2         # units of 10^45 kg km^2


spini = 1.                          # initial spin period in units of ms
omi = 2*np.pi/spini                 # initial spin frequency in units of 10^3 Hz
Ein=0.5*Ine*omi**2                  # initial spin energy in units of 10^51 erg

B = 5.                              # Magnetic field in units of 10^14 Gauss
tsdi = 3*Ine*c**3/(B**2*(r0)**6*omi**2)*10**5   # Initial spin down time for the standard magnetic dipole formula in units of seconds
Li=Ein/tsdi                         # Initial spindown lum. (?) 10^51 erg/s
E0=1.                               # Initial kinetic energy of the ejecta

k=0.4                               # k=4*epsilon_e kind of radiative efficiency
alpha=0.5                           # effective spin down power law index (?)
alpha1=0.5
alpha2=1.

a1=tsdi
a2=2./(2.-alpha)*2./3.*a1
Lni=Ein/(2./3.*a1)
tsd=a1*2./3.

print('Initial Iput values')
print('B = ',B,'omi = ',omi,' k = ',k,'E0 = ', E0, 'tsdi = ',tsdi)
print('  ')



"""
    DEFINE MODELS
"""

t=np.linspace(10,500000,10000)

Lsdold1=Ein/(tsdi*(1 + t/tsdi)**2)
Lsdold2= Li/(1 + t/a1)**2

#for z in time:
#    hg2=hyp2f1(2, 1 + k, 2 + k, -(z/a1))
#    print(hg2)


def model_a05_old(t,k,B,omi,E0):
    """
    Description: Energy evolution inside the external shock as due to radiative losses+energy injection 
    (Dall'Osso et al. 2011) assuming pure magnetic dipole, alpha1=0.5, as function of
        k = radiative efficiency (0.3)
        B = magnetic field (5. in units of 10^14 Gauss)
        omi = initial spin frequency (2pi)
        E0 = initial ejecta energy (1.)
    
    Usage: model_a05_old()
    """
    tsdi = 3*Ine*c**3/(B**2*(r0)**6*omi**2)*10**5
    Ein = 0.5*Ine*omi**2
    Li = Ein/tsdi
    hg1_a05_old=scipy.special.hyp2f1(2, 1 + k, 2 + k, -(1/a1))
    hg2_a05_old=scipy.special.hyp2f1(2, 1 + k, 2 + k, -(t/a1))
    f_a05_old=(k/t)*(1/(1 + k))*t**(-k)*(E0 + E0*k - Li*hg1_a05_old + Li*t**(1 + k)*hg2_a05_old)
    return f_a05_old

#flist=[]
#for z in time:
#    f0=float((k/z)*(1/(1 + k))*z**(-k)*(E0 + E0*k - Li*hg1 + Li*z**(1 + k)*hyp2f1(2, 1 + k, 2 + k, -(z/a1))))
#    flist=flist.append(f0)

#f=np.array(flist)



tsdnew= 2./3.*3.7991745075226948*10**6./(B**2. * omi**2.)
ls=r0**6/(4*c**3*10**5)*B**2*omi**4/(1 + (2 - alpha)/4*r0**6/(Ine*c**3*10**5)*B**2*omi**2*t)**((4 - alpha)/(2 - alpha))

#glist=[]
#for z in time:
#    g0=float((k/z)*(1./(1. + 1.*k))*(z**(-1.*k))*(3.6094*10**6*E0 + 3.6094*10**6*E0*k - 1.*B**2*omi**4*hg1 + B**2*omi**4*z**(1. + 1.*k)*hyp2f1((4.-alpha)/(2.-alpha), 1.+k, 2.+k, 1.97411*10**(-7)*(-2.+alpha)*B**2 * omi**2 * z)))
#    glist.append(g0)
#g=np.array(glist)

def model_a05(t,k,B,omi,E0):
    """
    Description: Energy evolution inside the external shock as due to radiative losses+energy injection introducing the Contopoulos and Spitkovsky 2006 formula assuming alpha=0.5, as function of
        k = radiative efficiency (0.3)
        B = magnetic field (5. in units of 10^14 Gauss)
        omi = initial spin frequency (2pi)
        E0 = initial ejecta energy (1.)
        
        Usage: model_a05()
     """
    
    hg1_a05=scipy.special.hyp2f1((4. - 1. *alpha)/(2. - 1. *alpha), 1. + 1.*k, 2. + 1. *k, 1.97411*10**(-7)*(-2. + 1.* alpha) * B**2 * omi**2)
    hg2_a05=scipy.special.hyp2f1((4.-alpha)/(2.-alpha), 1.+k, 2.+k, 1.97411*10**(-7)*(-2.+alpha)*B**2 * omi**2 * t)
    f_a05=(k/t)*(1/(1 + k))*((r0**6)/(4*c**3*10**5))*(t**(-k))*(3.6094*10**6*E0 + 3.6094*10**6*E0*k - B**2*omi**4*hg1_a05 + B**2*omi**4*t**(1 + k)*hg2_a05)
    return f_a05




ls=r0**6/(4*c**3*10**5)*B**2*omi**4/(1 + (2 - alpha2)/4*r0**6/(Ine*c**3*10**5)*B**2*omi**2*t)**((4 - alpha2)/(2 - alpha2))

#glist=[]
#for z in time:
#    g0=float((k/z)*(1./(1. + 1.*k))*(z**(-1.*k))*(3.6094*10**6*E0 + 3.6094*10**6*E0*k - 1.*B**2*omi**4*hg1 + B**2*omi**4*z**(1. + 1.*k)*hyp2f1((4.-alpha)/(2.-alpha), 1.+k, 2.+k, 1.97411*10**(-7)*(-2.+alpha)*B**2 * omi**2 * z)))
#    glist.append(g0)
#g=np.array(glist)

def model_a1(t,k,B,omi,E0):
    """
        Description: Energy evolution inside the external shock as due to radiative losses+energy injection introducing the Contopoulos and Spitkovsky 2006 formula assuming alpha2=1.0, as function of
        k = radiative efficiency (0.3)
        B = magnetic field (5. in units of 10^14 Gauss)
        omi = initial spin frequency (2pi)
        E0 = initial ejecta energy (1.)
        
        Usage: model_a1()
    """


    hg1_a1=scipy.special.hyp2f1((4. - alpha2)/(2. - alpha2), 1. + k, 2. + k, 1.97411*10**(-7)*(-2. + alpha2) * B**2 * omi**2)
    hg2_a1=scipy.special.hyp2f1((4.-alpha2)/(2.-alpha2), 1.+k, 2.+k, 1.97411*10**(-7)*(-2.+alpha2)*B**2 * omi**2 * t)
    f_a1=(k/t)*(1/(1 + k))*((r0**6)/(4*c**3*10**5))*(t**(-k))*(3.6094*10**6*E0 + 3.6094*10**6*E0*k - B**2*omi**4*hg1_a1 + B**2*omi**4*t**(1 + k)*hg2_a1)
    return f_a1



#plt.loglog(t,f_a05_old)
#plt.loglog(t,f_a05)
#plt.loglog(t,f_a1)
#plt.xlabel('time from trigger [s]')
#plt.ylabel('Luminosity / 10^51 [erg/s]')
#plt.show()





# FIT MODEL ON DATA
#http://www2.mpia-hd.mpg.de/~robitaille/PY4SCI_SS_2014/_static/15.%20Fitting%20models%20to%20data.html

from scipy.optimize import curve_fit
popt, pcov = curve_fit(model_a05_old, time, lum, sigma=dlum)

print "k =", popt[0], "+/-", pcov[0,0]**0.5
print "B =", popt[1], "+/-", pcov[1,1]**0.5
print "omi =", popt[2], "+/-", pcov[2,2]**0.5
print "E0 =", popt[3], "+/-", pcov[3,3]**0.5




# PLOT DATA AND BEST FIT MODEL

plt.loglog(time,lum,'.')
plt.xlabel('time from trigger [s]')
plt.ylabel('Luminosity x 10^51 [erg cm^-2 s^-1]')
plt.loglog(t, model_a05_old(t,popt[0],popt[1],popt[2],popt[3]),'r-')
plt.show()






