
"""
    Author: Giulia Stratta
This script [  --- up to 10/5/2017 IS ONLY FOR 061121 ----]
 1. Reads data from X-ray afterglow flux light curve
 2. Defines model: ejecta energy variation with time assuming energy injection (from Simone Notebook)
 3. Fits model to the data and compute best fit param and cov. matrix
 4. Plots best fit model on data
 5. Saves plot in ../output

 Per lanciare lo script: 
    ipython --pylab
    run lumfit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpmath import *
import scipy.special
from scipy.optimize import curve_fit
from scipy.stats import chisquare


plt.clf()
plt.close()

"""
 1. READ LIGHT CURVES
"""

#filename='grb120102flux.qdp'
#data=pd.read_csv(filename,comment='!', sep='\t', na_values='NO',header=None,skiprows=13,skip_blank_lines=True)

path = '/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/LUMFIT/SNR4/'
fi = raw_input(' grb light curve file [e.g. 050315]: ')
filename=path+fi+'.txt'
data=pd.read_csv(filename,comment='!', sep='\t', header=None,skiprows=None,skip_blank_lines=True)

# trasforma i dati in un ndarray
data_array=data.values

# elimina le righe con NAN alla prima colonna
table=data_array[np.logical_not(np.isnan(data_array[:,0])),:]

time0=table[:,0]
dtimep=table[:,1]
dtimem=table[:,2]
flux=table[:,3]
dfluxp=table[:,4]
dfluxm=table[:,5]

#---- Questi valori andrebbero letti nel file beta_parameters_and_Tt.dat
# 061121
z=1.314
a=0.97
#-----
# ...e DL calcolato!
# Questo usa i param. cosm. del paper:
DLq48=(9076.8*3.08568)**2.0


E01=0.3
E02=10.0
#E1=1.0
#E2=10000.0
E1=0.3
E2=100.0

#Kcorr=((E2**(1-a)-E1**(1-a))/(E02**(1-a)-E01**(1-a)))/(1+z)**(1-a)
Kcorr=((E2**(1-a)-E1**(1-a))/(E02**(1-a)-E01**(1-a)))
kzcorr=(1+z)**(1-a)

"""
# F[0.3,100]=F[0.3,10]*Kcorr
# F[0.3/(1+z),100/(1+z)] = F[0.3,100]/(1+z)**(1-a)
# L[0.3,100]=4*pi*DL^2*(F[0.3,10]*Kcorr/(1+z)^(1-a))
"""

Kdist51=4*np.pi*DLq48*1.e-3

time=time0/(1+z)
dtime=((dtimep-dtimem)/2.0)/(1+z)
dflux=(dfluxp-dfluxm)/2.0
lum=flux*Kdist51*Kcorr/kzcorr
dlum=dflux*Kdist51*Kcorr/kzcorr

"""
 PLOT LIGHT CURVE DATA POINT
"""

plt.title(fi)

plt.loglog(time,lum,'.',label='data')
plt.xlabel('time from trigger [s]')
plt.ylabel('Luminosity x 10^51 [erg s^-1]')
plt.show()

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

msun = 1.989                        # units of 10^33 gr
c = 2.99792458                      # units of 10^10 cm/s
r0 = 1.2                            # units of 10 km (10^6 cm)
Ine = 0.35*msun*1.4*(r0)**2         # units of 10^45 gr cm^2

spini = 1.                          # initial spin period in units of ms
omi = 2*np.pi/spini                 # initial spin frequency in units of 10^3 Hz
Ein=0.5*Ine*omi**2                  # initial spin energy in units of 10^51 erg

B = 5.0                             # Magnetic field in units of 10^14 Gauss= 1/(gr*cm*s)
tsdi = 3*Ine*c**3/(B**2*(r0)**6*omi**2)*10**5   # Initial spin down time for the standard magnetic dipole formula in units of seconds
Li=Ein/tsdi                         # Initial spindown lum. (?) 10^51 erg/s
E0=1.                               # Initial total energy (10^51 erg ??)

k=0.4                               # k=4*epsilon_e kind of radiative efficiency
alpha=0.5                           # effective spin down power law index (?)
#alpha1=0.5
alpha2=1.


a1=tsdi
a2=2./(2.-alpha)*2./3.*a1
Lni=Ein/(2./3.*a1)
tsd=a1*2./3.




"""
    DEFINE MODELS
"""

startTxrt = float(raw_input(' XRT start time in sec : '))

#t0=np.linspace(100.0,1.0e6,10000)
t0=np.logspace(1.,7., num=100, base=10.0)
t=t0[np.where(t0>startTxrt)]

Lsdold1=Ein/(tsdi*(1 + t/tsdi)**2)  # pure dipole radiation spin down lum.
Lsdold2= Li/(1 + t/a1)**2           # stessa formula di Lsdold1 ma scritta in modo piu semplice


# Initial spin down time for the new formula by C&S06
tsdnew= 2./3.*3.7991745075226948*10**6./(B**2. * omi**2.)
ls=r0**6/(4*c**3*10**5)*B**2*omi**4/(1 + (2 - alpha)/4*r0**6/(Ine*c**3*10**5)*B**2*omi**2*t)**((4 - alpha)/(2 - alpha))

# new spin down lum. expression for alpha2
#ls=r0**6/(4*c**3*10**5)*B**2*omi**4/(1 + (2 - alpha2)/4*r0**6/(Ine*c**3*10**5)*B**2*omi**2*t)**((4 - alpha2)/(2 - alpha2))


txrt=time[np.where(time>startTxrt)]
lxrt=lum[np.where(time>startTxrt)]
dlxrt=dlum[np.where(time>startTxrt)]


def model_old(t,k,B,omi,E0):
    """
    Description: Energy evolution inside the external shock as due to radiative losses+energy injection 
    (Dall'Osso et al. 2011) assuming pure magnetic dipole, as function of
        k = radiative efficiency (0.3)
        B = magnetic field (5. in units of 10^14 Gauss)
        omi = initial spin frequency (2pi)
        E0 = initial ejecta energy (1.)
    
    Usage: model_old()
    """
    tsdi = 3*Ine*c**3/(B**2*(r0)**6*omi**2)*10**5
    Ein = 0.5*Ine*omi**2
    Li = Ein/tsdi
    a1=tsdi
    hg1_old=scipy.special.hyp2f1(2, 1 + k, 2 + k, -(1/a1))
    hg2_old=scipy.special.hyp2f1(2, 1 + k, 2 + k, -(t/a1))
    f_old=(k/t)*(1/(1 + k))*t**(-k)*(E0 + E0*k - Li*hg1_old + Li*t**(1 + k)*hg2_old)
    return f_old



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
    alpha=0.5
    hg1_a05=scipy.special.hyp2f1((4. - 1. *alpha)/(2. - 1. *alpha), 1. + 1.*k, 2. + 1. *k, 1.97411*10**(-7)*(-2. + 1.* alpha) * B**2 * omi**2)
    hg2_a05=scipy.special.hyp2f1((4.-alpha)/(2.-alpha), 1.+k, 2.+k, 1.97411*10**(-7)*(-2.+alpha)*B**2 * omi**2 * t)
    f_a05=(k/t)*(1/(1 + k))*((r0**6)/(4*c**3*10**5))*(t**(-k))*(3.6094*10**6*E0 + 3.6094*10**6*E0*k - B**2*omi**4*hg1_a05 + B**2*omi**4*t**(1 + k)*hg2_a05)
    return f_a05





def model_a1(t,k,B,omi,E0):
    """
        Description: Energy evolution inside the external shock as due to radiative losses+energy injection introducing the Contopoulos and Spitkovsky 2006 formula assuming alpha2=1.0, as function of
        k = radiative efficiency (0.3)
        B = magnetic field (5. in units of 10^14 Gauss)
        omi = initial spin frequency (2pi)
        E0 = initial ejecta energy (1.)
        
        Usage: model_a1()
    """
    alpha=1.0
    hg1_a1=scipy.special.hyp2f1((4. - alpha)/(2. - alpha), 1. + k, 2. + k, 1.97411*10**(-7)*(-2. + alpha) * B**2 * omi**2)
    hg2_a1=scipy.special.hyp2f1((4.-alpha)/(2.-alpha), 1.+k, 2.+k, 1.97411*10**(-7)*(-2.+alpha)*B**2 * omi**2 * t)
    f_a1=(k/t)*(1/(1 + k))*((r0**6)/(4*c**3*10**5))*(t**(-k))*(3.6094*10**6*E0 + 3.6094*10**6*E0*k - B**2*omi**4*hg1_a1 + B**2*omi**4*t**(1 + k)*hg2_a1)
    return f_a1

# plotta modello con valori iniziali (prima del fit)
#plt.loglog(t,model_old(t,k,B,omi,E0),'g--')
#plt.loglog(t,model_a05(t,k,B,omi,E0),'r--')
#plt.loglog(t,model_a1(t,k,B,omi,E0),'b--')



#plt.loglog(t,f_a05)
#plt.loglog(t,f_a1)
#plt.xlabel('time from trigger [s]')
#plt.ylabel('Luminosity / 10^51 [erg/s]')
#plt.show()



# FIT MODEL ON DATA
#http://www2.mpia-hd.mpg.de/~robitaille/PY4SCI_SS_2014/_static/15.%20Fitting%20models%20to%20data.html

# initial model
#plt.loglog(txrt, model_old(txrt,k,B,omi,E0),'k--',label='2011 model')
plt.loglog(t, model_old(t,0.66,12.1,2.0*np.pi/1.18,1.0),'k--',label='2011 model')



def fitmodel(model, x, y, dy):
    #t,k,B,omi,E0
    p0=np.array([k,B,omi,E0])
    popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=(0., [4., 10., 100.,600.]))
    #    popt, pcov = curve_fit(model, x, y, sigma=dy)
    print " "
    print "k [",k,"] =", popt[0], "+/-", pcov[0,0]**0.5
    print "B [",B,"(10^14 G)]  =", popt[1], "+/-", pcov[1,1]**0.5
    print "omi [2pi/spin_i=",omi,"(10^3 Hz)] =", popt[2], "+/-", pcov[2,2]**0.5
    print "E0 [",E0,"(10^51 erg)] =", popt[3], "+/-", pcov[3,3]**0.5
    #ss_res = np.sum((lumxrt - model(txrt,popt[0],popt[1],popt[2],popt[3])) ** 2)
    #ss_tot = np.sum((lumxrt - np.mean(lumxrt))**2
    #r2 = 1. - (ss_res / ss_tot)
    return plt.loglog(t, model(t,popt[0],popt[1],popt[2],popt[3]),label='fit model')
#    plt.show()
#    return popt,pcov
#    return plt.show()



def fit(model):
    return fitmodel(model,txrt,lxrt,dlxrt)


#print('Initial Iput values')
#print('B = ',B,'omi = ',omi,' k = ',k,'E0 = ', E0, 'tsdi = ',tsdi)
#print('  ')


#green
print ' '
print 'model_old'
fit(model_old)

#----TO BE TESTED

#scipy.stats.chisquare(f_obs, f_exp=None, ddof=0, axis=0)

# residual sum of squares
#ss_res = np.sum((lumxrt - model_old(txrt,)) ** 2)
# total sum of squares
#ss_tot = np.sum((y - np.mean(y)) ** 2)
# r-squared
#r2 = 1 - (ss_res / ss_tot)
#-----

#red
print ' '
print 'model_a05'
fit(model_a05)

#cyan
print ' '
print 'model_a1'
fit(model_a1)

plt.legend()
plt.show()


#savefig = raw_input(' Save figure? [Yes=1/No=0]: ')
#if savefig == 1 :
#plt.savefig('../output/'+fi+'.png')




