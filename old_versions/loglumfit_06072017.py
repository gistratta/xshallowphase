
"""
    Author: Giulia Stratta
This script
 1. Reads data from X-ray afterglow luminosity light curve
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
import math
from mpmath import *
import scipy.special
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import scipy.stats as stats
import os


plt.clf()
plt.close()

outfileold="../output/SGRB_Tt/SGRBoldmodel.txt"
outfilenew="../output/SGRB_Tt/SGRBnewmodel.txt"

if not os.path.isfile(outfileold):
	os.system('touch ../output/SGRB_Tt/SGRBoldmodel.txt')
	out_file = open(outfileold,"a")
	out_file.write("GRB,k,dk,B14,dB14,fmHz,dfmHz,E051,dE051,chi2,dof,p-val"+"\n")
	out_file.close()

if not os.path.isfile(outfilenew):
	os.system('touch ../output/SGRB_Tt/SGRBnewmodel.txt')
	out_file = open(outfilenew,"a")
	out_file.write("GRB,k,dk,B14,dB14,fmHz,dfmHz,E051,dE051,alpha,dalpha,chi2,dof,p-val"+"\n")
	out_file.close()


"""
 1. READ LIGHT CURVES
"""

path = '/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/data/TimeLuminosityLC/'

#datagrb=pd.read_csv(path+"ShortGRB.dat")
#datagrb_array=datagrb.values

fi = raw_input(' grb light curve file [e.g. 050603]: ') or "050603"
#fi = raw_input(' grb light curve file [e.g. '+str(datagrb_array[0])+': ') or "str(datagrb_array[ind])"

filename=path+fi+'TimeLuminosityLC.txt'
data=pd.read_csv(filename,comment='!', sep='\t', header=None,skiprows=None,skip_blank_lines=True)

# trasforma i dati in un ndarray
data_array=data.values

# elimina le righe con NAN alla prima colonna
table=data_array[np.logical_not(np.isnan(data_array[:,0])),:]



# --- Calcola il fattore correttivo per portare la lum in banda 1-10000 keV
E1=1.0
E2=10000
#E1=0.3
#E2=100.0
E01=0.3
E02=10.0

# Read beta and Tstart from file
filein=path+'beta_parameters_and_Tt.dat'
dataparam=pd.read_csv(filein,comment='!', sep='\t', header=None,skiprows=None,skip_blank_lines=True)
dataparam_array=dataparam.values
print "Index for ",fi," : ", np.where(dataparam_array==fi)
datafi=dataparam_array[np.where(dataparam_array[:,0]==fi)]
datafiflat=datafi.flatten()
z=float(datafiflat[1])
beta=float(datafiflat[2])
dbeta=float(datafiflat[3])
Tt=float(datafiflat[5])

# luminosity correction from E01,E02 to E1,E2
K = (E2**(1-beta) - E1**(1-beta))/(E02**(1-beta) - E01**(1-beta))

# ----

# ---- Definisco i vettori logaritmo di luminosita e tempo
logtime=table[:,0]
dlogtime=table[:,1]
loglum=table[:,2]-51+np.log10(K)
dloglum=table[:,3]

# sort times and riassess lum vector
index=np.argsort(logtime)
ltime=logtime[index]
llum=loglum[index]
dllum=dloglum[index]
# -----

"""
 PLOT LIGHT CURVE DATA POINT
"""
En1=str(E1)
En2=str(E2)
plt.title('GRB'+fi+' ['+En1+'-'+En2+' keV]')

#plt.plot(logtime,loglum, color='r')
#plt.errorbar(logtime,loglum,yerr=dloglum,color='r')
plt.plot(ltime,llum,'.', label='data',color='b')
plt.errorbar(ltime, llum, yerr=dllum,fmt='none',ecolor='b')
plt.xlabel('time from trigger [s]')
plt.ylabel('log Luminosity x 10^51 [erg s^-1]')
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

# --- Constants
msun = 1.989                        # units of 10^33 gr
c = 2.99792458                      # units of 10^10 cm/s
r0 = 1.2                            # units of 10 km (10^6 cm)
Ine = 0.35*msun*1.4*(r0)**2         # 1.4 10^45 gr cm^2
#spini = 1.                          # initial spin period in units of ms
# ---

# --- Initial values of model parameters
B = 5.0                             # Magnetic field in units of 10^14 Gauss= 1/(gr*cm*s)
omi = 2*np.pi                       # initial spin frequency 2pi/spini = 6.28 10^3 Hz
E0=1.                               # Initial total energy (10^51 erg ??)
k=0.4                               # k=4*epsilon_e kind of radiative efficiency
# ---

alpha=1.0                         # effective spin down power law index (?)


# --- useful formulas to remind
#Ein = 0.5*Ine*omi**2                            # initial spin energy 27.7 10^51 erg
#tsdi = 3*Ine*c**3/(B**2*(r0)**6*omi**2)*10**5   # Initial spin down time for the standard magnetic dipole formula 3.799*10^6/B2*omi**2 s
#Li=Ein/tsdi                                     # Initial spindown lum. (?) 0.007 10^51 erg/s
#a1 = 3*Ine*c**3/(B**2*(r0)**6*omi**2)*10**5     # tsdi
# ----


# --- in the Notebook but not used in the code
#Lni=Ein/(2./3.*a1)
#tsd=a1*2./3.

#a1=tsdi
#a2=2./(2.-alpha)*(2./3.*a1)
# ----



"""
    DEFINE MODELS
"""

# Read and printout the start time of the plateau from the param_ file as logT in obs. frame
if Tt > 0.0 :
    startTxrtFromFile=(10**Tt)/(1+z)
if Tt == 0.0 :
    startTxrtFromFile=0.0
print("Start time in sec from file:", startTxrtFromFile)

# If the read start time is ok, just retype it, otherwise type the right start time
startTxrt = float(raw_input(' Start time in sec: '))

# Create a time vector with which plot the model
#t0=np.linspace(100.0,1.0e6,10000)
t0=np.logspace(1.,7., num=100, base=10.0)
t1=t0[np.where(t0>startTxrt)]
t=t1[np.where(t1<10**(ltime[-1]))]


# --- Useful formula to remind but not used in the code
#Lsdold1=Ein/(tsdi*(1 + t/tsdi)**2)  # pure dipole radiation spin down lum.
#Lsdold2= Li/(1 + t/a1)**2           # stessa formula di Lsdold1 ma scritta in modo piu semplice

# Initial spin down time for the new formula by C&S06
#tsdnew= (2./3.)*(3*Ine*c**3*10.**5/(r0**6))/(B**2. * omi**2.)   # tsdnew=(2/3)tsdi
#tsdnew= 2./3.*3.7991745075226948*10**6./(B**2. * omi**2.)
#ls=r0**6/(4*c**3*10**5)*B**2*omi**4/(1 + (2 - alpha)/4*r0**6/(Ine*c**3*10**5)*B**2*omi**2*t)**((4 - alpha)/(2 - alpha))

# new spin down lum. expression for alpha2
#ls=r0**6/(4*c**3*10**5)*B**2*omi**4/(1 + (2 - alpha2)/4*r0**6/(Ine*c**3*10**5)*B**2*omi**2*t)**((4 - alpha2)/(2 - alpha2))
# ----


# I modelli li devo definire in funzione di logt per poter fare il fit

def model_old(logt,k,B,omi,E0):
    """
    Description: Energy evolution inside the external shock as due to radiative losses+energy injection
    (Dall'Osso et al. 2011) assuming pure magnetic dipole, as function of
        k = radiative efficiency (0.3)
        B = magnetic field (5. in units of 10^14 Gauss)
        omi = initial spin frequency (2pi)
        E0 = initial ejecta energy (1.)

    NOTA: nelle funzioni che definiscono i modelli, tutti i parametri liberi vanno espressi esplicitamente

    Usage: model_old()
    """
    t=10**logt

    # a1=tsdi
    a1=(3.799*10**6)/(B**2*omi**2)

    # Li=Ein/tsdi  Initial spindown lum. (?) 0.007 10^51 erg/s
    # Ein=0.5*Ine*omi**2 = 0.7*omi**2
    Li=(0.7*B**2*omi**4)/(3.799*10**6)

    hg1_old=scipy.special.hyp2f1(2, 1 + k, 2 + k, -(1/a1))
    hg2_old=scipy.special.hyp2f1(2, 1 + k, 2 + k, -(t/a1))
    f_old=(k/t)*(1/(1 + k))*t**(-k)*(E0 + E0*k - Li*hg1_old + Li*t**(1 + k)*hg2_old)
    return np.log10(f_old)



def model_a(logt,k,B,omi,E0,alpha):
    """
    Description: Energy evolution inside the external shock as due to radiative losses+energy injection introducing the Contopoulos and Spitkovsky 2006 formula assuming alpha05=0.5, as function of
        k = radiative efficiency (0.3)
        B = magnetic field (5. in units of 10^14 Gauss)
        omi = initial spin frequency (2pi)
        E0 = initial ejecta energy (1.)
    """
    t=10**logt

    hg1_a=scipy.special.hyp2f1((4. - alpha)/(2. -alpha), 1. + k, 2.+k, 1.97411*10**(-7)*(alpha-2.)*B**2*omi**2)
    hg2_a=scipy.special.hyp2f1((4. -alpha)/(2. -alpha), 1.+k, 2.+k, 1.97411*10**(-7)*(alpha-2.)*B**2*omi**2*t)
    f_a=(k/t)*(1/(1 + k))*((r0**6)/(4.*c**3.*10**5))*(t**(-k))*(3.6094*10**6*E0 + 3.6094*10**6*E0*k - B**2*omi**4*hg1_a + B**2*omi**4*t**(1 + k)*hg2_a)

    return np.log10(f_a)


# --- TEST: Plotta modello con valori iniziali (prima del fit) come dashed lines:

plt.plot(np.log10(t),model_old(np.log10(t),k,B,omi,E0),'g--')
plt.plot(np.log10(t),model_a(np.log10(t),k,B,omi,E0,alpha),'r--')

#plt.loglog(t,f_a05)
#plt.loglog(t,f_a1)
#plt.xlabel('time from trigger [s]')
#plt.ylabel('Luminosity / 10^51 [erg/s]')
plt.show()



# FIT MODEL ON DATA
#http://www2.mpia-hd.mpg.de/~robitaille/PY4SCI_SS_2014/_static/15.%20Fitting%20models%20to%20data.html

# initial model
#plt.loglog(txrt, model_a05(txrt,k,B,omi,E0),'k--',label='start model')

# old model (2011 paper)
#plt.loglog(t, model_old(t,0.66,12.2,2*np.pi/1.18,1.04),'k--',label='start model')


# NOTA: Fissare Tstart significa fissare anche E0
# E0 puo essere indeterminato se Ein e molto maggiore
# E0/T0 = lumin. senza magnetar
# se Lin>>E0/T0 allora E0 non riesce ad essere determinare



def fitmodelold(model, x, y, dy):

    p0=np.array([k,B,omi,E0])
    popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=([0.001,0.0001,0.001,0.00001], [4.0, 100.,50.0,100.0]))
    #    popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=(0., [1.8, 10.,10.,100.]))
    #popt, pcov = curve_fit(model, x, y, p0, sigma=dy)
    print "------ "
    print "k [",k,"] =", "%.5f" %popt[0], "+/-", "%.5f" %pcov[0,0]**0.5
    print "B [",B,"(10^14 G)]  =", "%.5f" %popt[1], "+/-", "%.5f" %pcov[1,1]**0.5
    print "omi [2pi/spin_i=",omi,"(10^3 Hz)] =", "%.5f" %popt[2], "+/-", "%.5f" %pcov[2,2]**0.5
    print "E0 [",E0,"(10^51 erg)] =", "%.5f" %popt[3], "+/-", "%.5f" %pcov[3,3]**0.5
    print "------  "


    ym=model(x,popt[0],popt[1],popt[2],popt[3])
    print stats.chisquare(f_obs=y,f_exp=ym)
    mychi=sum(((y-ym)**2)/dy**2)
    #mychi=sum(((y-ym)**2)/ym)
    dof=len(x)-len(popt)
    print "my chisquare=",mychi
    print "dof=", dof
    p_value = 1-stats.chi2.cdf(x=mychi,df=dof)
    print "P value",p_value

    bfmodel=model(np.log10(t),popt[0],popt[1],popt[2],popt[3])

    #out_file=open("test.txt","a")
    #    out_file = open(outfileold,"a")
    #out_file.write(fi+","+str("%.5f" %popt[0])+","+str("%.5f" %pcov[0,0]**0.5)+","+str("%.5f" %popt[1])+","+str("%.5f" %pcov[1,1]**0.5)+","+str("%.5f" %popt[2])+","+str("%.5f" %pcov[2,2]**0.5)+","+str("%.5f" %popt[3])+","+str("%.5f" %pcov[3,3]**0.5)+","+str("%.5f" %mychi)+","+str("%.5f" %dof)+","+str("%.5f" %p_value)+"\n")
    #out_file.close()


    return plt.plot(np.log10(t), bfmodel,label='old model')



def fitmodelnew(model, x, y, dy):

    p0=np.array([k,B,omi,E0,alpha])
    popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=([0.001,0.0001,0.001,0.00001,0.0], [4.0, 100.,50.0,100.0,1.0]))
    #    popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=(0., [1.8, 10.,10.,100.]))
    #popt, pcov = curve_fit(model, x, y, p0, sigma=dy)
    print "------ "
    print "k [",k,"] =", "%.5f" %popt[0], "+/-", "%.5f" %pcov[0,0]**0.5
    print "B [",B,"(10^14 G)]  =", "%.5f" %popt[1], "+/-", "%.5f" %pcov[1,1]**0.5
    print "omi [2pi/spin_i=",omi,"(10^3 Hz)] =", "%.5f" %popt[2], "+/-", "%.5f" %pcov[2,2]**0.5
    print "E0 [",E0,"(10^51 erg)] =", "%.5f" %popt[3], "+/-", "%.5f" %pcov[3,3]**0.5
    print "alpha [",alpha,"] =", "%.5f" %popt[4], "+/-", "%.5f" %pcov[4,4]**0.5
    print "------  "

    ym=model(x,popt[0],popt[1],popt[2],popt[3],popt[4])
    print stats.chisquare(f_obs=y,f_exp=ym)
    mychi=sum(((y-ym)**2)/dy**2)
    #mychi=sum(((y-ym)**2)/ym)
    dof=len(x)-len(popt)
    print "my chisquare=",mychi
    print "dof=", dof
    p_value = 1-stats.chi2.cdf(x=mychi,df=dof)
    print "P value",p_value

    bfmodel=model(np.log10(t),popt[0],popt[1],popt[2],popt[3],popt[4])

#    out_file = open(outfilenew,"a")
#    out_file.write(fi+","+str("%.5f" %popt[0])+","+str("%.5f" %pcov[0,0]**0.5)+","+str("%.5f" %popt[1])+","+str("%.5f" %pcov[1,1]**0.5)+","+str("%.5f" %popt[2])+","+str("%.5f" %pcov[2,2]**0.5)+","+str("%.5f" %popt[3])+","+str("%.5f" %pcov[3,3]**0.5)+","+str("%.5f" %popt[4])+","+str("%.5f" %pcov[4,4]**0.5)+","+str("%.5f" %mychi)+","+str("%.5f" %dof)+","+str("%.5f" %p_value)+"\n")
#   out_file.close()

    return plt.plot(np.log10(t), bfmodel,label='new model')



# fitta un modello tra i 2 definiti sui dati logaritmici txrt lxrt

logstartTxrt=np.log10(startTxrt)
txrt=ltime[np.where(ltime>logstartTxrt)]
lxrt=llum[np.where(ltime>logstartTxrt)]
dlxrt=dllum[np.where(ltime>logstartTxrt)]


#def fitold(model):
#    return fitmodelold(model,txrt,lxrt,dlxrt)

#def fitnew(model):
#    return fitmodelnew(model,txrt,lxrt,dlxrt)


#blue
print ' '
print 'model_old'
fitmodelold(model_old,txrt,lxrt,dlxrt)

#orange
print ' '
print 'model_a'
#fitnew(model_a)
fitmodelnew(model_a,txrt,lxrt,dlxrt)


#----TO BE TESTED

#scipy.stats.chisquare(f_obs, f_exp=None, ddof=0, axis=0)

# residual sum of squares
#ss_res = np.sum((lumxrt - model_old(txrt,)) ** 2)
# total sum of squares
#ss_tot = np.sum((y - np.mean(y)) ** 2)
# r-squared
#r2 = 1 - (ss_res / ss_tot)
#-----



# --- Salva il plot nella directory output

plt.legend()
plt.show()
#plt.savefig('../output/SGRB_Tt/'+fi+'.png')
