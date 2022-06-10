import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.stats import biweight_location
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.stats import binned_statistic

def logistic_skycoords(galaxyra,galaxydec,galaxycz,galaxymag,galaxygrpid,kval=10,nstar=5):
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    galaxymag=np.array(galaxymag)
    galaxygrpid=np.array(galaxygrpid)
    dtr=np.pi/180.
    galaxyphi = galaxyra * dtr
    galaxytheta = np.pi/2. - galaxydec*dtr
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    groupra = np.zeros_like(galaxyra)
    groupdec = np.zeros_like(galaxydec)
    groupcz = np.zeros_like(galaxycz)
    for gg in np.unique(galaxygrpid):
        sel = np.where(galaxygrpid==gg)
        nmembers = len(sel[0])
        xavg = np.sum(galaxyx[sel]*galaxycz[sel])/nmembers
        yavg = np.sum(galaxyy[sel]*galaxycz[sel])/nmembers
        zavg = np.sum(galaxyz[sel]*galaxycz[sel])/nmembers
        censel = np.argmin(galaxymag[sel])
        xcen = galaxyx[sel][censel]*galaxycz[sel][censel]
        ycen = galaxyy[sel][censel]*galaxycz[sel][censel]
        zcen = galaxyz[sel][censel]*galaxycz[sel][censel]
        
        denom = 1+np.exp(-kval*(nmembers-nstar))
        xada = (xcen-xavg)/denom + xavg
        yada = (ycen-yavg)/denom + yavg
        zada = (zcen-zavg)/denom + zavg

        czada = np.sqrt(xada**2.+yada**2.+zada**2.)
        decada = np.arcsin(zada/czada)*180./np.pi
        if (yada >=0 and xada >=0):
            phicor = 0.0
        elif (yada < 0 and xada < 0):
            phicor = 180.0
        elif (yada >= 0 and xada < 0):
            phicor = 180.0
        elif (yada < 0 and xada >=0):
            phicor = 360.0
        elif (xada==0 and yada==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        raada=np.arctan(yada/xada)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = raada # in degrees
        groupdec[sel] = decada # in degrees
        groupcz[sel] = czada
    return groupra, groupdec, groupcz

def biweight_group_center(galaxyra,galaxydec,galaxycz,galaxygrpid):
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    galaxygrpid=np.array(galaxygrpid)
    dtr=np.pi/180.
    galaxyphi = galaxyra * dtr
    galaxytheta = np.pi/2. - galaxydec*dtr
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    xcen, ycen, zcen = biweight_location(galaxyx*galaxycz), biweight_location(galaxyy*galaxycz), biweight_location(galaxyz*galaxycz)
    czcen = np.sqrt(xcen*xcen + ycen*ycen + zcen*zcen)
    deccen = np.arcsin(zcen/czcen)*(1./dtr)
    if (ycen >=0 and xcen >=0):
        phicor = 0.0
    elif (ycen < 0 and xcen < 0):
        phicor = 180.0
    elif (ycen >= 0 and xcen < 0):
        phicor = 180.0
    elif (ycen < 0 and xcen >=0):
        phicor = 360.0
    elif (xcen==0 and ycen==0):
        print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
    # set up phicorrection and return phicen.
    racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
    return racen,deccen,czcen


def kde_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    galaxygrpid=np.array(galaxygrpid)
    ngalaxies=len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        if len(sel[0])<4:
            groupra[sel]=np.mean(galaxyra[sel])
            groupdec[sel]=np.mean(galaxydec[sel])
            groupcz[sel]=np.mean(galaxycz[sel])
        else:
            nmembers = len(galaxygrpid[sel])
            values = np.array([galaxyra[sel],galaxydec[sel],galaxycz[sel]]).T
            #print(galaxyra[sel],galaxydec[sel],galaxycz[sel])
            #print(values)
            #dens = gaussian_kde(values,'silverman')(values)
            dens = KernelDensity().fit(values).score_samples(values)
            maxdensloc = np.argmax(dens)
            groupra[sel] = galaxyra[sel][maxdensloc] # in degrees
            groupdec[sel] = galaxydec[sel][maxdensloc] # in degrees
            groupcz[sel] = galaxycz[sel][maxdensloc]
    return groupra, groupdec, groupcz


if __name__=='__main__':
    x = np.linspace(0,200,1000)
    for kk in [0.5,1,2,5,10]:
        plt.plot(x, 1/(1+np.exp(-kk*(x-5))), label=str(kk))
    plt.legend(loc='best')
    plt.xlim(0,20)
    plt.show()
    exit()

    df = pd.read_csv("ECOdata_G3catalog_luminosity.csv")
    df = df[(df.g3grp_l>0)]

    df.loc[:,'g3grpn_l']=df.g3grpngi_l+df.g3grpndw_l
    grp1ra,grp1de,grp1cz = logistic_skycoords(df.radeg,df.dedeg,df.cz,df.absrmag,df.g3grp_l, kval=1)
    grp2ra,grp2de,grp2cz = logistic_skycoords(df.radeg,df.dedeg,df.cz,df.absrmag,df.g3grp_l, kval=10)

    plt.figure()
    plt.plot(df.g3grpn_l, df.g3grptcross_l, 'k.')
    median = np.array([np.median(df.g3grptcross_l[df.g3grpn_l==uval]) for uval in np.unique(df.g3grpn_l)])
    plt.plot(np.unique(df.g3grpn_l), median, 'r^')
    plt.xlabel("group N")
    plt.ylabel("crossing time")
    plt.axhline(13.8)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


    kval=3
    grpra,grpde,grpcz = logistic_skycoords(df.radeg,df.dedeg,df.cz,df.absrmag,df.g3grp_l, kval=kval)
    df.loc[:,'adaptive_grpra']=grpra    
    df.loc[:,'adaptive_grpdec']=grpde    
    df.loc[:,'adaptive_grpcz']=grpcz
    df.loc[:,'g3grpn_l']=df.g3grpngi_l+df.g3grpndw_l
    df = df[df.g3fc_l==1.]

    #median_tcross = np.median(df.g3grptcross_l[df.g3grpn_l>1])
    #uniqN = np.unique(df.g3grpn_l)
    #uniqN = uniqN[uniqN>1]
    #frac_vir = np.array([np.sum((df.g3grptcross_l[df.g3grpn_l==nn]<0.25*13.8))/len(df.g3grptcross_l[df.g3grpn_l==nn]) for nn in uniqN])
    #plt.figure()
    #plt.plot(uniqN, frac_vir, 'k.')
    #plt.show()
    #plt.figure()
    #print('median tcross: ', np.median(df.g3grptcross_l[df.g3grpn_l>1]))
    #plt.scatter(df.g3grpn_l, df.g3grptcross_l, s=2, alpha=0.2)
    #plt.show()
    
    plt.figure()
    plt.title(r"$\mathcal{O}_{\rm ad}$ with $k$ = "+str(kval))
    plt.scatter(df.g3grpn_l, df.adaptive_grpra-df.radeg, color='b',s=2, label=r'$\mathcal{O}_{\rm ad} - \mathcal{O}_{\rm cen}$')
    plt.plot(df.g3grpn_l, df.adaptive_grpra-df.g3grpradeg_l, 'r+', markersize=2, label=r'$\mathcal{O}_{\rm ad} - \mathcal{O}_{\rm avg}$')
    eqn = r'$\mathcal{O}_{\rm ad} = \frac{\mathcal{O}_{\rm cen} - \mathcal{O}_{\rm avg}}{1+\exp\left(-k(N_{\rm grp}-5) \right)} + \mathcal{O}_{\rm avg}$'
    plt.annotate(eqn, xy=(15,-0.2), fontsize=16)
    plt.xlim(0,50)
    plt.xlabel(r"$N_{\rm grp}$")
    plt.ylabel("Offset in RA [deg]")
    plt.legend(loc='best')
    plt.show()
 
    plt.figure()
    plt.title(r"$\mathcal{O}_{\rm ad}$ with $k$ = "+str(kval))
    plt.scatter(df.g3grpn_l, df.adaptive_grpdec-df.dedeg, color='b',s=2, label=r'$\mathcal{O}_{\rm ad} - \mathcal{O}_{\rm cen}$')
    plt.plot(df.g3grpn_l, df.adaptive_grpdec-df.g3grpdedeg_l, 'r+', markersize=2, label=r'$\mathcal{O}_{\rm ad} - \mathcal{O}_{\rm avg}$')
    eqn = r'$\mathcal{O}_{\rm ad} = \frac{\mathcal{O}_{\rm cen} - \mathcal{O}_{\rm avg}}{1+\exp\left(-k(N_{\rm grp}-5) \right)} + \mathcal{O}_{\rm avg}$'
    #plt.annotate(eqn, xy=(15,-0.2), fontsize=16)
    plt.xlim(0,50)
    plt.xlabel(r"$N_{\rm grp}$")
    plt.ylabel("Offset in Declination [deg]")
    plt.legend(loc='best')
    plt.show()
    """
    df = df[df.g3grp_l==14.]
    biweight_grpra, biweight_grpdec, _ = biweight_group_center(df.radeg,df.dedeg,df.cz,df.g3grp_l)
    values = np.vstack([np.array(df.radeg),np.array(df.dedeg),np.array(df.cz)])
    dens = gaussian_kde(values)(values)
    kde_grpra, kde_grpdec = df.radeg.to_numpy()[np.argmax(dens)], df.dedeg.to_numpy()[np.argmax(dens)]
    plt.figure()
    plt.scatter(df.radeg,df.dedeg,s=1,color='k')
    plt.plot(df.radeg[df.absrmag==min(df.absrmag)],df.dedeg[df.absrmag==min(df.absrmag)], 'gx', label='BCG')
    plt.plot(df.g3grpradeg_l, df.g3grpdedeg_l, 'rx', label='Arithmetic Mean')
    #plt.plot(np.mean(df.radeg), np.mean(df.dedeg), 'rx', label='Arithmetic Mean')
    #plt.plot(biweight_grpra, biweight_grpdec, 'x', color='purple', label='Biweight')
    plt.plot(kde_grpra, kde_grpdec, 'x', color='purple', label='KDE')
    plt.legend(loc='best')
    plt.show()
    """
    #df=df[df.g3grp_l>0]
    #kdera, kdedec, kdecz = kde_skycoords(df.radeg,df.dedeg,df.cz,df.g3grp_l)
    #df['kdera']=kdera
    #df['kdedec']=kdedec
    #df['kdecz']=kdecz
    #print(df[df.g3grp_l==14.][['g3grpdsProb_l','g3grpadAlpha_l']])


    """
    lowN = (df.g3grpngi_l<5)
    df['adaptivera'] = lowN * df.g3grpradeg_l + (1-lowN) * df.radeg
    df['adaptivedec'] = lowN * df.g3grpdedeg_l + (1-lowN) * df.dedeg

    df=df[df.g3fc_l>0]
    plt.figure()
    plt.axvline(13, color='k',alpha=0.5)
    plt.axvline(13.5,color='k',alpha=0.5)
    plt.scatter(df.g3logmh_l, (df.adaptivedec-df.g3grpdedeg_l).abs()+0.25,s=2,label=r'$|\mathcal{O}_{\rm adaptive}-\mathcal{O}_{\rm avg}|$+0.25')
    plt.scatter(df.g3logmh_l, (df.adaptivedec-df.dedeg).abs()-0.25,s=2, label=r'$|\mathcal{O}_{\rm adaptive}-\mathcal{O}_{\rm central}|$-0.25')
    plt.legend(loc='best')
    plt.xlabel('log halo mass')
    plt.ylabel("On-Sky Separation [deg]")
    plt.show()
    """ 
