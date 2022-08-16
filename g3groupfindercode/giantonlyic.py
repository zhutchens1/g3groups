import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import cKDTree
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import os
import subprocess
import sys
from IPython import get_ipython
import foftools as fof
ipython = get_ipython()
from smoothedbootstrap import smoothedbootstrap as sbs
from scipy.optimize import curve_fit
from scipy.stats import mode 

def giantmodel(x, a, b):
    return np.abs(a)*np.log10(np.abs(b)*x+1)

def iterative_combination_giants(galaxyra,galaxydec,galaxycz,giantfofid,rprojboundary,vprojboundary,decisionmode,H0=100.):
    """
    Iteratively combine giant-only FoF groups using group N_giants-based boundaries.

    Parameters
    --------------
    galaxyra : array_like
        RA of giant galaxies in decimal degrees.
    galaxydec : array_like
        Dec of giant galaxies in decimal degrees.
    galaxycz : array_like
        cz of giant galaxies in km/s.
    giantfofid : array_like
        FoF group ID for each giant galaxy, length matches `galaxyra`.
    rprojboundary : callable
        Search boundary to apply on-sky. Should be callable function of group N_giants.
        Units Mpc/h with h being consistent with `H0` argument.
    vprojboundary : callable
        Search boundary to apply in line-of-sight.. Should be callable function of group N_giants.
        Units: km/s
    decisionmode : str
        'allgalaxies' or 'centers'. Specifies how to evaluate whether seed group pairs should be merged. 
    H0 : float
       Hubble constant in (km/s)/Mpc, default 100. 
    
    Returns
    --------------
    giantgroupid : np.array
        Array of group ID numbers following iterative combination. Unique values match that of `giantfofid`.
    """
    centermethod='arithmetic'
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    giantfofid=np.array(giantfofid)
    assert callable(rprojboundary),"Argument `rprojboundary` must callable function of N_giants."
    assert callable(vprojboundary),"Argument `vprojboundary` must callable function of N_giants."

    giantgroupid = np.copy(giantfofid)
    converged=False
    niter=0
    while (not converged):
        print("Giant-only iterative combination {} in progress...".format(niter))
        oldgiantgroupid = giantgroupid
        giantgroupid = nearest_neighbor_assign(galaxyra,galaxydec,galaxycz,oldgiantgroupid,rprojboundary,vprojboundary,centermethod,decisionmode,H0)
        converged = np.array_equal(oldgiantgroupid,giantgroupid)
        niter+=1
    print("Giant-only iterative combiation complete.")
    return giantgroupid


def nearest_neighbor_assign(galaxyra,galaxydec,galaxycz,grpid,rprojboundary,vprojboundary,centermethod,decisionmode,HUBBLE_CONST):
    """
    Refine input group ID by merging nearest-neighbor groups subject to boundary constraints.
    For info on arguments, see "giantonly_iterative combination."

    Returns
    --------------
    refinedgrpid : np.array
        Refined group ID numbers based on nearest-neighbor merging.
    """
    # Prepare output array
    refinedgrpid = deepcopy(grpid)
    # Get the group RA/Dec/cz for every galaxy
    groupra, groupdec, groupcz = fof.group_skycoords(galaxyra, galaxydec, galaxycz, grpid)
    groupN = fof.multiplicity_function(grpid,return_by_galaxy=True)
    # Get unique potential groups
    uniqgrpid, uniqind = np.unique(grpid, return_index=True)
    potra, potdec, potcz = groupra[uniqind], groupdec[uniqind], groupcz[uniqind]
    # Build & query the K-D Tree
    potphi = potra*np.pi/180.
    pottheta = np.pi/2. - potdec*np.pi/180.
    #zmpc = potcz/HUBBLE_CONST
    #xmpc = 2.*np.pi*zmpc*potra*np.cos(np.pi*potdec/180.) / 360.
    #ympc = np.float64(2.*np.pi*zmpc*potdec / 360.)
    zmpc = potcz/HUBBLE_CONST * np.cos(pottheta)
    xmpc = potcz/HUBBLE_CONST*np.sin(pottheta)*np.cos(potphi)
    ympc = potcz/HUBBLE_CONST*np.sin(pottheta)*np.sin(potphi)
    coords = np.array([xmpc, ympc, zmpc]).T
    kdt = cKDTree(coords)
    nndist, nnind = kdt.query(coords,k=2)
    nndist=nndist[:,1] # ignore self match
    nnind=nnind[:,1]

    # go through potential groups and adjust membership for input galaxies
    alreadydone=np.zeros(len(uniqgrpid)).astype(int)
    ct=0
    for idx, uid in enumerate(uniqgrpid):
        # find the nearest neighbor group
        nbridx = nnind[idx]
        Gpgalsel=np.where(grpid==uid)
        GNNgalsel=np.where(grpid==uniqgrpid[nbridx])
        combinedra,combineddec,combinedcz = np.hstack((galaxyra[Gpgalsel],galaxyra[GNNgalsel])),np.hstack((galaxydec[Gpgalsel],galaxydec[GNNgalsel])),np.hstack((galaxycz[Gpgalsel],galaxycz[GNNgalsel]))
        #combinedgroupN = int(groupN[Gpgalsel][0])+int(groupN[GNNgalsel][0])
        combinedgalgrpid = np.hstack((grpid[Gpgalsel],grpid[GNNgalsel]))
        if giants_fit_in_group(combinedra, combineddec, combinedcz, combinedgalgrpid, rprojboundary, vprojboundary, centermethod, decisionmode, HUBBLE_CONST) and (not alreadydone[idx]) and (not alreadydone[nbridx]):
            # check for reciprocity: is the nearest-neighbor of GNN Gp? If not, leave them both as they are and let it be handled during the next iteration.
            nbrnnidx = nnind[nbridx]
            if idx==nbrnnidx:
                # change group ID of NN galaxies
                refinedgrpid[GNNgalsel]=int(grpid[Gpgalsel][0])
                alreadydone[idx]=1
                alreadydone[nbridx]=1
            else:
                alreadydone[idx]=1
        else:
            alreadydone[idx]=1
    return refinedgrpid

def giants_fit_in_group(galra, galdec, galcz, galgrpid, rprojboundary, vprojboundary, center, decisionmode, HUBBLE_CONST):
    """
    Evalaute whether two giant-only groups satisfy the specified boundary criteria.

    Parameters
    --------------------
    galra, galdec, galcz : iterable
        Coordinates of input galaxies -- all galaxies belonging to the pair of groups that are being assessed.
    galgrpid : iterable
        Seed group ID number for each galaxy (should be two unique values).
    totalgrpn : int
        Total group N (group N as if two seed groups were a single giant-only group).
    rprojboundary : callable
        Search boundary to apply on-sky. Should be callable function of group N_giants.
        Units Mpc/h with h being consistent with `H0` argument.
    vprojboundary : callable
        Search boundary to apply in line-of-sight.. Should be callable function of group N_giants.
        Units: km/s
    centermethod : str
        'arithmetic' or 'luminosity'. Specifies how to propose group centers during the combination process.
    decisionmode : str
        'allgalaxies' or 'centers'. Specifies how to evaluate whether seed group pairs should be merged.
    HUBBLE_CONST : float
       Hubble constant in (km/s)/Mpc, default 100. 
    """
    if decisionmode=='allgalaxies':
        grpn = len(galra)
        galphi = galra*np.pi/180.
        galtheta = np.pi/2. - galdec*np.pi/180.
        galx = np.sin(galtheta)*np.cos(galphi)
        galy = np.sin(galtheta)*np.sin(galphi)
        galz = np.cos(galtheta)
        if center=='arithmetic':
            xcen = np.sum(galcz*galx)/grpn
            ycen = np.sum(galcz*galy)/grpn
            zcen = np.sum(galcz*galz)/grpn
            czcenter = np.sqrt(xcen**2+ycen**2+zcen**2)
            deccenter = np.arcsin(zcen/czcenter)*(180.0/np.pi)
            phicorr = 0.0*int((ycen >=0 and xcen >=0)) + 180.0*int((ycen < 0 and xcen < 0) or (ycen >= 0 and xcen < 0)) + 360.0*int((ycen < 0 and xcen >=0))
            racenter = np.arctan(ycen/xcen)*(180.0/np.pi)+phicorr
        #elif center=='luminosity':
        #    unlogmag = 10**(-0.4*galmag)
        #    xcen = np.sum(galcz*galx*unlogmag)/np.sum(unlogmag)
        #    ycen = np.sum(galcz*galy*unlogmag)/np.sum(unlogmag)
        #    zcen = np.sum(galcz*galz*unlogmag)/np.sum(unlogmag)
        #    czcenter = np.sqrt(xcen**2+ycen**2+zcen**2)
        #    deccenter = np.arcsin(zcen/czcenter)*(180.0/np.pi)
        #    phicorr = 0.0*int((ycen >=0 and xcen >=0)) + 180.0*int((ycen < 0 and xcen < 0) or (ycen >= 0 and xcen < 0)) + 360.0*int((ycen < 0 and xcen >=0))
        #    racenter = np.arctan(ycen/xcen)*(180.0/np.pi)+phicorr
        # check if all members are within rproj and vproj of group center
        halfangle = fof.angular_separation(racenter,deccenter,galra[:,None],galdec[:,None])/2.0
        rprojsep = (galcz[:,None]+czcenter)/HUBBLE_CONST * halfangle
        lossep = np.abs(galcz[:,None]-czcenter)
        fitingroup=(np.all(rprojsep<rprojboundary(grpn)) and np.all(lossep<vprojboundary(grpn)))
    elif decisionmode=='centers':
        uniqIDnums = np.unique(galgrpid)
        assert len(uniqIDnums)==2, "galgrpid must have two unique entries (two seed groups)."
        seed1sel = (galgrpid==uniqIDnums[0])
        seed1grpra,seed1grpdec,seed1grpcz = fof.group_skycoords(galra[seed1sel],galdec[seed1sel],galcz[seed1sel],galgrpid[seed1sel])
        seed2sel = (galgrpid==uniqIDnums[1])
        seed2grpra,seed2grpdec,seed2grpcz = fof.group_skycoords(galra[seed2sel],galdec[seed2sel],galcz[seed2sel],galgrpid[seed2sel])
        halfangle = fof.angular_separation(seed1grpra[0],seed1grpdec[0],seed2grpra[0],seed2grpdec[0])/2.
        rprojsep = (seed1grpcz[0]+seed2grpcz[0])/HUBBLE_CONST * np.sin(halfangle)
        lossep = np.abs(seed1grpcz[0]-seed2grpcz[0])
        totalgrpN = len(seed1grpra)+len(seed2grpra)
        fitingroup=((rprojsep<rprojboundary(totalgrpN).all()) and (lossep<vprojboundary(totalgrpN)).all())
    else:
        assert False, "Function argument `decisionmode` must be either `allgalaxies` or `centers`."
        sys.exit()
    return fitingroup


if __name__=='__main__':
    eco = pd.read_csv("../ECOdata_080822.csv")
    eco = eco[(eco.absrmag<-19.5)]
   
    # giant-only FoF, derivation of boundaries for ic 
    giantfofid = fof.fast_fof(eco.radeg,eco.dedeg,eco.cz,0.07,1.1,3.39)
    ecogiantgrpn = fof.multiplicity_function(giantfofid,return_by_galaxy=True)
    ecogiantgrpra, ecogiantgrpdec, ecogiantgrpcz = fof.group_skycoords(np.array(eco.radeg), np.array(eco.dedeg), np.array(eco.cz), giantfofid)
    relvel = np.abs(ecogiantgrpcz - np.array(eco.cz))
    relprojdist = (ecogiantgrpcz + np.array(eco.cz))/100. * fof.angular_separation(ecogiantgrpra, ecogiantgrpdec, np.array(eco.radeg), np.array(eco.dedeg))/2.0
    ecogiantgrpn = fof.multiplicity_function(giantfofid, return_by_galaxy=True)
    uniqecogiantgrpn, uniqindex = np.unique(ecogiantgrpn, return_index=True)
    keepcalsel = np.where(uniqecogiantgrpn>1)

    median_relprojdist = np.array([np.median(relprojdist[np.where(ecogiantgrpn==sz)]) for sz in uniqecogiantgrpn[keepcalsel]])
    median_relvel = np.array([np.median(relvel[np.where(ecogiantgrpn==sz)]) for sz in uniqecogiantgrpn[keepcalsel]])

    rproj_median_error = np.std(np.array([sbs(relprojdist[np.where(ecogiantgrpn==sz)], 10000, np.median, kwargs=dict({'axis':1 })) for sz in uniqecogiantgrpn[keepcalsel]]), axis=1)
    dvproj_median_error = np.std(np.array([sbs(relvel[np.where(ecogiantgrpn==sz)], 10000, np.median, kwargs=dict({'axis':1})) for sz in uniqecogiantgrpn[keepcalsel]]), axis=1)

    poptrproj,cov1 = curve_fit(giantmodel, uniqecogiantgrpn[keepcalsel], median_relprojdist, sigma=rproj_median_error)#, p0=[0.1, -2, 3, -0.1])
    poptdvproj,cov2 = curve_fit(giantmodel, uniqecogiantgrpn[keepcalsel], median_relvel, sigma=dvproj_median_error)#, p0=[160,6.5,45,-600])
    print("Giant model params.", poptrproj, poptdvproj)
    print("errors: ",np.sqrt(np.diag(cov1)),np.sqrt(np.diag(cov2)))
    rproj_boundary = lambda N: 3.5*giantmodel(N, *poptrproj) #3*(rprojslope*N+rprojint)
    vproj_boundary = lambda N: 6.5*giantmodel(N, *poptdvproj) #4.5*(dvprojslope*N+dvprojint)
    assert ((rproj_boundary(1)>0) and (vproj_boundary(1)>0)), "Cannot extrapolate Rproj or Vproj to N=1"

    # iterative combination, giants
    itergiantfofid = iterative_combination_giants(eco.radeg,eco.dedeg,eco.cz,giantfofid,rproj_boundary,vproj_boundary,decisionmode='centers',H0=100.)

    if False:
        plt.figure()
        binv = np.arange(0.5,500.5,1)
        plt.hist(fof.multiplicity_function(giantfofid), bins=binv, log=True, histtype='step', color='k',hatch='//',label='Giant-only FoF')
        plt.hist(fof.multiplicity_function(itergiantfofid), bins=binv, log=True, histtype='step', color='orange', linewidth=3, label='Giant-Only FoF + Iter. Combination')
        plt.legend(loc='best')
        plt.xlabel("Number of Giant Galaxies per Group")
        plt.ylabel("Number of Giant-Only Groups")
        plt.show() 

    coma_id,_ = mode(giantfofid)
    print(coma_id) 
    sel = (itergiantfofid==coma_id)
    plt.figure()
    coma_reg_fofids = giantfofid[sel]
    for uid in np.unique(coma_reg_fofids):
        regsel = (giantfofid==uid)
        plt.plot(eco.radeg[regsel],eco.dedeg[regsel],'s',label='Original FoF groups (now merged)',alpha=0.5)
    plt.plot(eco.radeg[sel],eco.dedeg[sel], 'k.', label='Group 14 after iterative combination')
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.gca().invert_xaxis()
    plt.legend(loc='best')
    plt.show()
