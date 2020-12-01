import numpy as np
import pandas as pd
import pickle
from scipy.spatial import cKDTree
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import os
import subprocess

from IPython import get_ipython
ipython = get_ipython()

HUBBLE_CONST = 100.

"""
Zackary Huthens - zhutchen [at] live.unc.edu
University of North Carolina at Chapel Hill
"""

def iterative_combination(galaxyra, galaxydec, galaxycz, galaxymag, rprojboundary, vprojboundary, centermethod, starting_id=1):
    """
    Perform iterative combination on a list of input galaxies.
    
    Parameters
    ------------
    galaxyra, galaxydec : iterable
       Right-ascension and declination of the input galaxies in decimal degrees.
    galaxycz : iterable
       Redshift velocity (corrected for Local Group motion) of input galaxies in km/s.
    galaxymag : iterable
       M_r absolute magnitudes of input galaxies, or galaxy stellar/baryonic masses (the code will be able to differentiate.)
    rprojboundary : callable
       Search boundary to apply in projection on the sky for grouping input galaxies, function of group-integrated M_r, in units Mpc/h.
    vprojboundary : callable
       Search boundary to apply in velocity  on the sky for grouping input galaxies, function of group-integrated M_r or mass, in units km/s.
    centermethod : str
        'arithmetic' or 'luminosity'. Specifies how to propose group centers during the combination process.
    starting_id : int, default 1
       Base ID number to assign to identified groups (all group IDs will be >= starting_id).
    
    Returns
    -----------
    itassocid: Group ID numbers for every input galaxy. Shape matches `galaxyra`.
    """
    print("Beginning iterative combination...")
    # Check user input
    assert (callable(rprojboundary) and callable(vprojboundary)),"Inputs `rprojboundary` and `vprojboundary` must be callable."
    assert (len(galaxyra)==len(galaxydec) and len(galaxydec)==len(galaxycz)),"RA/Dec/cz inputs must have same shape."
    # Convert everything to numpy + create ID array (assuming for now all galaxies are isolated)
    galaxyra = np.array(galaxyra)
    galaxydec = np.array(galaxydec)
    galaxycz = np.array(galaxycz)
    galaxymag = np.array(galaxymag)
    itassocid = np.arange(starting_id, starting_id+len(galaxyra))
    # Begin algorithm. 
    converged=False
    niter=0
    while (not converged):
        print("iteration {} in progress...".format(niter))
        # Compute based on updated ID number
        olditassocid = itassocid
        itassocid = nearest_neighbor_assign(galaxyra, galaxydec, galaxycz, galaxymag, olditassocid, rprojboundary, vprojboundary, centermethod)
        # check for convergence
        converged = np.array_equal(olditassocid, itassocid)
        niter+=1
    print("Iterative combination complete.")
    return itassocid
       

# ------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------#
#   Supporting functions
# ------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------#


def group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.

    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupcz : Redshift velocity in km/s of galaxy i's group center.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    galaxyra=np.asarray(galaxyra)
    galaxydec=np.asarray(galaxydec)
    galaxycz=np.asarray(galaxycz)
    galaxygrpid=np.asarray(galaxygrpid)
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
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
        nmembers = len(galaxygrpid[sel])
        xcen=np.sum(galaxycz[sel]*galaxyx[sel])/nmembers
        ycen=np.sum(galaxycz[sel]*galaxyy[sel])/nmembers
        zcen=np.sum(galaxycz[sel]*galaxyz[sel])/nmembers
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        deccen = np.arcsin(zcen/czcen)*180.0/np.pi # degrees
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
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz


def nearest_neighbor_assign(galaxyra, galaxydec, galaxycz, galaxymag, grpid, rprojboundary, vprojboundary, centermethod):
    """
    For a list of galaxies defined by groups, refine group ID numbers using a nearest-neighbor
    search and applying the search boundaries.

    Parameters
    ------------
    galaxyra, galaxydec, galaxycz : iterable
        Input coordinates of galaxies (RA/Dec in decimal degrees, cz in km/s)
    galaxymag : iterable
        M_r magnitudes or stellar/baryonic masses of input galaxies. (note code refers to 'mags' throughout, but
        underlying `fit_in_group` function will distinguish the two.)
    grpid : iterable
        Group ID number for every input galaxy, at current iteration (potential group).
    rprojboundary, vprojboundary : callable
        Input functions to assess the search boundaries around potential groups, function of group-integrated luminosity, units Mpc/h and km/s.

    Returns
    ------------
    associd : iterable
        Refined group ID numbers based on NN "stitching" of groups.
    """
    # Prepare output array
    associd = deepcopy(grpid)
    # Get the group RA/Dec/cz for every galaxy
    groupra, groupdec, groupcz = group_skycoords(galaxyra, galaxydec, galaxycz, grpid) 
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
        combinedmag = np.hstack((galaxymag[Gpgalsel], galaxymag[GNNgalsel]))
        if fit_in_group(combinedra, combineddec, combinedcz, combinedmag, rprojboundary, vprojboundary, centermethod) and (not alreadydone[idx]) and (not alreadydone[nbridx]):
            # check for reciprocity: is the nearest-neighbor of GNN Gp? If not, leave them both as they are and let it be handled during the next iteration.
            nbrnnidx = nnind[nbridx]
            if idx==nbrnnidx:
                # change group ID of NN galaxies
                associd[GNNgalsel]=int(grpid[Gpgalsel][0])
                alreadydone[idx]=1
                alreadydone[nbridx]=1
            else:
                alreadydone[idx]=1
        else:
            alreadydone[idx]=1
    return associd  


def fit_in_group(galra, galdec, galcz, galmag, rprojboundary, vprojboundary, center='arithmetic'):
    """
    Check whether two potential groups can be merged based on the integrated luminosity of the 
    potential members, given limiting input group sizes.
    
    Parameters
    ----------------
    galra, galdec, galcz : iterable
        Coordinates of input galaxies -- all galaxies belonging to the pair of groups that are being assessed. 
    galmag : iterable
        M_r absolute magnitudes of all input galaxies, or galaxy stellar masses - the function can distinguish the two.
    rprojboundary, vprojboundary : callable
        Limiting projected- and velocity-space group sizes as function of group-integrated luminosity or stellar mass.
    center : str
        Specifies method of computing proposed center. Options: 'arithmetic' or 'luminosity'. The latter is a weighted-mean positioning based on M_r (like center of mass).

    Returns
    ----------------
    fitingroup : bool
        Bool indicating whether the series of input galaxies can be merged into a single group of the specified size.
    """
    if (galmag<0).all():
        memberintmag = get_int_mag(galmag, np.full(len(galmag), 1))
    elif (galmag>0).all():
        memberintmag = get_int_mass(galmag, np.full(len(galmag), 1))
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
    elif center=='luminosity':
        unlogmag = 10**(-0.4*galmag)
        xcen = np.sum(galcz*galx*unlogmag)/np.sum(unlogmag)
        ycen = np.sum(galcz*galy*unlogmag)/np.sum(unlogmag)
        zcen = np.sum(galcz*galz*unlogmag)/np.sum(unlogmag)
        czcenter = np.sqrt(xcen**2+ycen**2+zcen**2)
        deccenter = np.arcsin(zcen/czcenter)*(180.0/np.pi)
        phicorr = 0.0*int((ycen >=0 and xcen >=0)) + 180.0*int((ycen < 0 and xcen < 0) or (ycen >= 0 and xcen < 0)) + 360.0*int((ycen < 0 and xcen >=0))
        racenter = np.arctan(ycen/xcen)*(180.0/np.pi)+phicorr
    # check if all members are within rproj and vproj of group center
    halfangle = angular_separation(racenter,deccenter,galra[:,None],galdec[:,None])/2.0
    projsep = (galcz[:,None]+czcenter)/100. * halfangle
    lossep = np.abs(galcz[:,None]-czcenter)
    fitingroup=(np.all(projsep<rprojboundary(memberintmag)) and np.all(lossep<vprojboundary(memberintmag)))
    return fitingroup

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.

    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid = np.unique(grpids)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        occurences=[]
        for uid in uniqid:
            sel = np.where(grpids==uid)
            occurences.append(len(grpids[sel]))
        return occurences

def get_int_mag(galmags, grpid):
    """
    Given a list of galaxy absolute magnitudes and group ID numbers,
    compute group-integrated total magnitudes.

    Parameters
    ------------
    galmags : iterable
       List of absolute magnitudes for every galaxy (SDSS r-band).
    grpid : iterable
       List of group ID numbers for every galaxy.

    Returns
    ------------
    grpmags : np array
       Array containing group-integrated magnitudes for each galaxy. Length matches `galmags`.
    """
    galmags=np.asarray(galmags)
    grpid=np.asarray(grpid)
    grpmags = np.zeros(len(galmags))
    uniqgrpid=np.unique(grpid)
    for uid in uniqgrpid:
        sel=np.where(grpid==uid)
        totalmag = -2.5*np.log10(np.sum(10**(-0.4*galmags[sel])))
        grpmags[sel]=totalmag
    return grpmags


def get_int_mass(galmass, grpid):
    """
    Given a list of galaxy stellar or baryonic masses and group ID numbers,
    compute the group-integrated stellar or baryonic mass, galaxy-wise.
    
    Parameters
    ---------------
    galmass : iterable
        List of galaxy log(mass).
    grpid : iterable
        List of group ID numbers for every galaxy.


    Returns
    ---------------
    grpmstar : np.array
         Array containing group-integrated stellar masses for each galaxy; length matches `galmstar`.
    """
    galmass=np.asarray(galmass)
    grpid=np.asarray(grpid)
    grpmass = np.zeros(len(galmass))
    uniqgrpid=np.unique(grpid)
    for uid in uniqgrpid:
        sel=np.where(grpid==uid)
        totalmass = np.log10(np.sum(10**galmass[sel]))
        grpmass[sel]=totalmass
    return grpmass

def HAMwrapper(galra, galdec, galcz, galmag, galgrpid, volume,  inputfilename=None, outputfilename=None):
    """
    Perform halo abundance matching on a galaxy group catalog (wrapper around the C code of A.A. Berlind).

    Parameters
    -------------
    galra, galdec, galcz : iterable
        Input coordinates of galaxies (in decimal degrees, and km/s).
    galmag : iterable
        Input r-band absolute magnitudes of galaxies, or their stellar or baryonic masses. The code
        will distinguish between mags/masses, albeit variables in the code refer to 'mag' throughout.
    galgrpid : iterable
        Group ID number for every input galaxy.
        value to true if matching abundance on mass (like group-int stellar mass), for which the values are >0.
    volume : float, units in (Mpc/h)^3
        Survey volume. 
    inputfilename : string, default None
        Filename to save input HAM file for the C executable. If None, the file is removed during execution.
    outputfilename : string, default None
        Filename to save output HAM file from the C executable. If None, the file is removed during execution.

    Returns
    -------------
    haloid : np.float64 array
        ID of abundance-matched halos (matches number of unique values in `galgrpid`).
    halologmass : np.float64 array
        Log(halo mass per h) of each halo, length matches `haloid`.
    halorvir : np.float64 array
         Virial radii of each halo.
    halosigma : np.float64 array
         Theoretical velocity dispersion of each halo.
    """
    deloutfile=(outputfilename==None)
    delinfile=(inputfilename==None)
    # Prepare inputs
    grpra, grpdec, grpcz = group_skycoords(galra, galdec, galcz, galgrpid)
    if (galmag<0).all():
        grpmag = get_int_mag(galmag, galgrpid)
    elif (galmag>0).all():
        grpmag = -1*get_int_mass(galmag, galgrpid) # need -1 to trick Andreas' HAM code into using masses.
    grprproj, grpsigma = get_rproj_czdisp(galra, galdec, galcz, galgrpid)
    # Reshape them to match len grps
    uniqgrpid, uniqind = np.unique(galgrpid, return_index=True)
    grpra=grpra[uniqind]
    grpdec=grpdec[uniqind]
    grpcz=grpcz[uniqind]
    grprproj=grprproj[uniqind]
    grpsigma=grpsigma[uniqind]
    grpmag = grpmag[uniqind]
    grpN=[]
    for uid in uniqgrpid: grpN.append(len(galgrpid[np.where(galgrpid==uid)]))
    grpN=np.asarray(grpN)
    # Create input file and write to it
    if inputfilename is  None:
       inputfilename = "temporaryhaminput"+str(time.time())+".txt"
    f = open(inputfilename, 'w')
    for i in range(0,len(grpra)):   
        f.write("G\t{a}\t{b}\t{c}\t{d}\t{e}\t{f}\t{g}\t{h}\n".format(a=int(uniqgrpid[i]),b=grpra[i],c=grpdec[i],d=grpcz[i],e=grpN[i],f=grpsigma[i],g=grprproj[i],h=grpmag[i]))
    f.close()
    # try to do the HAM
    if outputfilename is None: outputfilename='temporaryhamoutput'+str(time.time())+".txt"
    #try:
    hamcommand = "./massmatch Mass_function.dat {} < ".format(volume)+inputfilename+" > "+outputfilename
    try:
        os.system(hamcommand)
    except:
        raise RunTimeError("OS call to HAM C executable failed; check input data type")
    hamfile=np.genfromtxt(outputfilename)
    haloid = np.float64(hamfile[:,0])
    halologmass = np.float64(hamfile[:,1])
    halorvir = np.float64(hamfile[:,2])
    halosigma = np.float64(hamfile[:,3])
    if deloutfile: os.remove(outputfilename)
    if delinfile: os.remove(inputfilename)
    return haloid, halologmass, halorvir, halosigma

def get_rproj_czdisp(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    Compute the observational projected radius, in Mpc/h, and the observational
    velocity dispersion, in km/s, for a galaxy group catalog. Input should match
    the # of galaxies, and the output will as well. Based on FoF4 code of Berlind+ 
    2006.
    
    Parameters
    ----------
    galaxyra : iterable
        Right-ascension of grouped galaxies in decimal degrees.
    galaxydec : iterable
        Declination of grouped galaxies in decimal degrees.
    galaxycz : iterable
        Redshift velocity (cz) of grouped galaxies in km/s.
    galaxygrpid : iterable
        Group ID numbers of grouped galaxies, shape should match `galaxyra`.

    Returns
    -------
    rproj : np.array, shape matches `galaxyra`
        For element index i, projected radius of galaxy group to which galaxy i belongs, in Mpc/h.
    vdisp : np.array, shape matches `galaxyra`
        For element index i, velocity dispersion of galaxy group to which galaxy i belongs, in km/s.

    """
    galaxyra=np.asarray(galaxyra)
    galaxydec=np.asarray(galaxydec)
    galaxycz=np.asarray(galaxycz)
    galaxygrpid=np.asarray(galaxygrpid)
    rproj=np.zeros(len(galaxyra))
    vdisp=np.zeros(len(galaxyra))
    grpra, grpdec, grpcz = group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid)
    grpra = grpra*np.pi/180. #convert  everything to radians
    galaxyra=galaxyra*np.pi/180.
    galaxydec=galaxydec*np.pi/180.
    grpdec = grpdec*np.pi/180.
    uniqid = np.unique(galaxygrpid)
    cspeed=299800 # km/s
    for uid in uniqid:
        sel = np.where(galaxygrpid==uid)
        nmembers=len(sel[0])
        if nmembers==1:
            rproj[sel]=0.
            vdisp[sel]=0.
        else:
            phicen=grpra[sel][0]
            thetacen=grpdec[sel][0]
            cosDpsi=np.cos(thetacen)*np.cos(galaxydec[sel])+np.sin(thetacen)*np.sin(galaxydec[sel])*np.cos((phicen - galaxyra[sel]))
            sinDpsi=np.sqrt(1-cosDpsi**2)
            rp=sinDpsi*galaxycz[sel]/HUBBLE_CONST
            rproj[sel]=np.sqrt(np.sum(rp**2)/len(sel[0]))
            czcen = grpcz[sel][0]
            Dz2 = np.sum((galaxycz[sel]-czcen)**2.0)
            vdisp[sel]=np.sqrt(Dz2/(nmembers-1))/(1.+czcen/cspeed)
    return rproj, vdisp

