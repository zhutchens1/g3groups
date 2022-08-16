import numpy as np
import foftools as fof
import iterativecombination as ic

def g3groupfinder(radeg,dedeg,cz,groupproperty,dwarfgiantdivide,fof_bperp=0.07,fof_blos=1.1,fof_sep=None,\
                 volume=None,iterative_giant_only_groups=False, rproj_fit_guess=None, rproj_fit_params = None,\
                 vproj_fit_guess = None, vproj_fit_params = None, gd_rproj_fit_guess=None, gd_rproj_fit_params = None,\
                 gd_vproj_fit_guess=None, gd_vproj_fit_params = None, ic_center_mode='arithmetic', ic_decision_mode='centers',\
                 showplots=False, saveplotspdf=False, H0=100.):
    """
    Identify galaxy groups in redshift space using the RESOLVE-G3 algorithm (Hutchens et al. 2022).
    
    Parameters
    -------------------
    radeg : array_like
        Right ascension of input galaxies in decimal degrees.
    dedeg : array_like
        Declination of input galaxies in decimal degrees.
    cz : array_like
        Recessional velocities of input galaxies in decimal degrees.
    groupproperty : array_like
        Group property by which giants and dwarfs will be selected
        (e.g., stellar mass or galaxy luminosity). If all values are negative,
        absolute magnitude is the assumed property. 
    dwarfgiantdivide : float
        Value that will divide giants and dwarfs according to galproperty.
    fof_bperp : float
        Perpendicular FoF linking length, default 0.07.
    fof_blos : float
        Line-of-sight FoF linking length, default 1.1.
    fof_sep : float
        Mean galaxy separation used for FoF. Should be expressed in units of (Mpc/h) with 
        h corresponding to the `H0` argument (i.e. use h=0.7 if setting H0=70.). If None
        (default), fof_sep will be determined using the number of galaxies and `volume`.
    volume : float
        Group finding volume in (Mpc/h)^3 with h corresponding to the `H0` argument, default
        None. This argument is unnecessary if fof_sep is provided. `fof_sep` and `volume`
        cannot both be `None`.
    iterative_giant_only_groups : bool 
        If False (default), giant-only groups are determined with a single run of FoF.
        If True, giant-only groups are determined iteratively, starting with FoF and refining
        based on iteratively-updated group boundaries.
    rproj_fit_guess 
    """
