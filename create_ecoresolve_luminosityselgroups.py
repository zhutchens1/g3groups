"""
Zackary Hutchens - November 2020

This file creates group catalogs for RESOLVE/ECO based on the G3 group-finding procedure. This procedure has three steps: giant-only FoF, association of dwarfs, and identification
of dwarf-only groups.

The outline of this code is:
(1) Read in observational data from RESOLVE-B and ECO (the latter includes RESOLVE-A).
(2) Prepare arrays of input parameters and for storing results.
(3) Perform FoF only for giants in ECO, using an adaptive linking strategy.
    (a) Get the adaptive links for every ECO galaxy.
    (b) Fit those adaptive links for use in RESOLVE-B.
    (c) Perform giant-only FoF for ECO
    (d) Perform giant-only FoF for RESOLVE-B, by interpolating the fit to obtain separations for RESOLVE-B. 
(4) From giant-only groups, fit model for individual giant projected radii and peculiar velocites, to use for association.
(5) Associate dwarf galaxies to giant-only FoF groups for ECO and RESOLVE-B (note different selection floors).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import foftools as fof
import iterativecombination as ic

if __name__=='__main__':
    ####################################
    # Step 1: Read in obs data
    ####################################
    ecodata = pd.read_csv("updatedECO_fora100match.csv")
    resolvebdata = pd.read_csv("RESOLVEdata_Nov1820.csv")
    resolvebdata = resolvebdata[resolvebdata.f_b==1]

    ####################################
    # Step 2: Prepare arrays
    ####################################
    ecosz = len(ecodata)
    ecoradeg = np.array(ecodata.radeg)
    ecodedeg = np.array(ecodata.dedeg)
    ecocz = np.array(ecodata.cz)
    ecoabsrmag = np.array(ecodata.absrmag)
    ecog3grp = np.full(ecosz, -99.) # id number of g3 group
    ecog3grpn = np.full(ecosz, -99.) # multiplicity of g3 group
    ecog3grpradeg = np.full(ecosz,-99.) # ra of group center
    ecog3grpdedeg = np.full(ecosz,-99.) # dec of group center
    ecog3grpcz = np.full(ecosz,-99.) # cz of group center
    ecog3logmh = np.full(ecosz,-99.) # abundance-matched halo mass 
    ecog3intmag = np.full(ecosz,-99.) # group-integrated r-band mag
    ecog3intmstar = np.full(ecosz,-99.) # group-integrated stellar mass
    
    resbsz = int(len(resolvebdata))
    resbradeg = np.array(resolvebdata.radeg)
    resbdedeg = np.array(resolvebdata.dedeg)
    resbcz = np.array(resolvebdata.cz)
    resbabsrmag = np.array(resolvebdata.absrmag)
    resbg3grp = np.full(resbsz, -99.)
    resbg3grpn = np.full(resbsz, -99.) 
    resbg3grpradeg = np.full(resbsz, -99.)
    resbg3grpdedeg = np.full(resbsz, -99.)
    resbg3grpcz = np.full(resbsz, -99.)
    resbg3logmh = np.full(resbsz, -99.)
    resbg3intmag = np.full(resbsz, -99.)
    resbg3intmstar = np.full(resbsz, -99.)

    ####################################
    # Step 3: Giant-Only FOF
    ####################################
    ecogiantsel = (ecoabsrmag<=-19.4) & (ecocz>2530.) & (ecocz<7470.)
    # (a) compute sep values for eco giants
    ecovolume = 192351.36 # Mpc^3 with h=1 **
    meansep0 = (ecovolume/len(ecoabsrmag[ecogiantsel]))**(1/3.)
    ecogiantmags = ecoabsrmag[ecogiantsel]
    ecogiantsep = np.array([(192351.36/len(ecogiantmags[ecogiantmags<=Mr]))**(1/3.) for Mr in ecogiantmags])
    ecogiantsep = ecogiantsep*meansep0/np.median(ecogiantsep)

    # (b) make an interpolation function use this for RESOLVE-B  
    meansepinterp = interp1d(ecogiantmags, ecogiantsep, fill_value='extrapolate')
    resbgiantsel = (resbabsrmag<=-19.4) & (resbcz>4250) & (resbcz<7250)
    resbgiantsep = meansepinterp(resbabsrmag[resbgiantsel])

    plt.figure()
    plt.axhline(meansep0, label=r'Mean Separation of ECO Giant Galaxies, $s = (V/N)^{1/3}$')
    plt.plot(ecogiantmags, ecogiantsep, 'k.', alpha=1, label=r'ECO Giant Galaxies ($M_r \leq -19.4$)')
    plt.plot(resbabsrmag[resbgiantsel], resbgiantsep, 'r^', alpha=0.4, label=r'RESOLVE-B Giant Galaxies (interpolated, $M_r \leq -19.4$)')
    plt.xlabel("Absolute $M_r$ of Giant Galaxy")
    plt.ylabel(r"$s_i$ - Separation used for Galaxy $i$ in Giant-Only FoF [Mpc/h]")
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.savefig("meansep_M_r_plot.jpg")
    plt.show()

    # (c) perform giant-only FoF on ECO
    blos = 1.1
    bperp = 0.07 # from Duarte & Mamon 2014
    ecogiantfofid = fof.fast_fof(ecoradeg[ecogiantsel], ecodedeg[ecogiantsel], ecocz[ecogiantsel], blos, bperp, ecogiantsep)
    ecog3grp[ecogiantsel] = ecogiantfofid

    # (d) perform giant-only FoF on RESOLVE-B
    resbgiantfofid = fof.fast_fof(resbradeg[resbgiantsel], resbdedeg[resbgiantsel], resbcz[resbgiantsel], blos, bperp, resbgiantsep)
    resbg3grp[resbgiantsel] = resbgiantfofid
 
    # (e) check the FOF results
    plt.figure()
    binv = np.arange(0.5,300.5,3)
    plt.hist(fof.multiplicity_function(ecog3grp[ecog3grp!=-99.], return_by_galaxy=False), bins=binv, histtype='step', linewidth=3, label='ECO Giant-Only FoF Groups')
    plt.hist(fof.multiplicity_function(resbg3grp[resbg3grp!=-99.], return_by_galaxy=False), bins=binv, histtype='step', linewidth=1.5, hatch='\\', label='RESOLVE-B Giant-Only FoF Groups')
    plt.xlabel("Number of Giant Galaxies per Group")
    plt.ylabel("Number of Giant-Only FoF Groups")
    plt.yscale('log')
    plt.legend(loc='best')
    plt.xlim(0,80)
    plt.savefig("giantonlymult.jpg")
    plt.show()
     
    

  


 

