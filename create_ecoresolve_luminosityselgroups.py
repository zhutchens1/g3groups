"""
Zackary Hutchens - November 2020

This program creates luminosity-selected  group catalogs for ECO/RESOLVE-G3 using the new algorithm, described in the readme markdown.

The outline of this code is:
(1) Read in observational data from RESOLVE-B and ECO (the latter includes RESOLVE-A).
(2) Prepare arrays of input parameters and for storing results.
(3) Perform FoF only for giants in ECO, using an adaptive linking strategy.
    (a) Get the adaptive links for every ECO galaxy.
    (b) Fit those adaptive links for use in RESOLVE-B.
    (c) Perform giant-only FoF for ECO
    (d) Perform giant-only FoF for RESOLVE-B, by interpolating the fit to obtain separations for RESOLVE-B. 
(4) From giant-only groups, fit model for individual giant projected radii and peculiar velocites, to use for association.
(5) Associate dwarf galaxies to giant-only FoF groups for ECO and RESOLVE-B (note different selection floors for dwarfs).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import foftools as fof
import iterativecombination as ic
from smoothedbootstrap import smoothedbootstrap as sbs

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
    plt.axhline(meansep0, label=r'Mean Separation of ECO Giant Galaxies, $s_0 = (V/N)^{1/3}$')
    plt.plot(ecogiantmags, ecogiantsep, 'k.', alpha=1, label=r'ECO Giant Galaxies ($M_r \leq -19.4$)')
    plt.plot(resbabsrmag[resbgiantsel], resbgiantsep, 'r^', alpha=0.4, label=r'RESOLVE-B Giant Galaxies (interpolated, $M_r \leq -19.4$)')
    plt.xlabel("Absolute $M_r$ of Giant Galaxy")
    plt.ylabel(r"$s_i$ - Separation used for Galaxy $i$ in Giant-Only FoF [Mpc/h]")
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.savefig("images/meansep_M_r_plot.jpg")
    plt.show()

    # (c) perform giant-only FoF on ECO
    blos = 1.1
    bperp = 0.07 # from Duarte & Mamon 2014
    ecogiantfofid = fof.fast_fof(ecoradeg[ecogiantsel], ecodedeg[ecogiantsel], ecocz[ecogiantsel], bperp, blos, ecogiantsep)
    ecog3grp[ecogiantsel] = ecogiantfofid

    # (d) perform giant-only FoF on RESOLVE-B
    resbgiantfofid = fof.fast_fof(resbradeg[resbgiantsel], resbdedeg[resbgiantsel], resbcz[resbgiantsel], bperp, blos, resbgiantsep)
    resbg3grp[resbgiantsel] = resbgiantfofid
 
    # (e) check the FOF results
    plt.figure()
    binv = np.arange(0.5,3000.5,3)
    plt.hist(fof.multiplicity_function(ecog3grp[ecog3grp!=-99.], return_by_galaxy=False), bins=binv, histtype='step', linewidth=3, label='ECO Giant-Only FoF Groups')
    plt.hist(fof.multiplicity_function(resbg3grp[resbg3grp!=-99.], return_by_galaxy=False), bins=binv, histtype='step', linewidth=1.5, hatch='\\', label='RESOLVE-B Giant-Only FoF Groups')
    plt.xlabel("Number of Giant Galaxies per Group")
    plt.ylabel("Number of Giant-Only FoF Groups")
    plt.yscale('log')
    plt.legend(loc='best')
    plt.xlim(0,80)
    plt.savefig("images/giantonlymult.jpg")
    plt.show()
    
    ##########################################
    # Step 4: Compute Association Boundaries
    ##########################################
    ecogiantgrpra, ecogiantgrpdec, ecogiantgrpcz = fof.group_skycoords(ecoradeg[ecogiantsel], ecodedeg[ecogiantsel], ecocz[ecogiantsel], ecogiantfofid)
    relvel = np.abs(ecogiantgrpcz - ecocz[ecogiantsel])
    relprojdist = (ecogiantgrpcz + ecocz[ecogiantsel])/100. * ic.angular_separation(ecogiantgrpra, ecogiantgrpdec, ecoradeg[ecogiantsel], ecodedeg[ecogiantsel])/2.0
    ecogiantgrpn = fof.multiplicity_function(ecogiantfofid, return_by_galaxy=True)
    uniqecogiantgrpn, uniqindex = np.unique(ecogiantgrpn, return_index=True)
    keepcalsel = np.where(uniqecogiantgrpn>1)

    median_relprojdist = np.array([np.median(relprojdist[np.where(ecogiantgrpn==sz)]) for sz in uniqecogiantgrpn[keepcalsel]])
    median_relvel = np.array([np.median(relvel[np.where(ecogiantgrpn==sz)]) for sz in uniqecogiantgrpn[keepcalsel]])

    rproj_median_error = np.std(np.array([sbs(relprojdist[np.where(ecogiantgrpn==sz)], 100000, np.median, kwargs=dict({'axis':1 })) for sz in uniqecogiantgrpn[keepcalsel]]), axis=1)
    dvproj_median_error = np.std(np.array([sbs(relvel[np.where(ecogiantgrpn==sz)], 100000, np.median, kwargs=dict({'axis':1})) for sz in uniqecogiantgrpn[keepcalsel]]), axis=1)

    rprojslope, rprojint = np.polyfit(uniqecogiantgrpn[keepcalsel], median_relprojdist, w=1/uniqecogiantgrpn[keepcalsel], deg=1)
    dvprojslope, dvprojint = np.polyfit(uniqecogiantgrpn[keepcalsel], median_relvel, w=1/uniqecogiantgrpn[keepcalsel], deg=1)
    print(rprojslope, rprojint)
    print(dvprojslope, dvprojint)
    rproj_boundary = lambda N: 3*(rprojslope*N+rprojint)
    vproj_boundary = lambda N: 4.5*(dvprojslope*N+dvprojint)

    plt.figure()
    sel = (ecogiantgrpn>1)
    plt.plot(ecogiantgrpn[sel], relvel[sel], 'r.', alpha=0.2, label='ECO Giant Galaxies')
    plt.errorbar(uniqecogiantgrpn[keepcalsel], median_relvel, fmt='k^', label=r'$\Delta v_{\rm proj}$ (Median of $\Delta v_{\rm proj,\, gal}$)',yerr=dvproj_median_error)
    tx = np.linspace(0,max(ecogiantgrpn),10)
    plt.plot(tx, (dvprojslope*tx+dvprojint), label=r'$1\Delta v_{\rm proj}^{\rm fit}$')
    plt.plot(tx, 4.5*(dvprojslope*tx+dvprojint), 'g',  label=r'$4.5\Delta v_{\rm proj}^{\rm fit}$', linestyle='-.')
    plt.xlabel("Number of Giant Members")
    plt.ylabel("Relative Velocity to Group Center [km/s]")
    plt.legend(loc='best')
    plt.show()

    plt.clf()
    plt.plot(ecogiantgrpn[sel], relprojdist[sel], 'r.', alpha=0.2, label='ECO Giant Galaxies')
    plt.errorbar(uniqecogiantgrpn[keepcalsel], median_relprojdist, fmt='k^', label=r'$R_{\rm proj}$ (Median of $R_{\rm proj,\, gal}$)',yerr=rproj_median_error)
    plt.plot(tx, (rprojslope*tx+rprojint), label=r'$1R_{\rm proj}^{\rm fit}$')
    plt.plot(tx, 3*(rprojslope*tx+rprojint), 'g', label=r'$3R_{\rm proj}^{\rm fit}$', linestyle='-.')
    plt.xlabel("Number of Giant Members in Galaxy's Group")
    plt.ylabel("Projected Distance from Giant to Group Center [Mpc/h]")
    plt.legend(loc='best')
    plt.xlim(0,20)
    plt.ylim(0,2.5)
    plt.xticks(np.arange(0,22,2))
    plt.savefig("images/rproj_calibration_assoc.jpg")
    plt.show()

    ####################################
    # Step 5: Association of Dwarfs
    ####################################
    ecodwarfsel = (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.33) & (ecocz>2530) & (ecocz<7470)
    resbdwarfsel = (resbabsrmag>-19.4) & (resbabsrmag<=-17.0) & (resbcz>4250) & (resbcz<7250)

    resbgiantgrpra, resbgiantgrpdec, resbgiantgrpcz = fof.group_skycoords(resbradeg[resbgiantsel], resbdedeg[resbgiantsel], resbcz[resbgiantsel], resbgiantfofid)
    resbgiantgrpn = fof.multiplicity_function(resbgiantfofid, return_by_galaxy=True)
    ecodwarfassocid = fof.fast_faint_assoc(ecoradeg[ecodwarfsel],ecodedeg[ecodwarfsel],ecocz[ecodwarfsel],ecogiantgrpra,ecogiantgrpdec,ecogiantgrpcz,ecogiantfofid,\
                   rproj_boundary(ecogiantgrpn),vproj_boundary(ecogiantgrpn))
    resbdwarfassocid = fof.fast_faint_assoc(resbradeg[resbdwarfsel],resbdedeg[resbdwarfsel],resbcz[resbdwarfsel],resbgiantgrpra,resbgiantgrpdec,resbgiantgrpcz,resbgiantfofid,\
                   rproj_boundary(resbgiantgrpn),vproj_boundary(resbgiantgrpn))

    print(ecodwarfassocid)
     
    

  


 

