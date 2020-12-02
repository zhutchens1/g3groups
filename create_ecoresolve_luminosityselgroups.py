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
(6) Based on giant+dwarf groups, calibrate boundaries (as function of giant+dwarf integrated luminosity) for iterative combination
(7) Iterative combination on remaining ungrouped dwarf galaxies
(8) halo mass assignment
(9) Finalize arrays + output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import foftools as fof
import iterativecombination as ic
from smoothedbootstrap import smoothedbootstrap as sbs
import pdb

def sqrtmodel(x, a, b):
    return a*np.sqrt(b*x)

def decayexp(x, a, b, c, d):
    return np.abs(a)*np.exp(-1*np.abs(b)*x + c)

if __name__=='__main__':
    ####################################
    # Step 1: Read in obs data
    ####################################
    ecodata = pd.read_csv("updatedECO_fora100match.csv")
    resolvedata = pd.read_csv("RESOLVEdata_Nov1820.csv")
    resolvebdata = resolvedata[resolvedata.f_b==1]

    ####################################
    # Step 2: Prepare arrays
    ####################################
    ecosz = len(ecodata)
    econame = np.array(ecodata.name)
    ecoresname = np.array(ecodata.resname)
    ecoradeg = np.array(ecodata.radeg)
    ecodedeg = np.array(ecodata.dedeg)
    ecocz = np.array(ecodata.cz)
    ecoabsrmag = np.array(ecodata.absrmag)
    ecologmstar = np.array(ecodata.logmstar)
    ecog3grp = np.full(ecosz, -99.) # id number of g3 group
    ecog3grpn = np.full(ecosz, -99.) # multiplicity of g3 group
    ecog3grpradeg = np.full(ecosz,-99.) # ra of group center
    ecog3grpdedeg = np.full(ecosz,-99.) # dec of group center
    ecog3grpcz = np.full(ecosz,-99.) # cz of group center
    ecog3logmh = np.full(ecosz,-99.) # abundance-matched halo mass 
    ecog3intmag = np.full(ecosz,-99.) # group-integrated r-band mag
    ecog3intmstar = np.full(ecosz,-99.) # group-integrated stellar mass

    resbana_g3grp = np.full(ecosz,-99.) # for RESOLVE-B analogue dataset
    
    resbsz = int(len(resolvebdata))
    resbname = np.array(resolvebdata.name)
    resbradeg = np.array(resolvebdata.radeg)
    resbdedeg = np.array(resolvebdata.dedeg)
    resbcz = np.array(resolvebdata.cz)
    resbabsrmag = np.array(resolvebdata.absrmag)
    resblogmstar = np.array(resolvebdata.logmstar)
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
    resbana_g3grp[ecogiantsel] = ecogiantfofid # RESOLVE-B analogue dataset
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

    rproj_median_error = np.std(np.array([sbs(relprojdist[np.where(ecogiantgrpn==sz)], 10000, np.median, kwargs=dict({'axis':1 })) for sz in uniqecogiantgrpn[keepcalsel]]), axis=1)
    dvproj_median_error = np.std(np.array([sbs(relvel[np.where(ecogiantgrpn==sz)], 10000, np.median, kwargs=dict({'axis':1})) for sz in uniqecogiantgrpn[keepcalsel]]), axis=1)

    #rprojslope, rprojint = np.polyfit(uniqecogiantgrpn[keepcalsel], median_relprojdist, deg=1, w=1/rproj_median_error)
    #dvprojslope, dvprojint = np.polyfit(uniqecogiantgrpn[keepcalsel], median_relvel, deg=1, w=1/dvproj_median_error)
    poptrproj, jk = curve_fit(sqrtmodel, uniqecogiantgrpn[keepcalsel], median_relprojdist, sigma=rproj_median_error)
    poptdvproj,jk = curve_fit(sqrtmodel, uniqecogiantgrpn[keepcalsel], median_relvel, sigma=dvproj_median_error) 
    rproj_boundary = lambda N: 3*sqrtmodel(N, *poptrproj) #3*(rprojslope*N+rprojint)
    vproj_boundary = lambda N: 4.5*sqrtmodel(N, *poptdvproj) #4.5*(dvprojslope*N+dvprojint)
    print("--- best fit parameters for dwarf assoc ----")
    print(poptrproj)
    print(poptdvproj)

    plt.figure()
    sel = (ecogiantgrpn>1)
    plt.plot(ecogiantgrpn[sel], relvel[sel], 'r.', alpha=0.2, label='ECO Giant Galaxies')
    plt.errorbar(uniqecogiantgrpn[keepcalsel], median_relvel, fmt='k^', label=r'$\Delta v_{\rm proj}$ (Median of $\Delta v_{\rm proj,\, gal}$)',yerr=dvproj_median_error)
    tx = np.linspace(0,max(ecogiantgrpn),1000)
    plt.plot(tx, sqrtmodel(tx, *poptdvproj), label=r'$1\Delta v_{\rm proj}^{\rm fit}$')
    plt.plot(tx, 4.5*sqrtmodel(tx, *poptdvproj), 'g',  label=r'$4.5\Delta v_{\rm proj}^{\rm fit}$', linestyle='-.')
    plt.xlabel("Number of Giant Members")
    plt.ylabel("Relative Velocity to Group Center [km/s]")
    plt.legend(loc='best')
    plt.show()

    plt.clf()
    plt.plot(ecogiantgrpn[sel], relprojdist[sel], 'r.', alpha=0.2, label='ECO Giant Galaxies')
    plt.errorbar(uniqecogiantgrpn[keepcalsel], median_relprojdist, fmt='k^', label=r'$R_{\rm proj}$ (Median of $R_{\rm proj,\, gal}$)',yerr=rproj_median_error)
    plt.plot(tx, sqrtmodel(tx, *poptrproj), label=r'$1R_{\rm proj}^{\rm fit}$')
    plt.plot(tx, 3*sqrtmodel(tx, *poptrproj), 'g', label=r'$3R_{\rm proj}^{\rm fit}$', linestyle='-.')
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
    resbana_dwarfsel = (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.0) & (ecocz>2530) & (ecocz<7470)    

    resbgiantgrpra, resbgiantgrpdec, resbgiantgrpcz = fof.group_skycoords(resbradeg[resbgiantsel], resbdedeg[resbgiantsel], resbcz[resbgiantsel], resbgiantfofid)
    resbgiantgrpn = fof.multiplicity_function(resbgiantfofid, return_by_galaxy=True)
    ecodwarfassocid, junk = fof.fast_faint_assoc(ecoradeg[ecodwarfsel],ecodedeg[ecodwarfsel],ecocz[ecodwarfsel],ecogiantgrpra,ecogiantgrpdec,ecogiantgrpcz,ecogiantfofid,\
                   rproj_boundary(ecogiantgrpn),vproj_boundary(ecogiantgrpn))
    resbdwarfassocid, junk = fof.fast_faint_assoc(resbradeg[resbdwarfsel],resbdedeg[resbdwarfsel],resbcz[resbdwarfsel],resbgiantgrpra,resbgiantgrpdec,resbgiantgrpcz,resbgiantfofid,\
                   rproj_boundary(resbgiantgrpn),vproj_boundary(resbgiantgrpn))
    
    resbana_dwarfassocid, jk = fof.fast_faint_assoc(ecoradeg[resbana_dwarfsel], ecodedeg[resbana_dwarfsel], ecocz[resbana_dwarfsel], ecogiantgrpra, ecogiantgrpdec, ecogiantgrpcz, ecogiantfofid,\
                                                    rproj_boundary(ecogiantgrpn), vproj_boundary(ecogiantgrpn))

    
    ecog3grp[ecodwarfsel] = ecodwarfassocid
    resbg3grp[resbdwarfsel] = resbdwarfassocid
    resbana_g3grp[resbana_dwarfsel] = resbana_dwarfassocid

    ###############################################
    # Step 6: Calibration for Iter. Combination
    ###############################################
    ecogdgrpn = fof.multiplicity_function(ecog3grp, return_by_galaxy=True)
    #ecogdsel = np.logical_not((ecogdgrpn==1) & (ecoabsrmag>-19.4) & (ecog3grp>0)) # select galaxies that AREN'T ungrouped dwarfs
    ecogdsel = np.logical_not(np.logical_or(ecog3grp==-99., ((ecogdgrpn==1) & (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.0))))
    ecogdgrpra, ecogdgrpdec, ecogdgrpcz = fof.group_skycoords(ecoradeg[ecogdsel], ecodedeg[ecogdsel], ecocz[ecogdsel], ecog3grp[ecogdsel])
    ecogdrelvel = np.abs(ecogdgrpcz - ecocz[ecogdsel])
    ecogdrelprojdist = (ecogdgrpcz + ecocz[ecogdsel])/100. * ic.angular_separation(ecogdgrpra, ecogdgrpdec, ecoradeg[ecogdsel], ecodedeg[ecogdsel])/2.0
    ecogdn = ecogdgrpn[ecogdsel]
    ecogdtotalmag = ic.get_int_mag(ecoabsrmag[ecogdsel], ecog3grp[ecogdsel])
   
    magbins=np.arange(-24,-19,0.5)
    binsel = np.where(np.logical_and(ecogdn>1, ecogdtotalmag>-24))
    gdmedianrproj, magbinedges, jk = binned_statistic(ecogdtotalmag[binsel], ecogdrelprojdist[binsel], lambda x:np.percentile(x,99), bins=magbins)
    gdmedianrelvel, jk, jk = binned_statistic(ecogdtotalmag[binsel], ecogdrelvel[binsel], lambda x: np.percentile(x,99), bins=magbins)
    print(magbinedges[:-1], gdmedianrproj, gdmedianrelvel)
    poptr, pcovr = curve_fit(decayexp, magbinedges[:-1], gdmedianrproj)
    poptv, pcovv = curve_fit(decayexp, magbinedges[:-1], gdmedianrelvel) 

    tx = np.linspace(-27,-17,100)
    plt.figure()
    plt.plot(ecogdtotalmag[binsel], ecogdrelprojdist[binsel], 'k.', alpha=0.2, label='ECO Galaxies in N>1 Giant+Dwarf Groups')
    plt.plot(magbinedges[:-1], gdmedianrproj, 'r^', label='99th percentile in bin')
    plt.plot(tx, decayexp(tx,*poptr))
    plt.xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    plt.ylabel("Projected Distance from Galaxy to Group Center [Mpc/h]")
    plt.legend(loc='best')
    plt.xlim(-25,-17)
    plt.ylim(0,1.3)
    plt.gca().invert_xaxis()
    plt.savefig("images/itercombboundaries.jpeg")
    plt.show()

    plt.figure()
    plt.plot(ecogdtotalmag[binsel], ecogdrelvel[binsel], 'k.', alpha=0.2, label='Mock Galaxies in N=2 Giant+Dwarf Groups')
    plt.plot(magbinedges[:-1], gdmedianrelvel,'r^',label='Medians')
    plt.plot(tx, decayexp(tx, *poptv))
    plt.ylabel("Relative Velocity between Galaxy and Group Center")
    plt.xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    plt.show()

    rproj_for_iteration = lambda M: decayexp(M, *poptr)
    vproj_for_iteration = lambda M: decayexp(M, *poptv)

    # --------------- now need to do this calibration for the RESOLVE-B analogue dataset, down to -17.0) -------------$
    resbana_gdgrpn = fof.multiplicity_function(resbana_g3grp, return_by_galaxy=True)
    #resbana_gdsel = np.logical_not((resbana_gdgrpn==1) & (ecoabsrmag>-19.4) & (resbana_g3grp!=-99.) & (resbana_g3grp>0)) # select galaxies that AREN'T ungrouped dwarfs
    resbana_gdsel = np.logical_not(np.logical_or(resbana_g3grp==-99., ((resbana_gdgrpn==1) & (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.0))))
    print(np.min(resbana_g3grp[resbana_gdsel]))
    resbana_gdgrpra, resbana_gdgrpdec, resbana_gdgrpcz = fof.group_skycoords(ecoradeg[resbana_gdsel], ecodedeg[resbana_gdsel], ecocz[resbana_gdsel], resbana_g3grp[resbana_gdsel])
    resbana_gdrelvel = np.abs(resbana_gdgrpcz - ecocz[resbana_gdsel])
    resbana_gdrelprojdist = (resbana_gdgrpcz + ecocz[resbana_gdsel])/100. * ic.angular_separation(resbana_gdgrpra, resbana_gdgrpdec, ecoradeg[resbana_gdsel], ecodedeg[resbana_gdsel])/2.0
    print(resbana_gdrelvel, resbana_gdrelprojdist)

    resbana_gdn = resbana_gdgrpn[resbana_gdsel]
    resbana_gdtotalmag = ic.get_int_mag(ecoabsrmag[resbana_gdsel], resbana_g3grp[resbana_gdsel])

    magbins2=np.arange(-24,-19,0.5)
    binsel2 = np.where(np.logical_and(resbana_gdn>1, resbana_gdtotalmag>-24))
    gdmedianrproj, magbinedges, jk = binned_statistic(resbana_gdtotalmag[binsel2], resbana_gdrelprojdist[binsel2], lambda x:np.percentile(x,99), bins=magbins2)
    gdmedianrelvel, jk, jk = binned_statistic(resbana_gdtotalmag[binsel2], resbana_gdrelvel[binsel2], lambda x: np.percentile(x,99), bins=magbins2)
    poptr_resbana, jk = curve_fit(decayexp, magbinedges[:-1], gdmedianrproj)
    poptv_resbana, jk = curve_fit(decayexp, magbinedges[:-1], gdmedianrelvel)
    print(magbinedges[:-1], gdmedianrproj)    

    tx = np.linspace(-27,-16,100)
    plt.figure()
    plt.plot(resbana_gdtotalmag[binsel2], resbana_gdrelprojdist[binsel2], 'k.', alpha=0.2, label='Mock Galaxies in N>1 Giant+Dwarf Groups')
    plt.plot(magbinedges[:-1], gdmedianrproj, 'r^', label='99th percentile in bin')
    plt.plot(tx, decayexp(tx,*poptr_resbana))
    plt.xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    plt.ylabel("Projected Distance from Galaxy to Group Center [Mpc/h]")
    plt.legend(loc='best')
    plt.xlim(-25,-17)
    #plt.ylim(0,1.3)
    plt.gca().invert_xaxis()
    plt.show()

    plt.figure()
    plt.plot(resbana_gdtotalmag[binsel2], resbana_gdrelvel[binsel2], 'k.', alpha=0.2, label='Mock Galaxies in N=2 Giant+Dwarf Groups')
    plt.plot(magbinedges[:-1], gdmedianrelvel,'r^',label='Medians')
    plt.plot(tx, decayexp(tx, *poptv_resbana))
    plt.ylabel("Relative Velocity between Galaxy and Group Center")
    plt.xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    plt.show()

    rproj_for_iteration_resbana = lambda M: decayexp(M, *poptr_resbana)
    vproj_for_iteration_resbana = lambda M: decayexp(M, *poptv_resbana)


    ###########################################################
    # Step 7: Iterative Combination of Dwarf Galaxies
    ###########################################################
    assert (ecog3grp[(ecoabsrmag<=-19.4) & (ecocz<7470) & (ecocz>2530)]!=-99.).all(), "Not all giants are grouped."
    ecogrpnafterassoc = fof.multiplicity_function(ecog3grp, return_by_galaxy=True)
    resbgrpnafterassoc = fof.multiplicity_function(resbg3grp, return_by_galaxy=True)
    resbana_grpnafterassoc = fof.multiplicity_function(resbana_g3grp, return_by_galaxy=True)

    eco_ungroupeddwarf_sel = (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.33) & (ecocz<7470) & (ecocz>2530) & (ecogrpnafterassoc==1)
    print(eco_ungroupeddwarf_sel)
    ecoitassocid = ic.iterative_combination(ecoradeg[eco_ungroupeddwarf_sel], ecodedeg[eco_ungroupeddwarf_sel], ecocz[eco_ungroupeddwarf_sel], ecoabsrmag[eco_ungroupeddwarf_sel],\
                                           rproj_for_iteration, vproj_for_iteration, starting_id=np.max(ecog3grp)+1, centermethod='arithmetic')
    
    resb_ungroupeddwarf_sel = (resbabsrmag>-19.4) & (resbabsrmag<=-17.0) & (resbcz<7250) & (resbcz>4250) & (resbgrpnafterassoc==1)
    resbitassocid = ic.iterative_combination(resbradeg[resb_ungroupeddwarf_sel], resbdedeg[resb_ungroupeddwarf_sel], resbcz[resb_ungroupeddwarf_sel], resbabsrmag[resb_ungroupeddwarf_sel],\
                                            rproj_for_iteration, vproj_for_iteration, starting_id=np.max(resbg3grp)+1, centermethod='arithmetic')
    
    resbana_ungroupeddwarf_sel = (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.0) & (ecocz<7470) & (ecocz>2530) & (resbana_grpnafterassoc==1)
    resbana_itassocid = ic.iterative_combination(ecoradeg[resbana_ungroupeddwarf_sel], ecodedeg[resbana_ungroupeddwarf_sel], ecocz[resbana_ungroupeddwarf_sel], ecoabsrmag[resbana_ungroupeddwarf_sel],\
                                                 rproj_for_iteration_resbana, vproj_for_iteration_resbana, starting_id=np.max(resbana_g3grp)+1, centermethod='arithmetic')

    ecog3grp[eco_ungroupeddwarf_sel] = ecoitassocid
    resbg3grp[resb_ungroupeddwarf_sel] = resbitassocid
    resbana_g3grp[resbana_ungroupeddwarf_sel] = resbana_itassocid
    #plt.figure()
    #plt.hist(fof.multiplicity_function(ecoitassocid, return_by_galaxy=False), log=True)
    #plt.hist(fof.multiplicity_function(resbitassocid, return_by_galaxy=False), log=True, histtype='step')
    #plt.show()
    
    plt.figure()
    binv = np.arange(0.5,1200.5,3)
    plt.hist(fof.multiplicity_function(ecog3grp[ecog3grp!=-99.], return_by_galaxy=False), bins=binv, log=True, label='ECO Groups', histtype='step', linewidth=3)
    plt.hist(fof.multiplicity_function(resbg3grp[resbg3grp!=-99.], return_by_galaxy=False), bins=binv, log=True, label='RESOLVE-B Groups', histtype='step', hatch='\\')
    plt.xlabel("Number of Giant + Dwarf Group Members")
    plt.ylabel("Number of Groups")
    plt.legend(loc='best')
    plt.xlim(0,100)
    plt.show()
 
    ############################################################
    # Step 8: Halo Abundance Matching
    ###########################################################
    # --- for RESOLVE-B analogue ----#
    resbana_hamsel = (resbana_g3grp!=-99.)
    resbana_haloid, resbana_halomass, jk, jk = ic.HAMwrapper(ecoradeg[resbana_hamsel], ecodedeg[resbana_hamsel], ecocz[resbana_hamsel], ecoabsrmag[resbana_hamsel], resbana_g3grp[resbana_hamsel],\
                                                                ecovolume, inputfilename=None, outputfilename=None)
    junk, uniqindex = np.unique(resbana_g3grp[resbana_hamsel], return_index=True)
    resbana_intmag = ic.get_int_mag(ecoabsrmag[resbana_hamsel], resbana_g3grp[resbana_hamsel])[uniqindex]
    sortind = np.argsort(resbana_intmag)
    sortedmag = resbana_intmag[sortind]
    resbcubicspline = interp1d(sortedmag, resbana_halomass[sortind])    
    
    resbintmag = ic.get_int_mag(resbabsrmag[resbg3grp!=-99.], resbg3grp[resbg3grp!=-99.])
    resbg3logmh[resbg3grp!=-99.] = resbcubicspline(resbintmag)-np.log10(0.7)
 
    # ---- for ECO ----- #
    ecohamsel = (ecog3grp!=-99.)
    haloid, halomass, junk, junk = ic.HAMwrapper(ecoradeg[ecohamsel], ecodedeg[ecohamsel], ecocz[ecohamsel], ecoabsrmag[ecohamsel], ecog3grp[ecohamsel],\
                                                     ecovolume, inputfilename=None, outputfilename=None)
    junk, uniqindex = np.unique(ecog3grp[ecohamsel], return_index=True)
    halomass = halomass-np.log10(0.7)
    for i,idv in enumerate(haloid):
        sel = np.where(ecog3grp==idv)
        ecog3logmh[sel] = halomass[i]


    ecointmag = ic.get_int_mag(ecoabsrmag[ecohamsel], ecog3grp[ecohamsel])
    plt.figure()
    plt.plot(ecointmag, ecog3logmh[ecog3grp!=-99.], '.', color='palegreen', alpha=0.6, label='ECO', markersize=11)
    plt.plot(resbintmag, resbg3logmh[resbg3grp!=-99.], 'k.', alpha=1, label='RESOLVE-B', markersize=3)
    plt.plot
    plt.xlabel("group-integrated r-band luminosity")
    plt.ylabel(r"group halo mass (log$M_\odot$)")
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.savefig("images/hamLrrelation.jpeg")
    plt.show()

    ########################################
    # (9) Output arrays     
    ########################################
    # ---- first get the quantities for ECO ---- #
    #eco_in_gf = np.where(ecog3grp!=-99.)
    ecog3grpn = fof.multiplicity_function(ecog3grp, return_by_galaxy=True)
    ecog3grpradeg, ecog3grpdedeg, ecog3grpcz = fof.group_skycoords(ecoradeg, ecodedeg, ecocz, ecog3grp)
    ecog3intmag = ic.get_int_mag(ecoabsrmag, ecog3grp)
    ecog3intmstar = ic.get_int_mass(ecologmstar, ecog3grp)
  
    outofsample = (ecog3grp==99.)
    ecog3grpn[outofsample]=-99.
    ecog3grpradeg[outofsample]=-99.
    ecog3grpdedeg[outofsample]=-99.
    ecog3grpcz[outofsample]=-99.
    ecog3intmag[outofsample]=-99.
    ecog3intmstar[outofsample]=-99.
    ecog3logmh[outofsample]=-99.
 
    ecodata['g3grp'] = ecog3grp
    ecodata['g3grn'] = ecog3grpn
    ecodata['g3grpradeg'] = ecog3grpradeg
    ecodata['g3grpdedeg'] = ecog3grpdedeg
    ecodata['g3grpcz'] = ecog3grpcz
    ecodata['g3grpabsrmag'] = ecog3intmag
    ecodata['g3grpmstar'] = ecog3intmstar
    ecodata['g3logmh'] = ecog3logmh
    ecodata.to_csv("ECOdata_G3catalog_luminosity.csv")    

    # ------ now do RESOLVE
    sz = len(resolvedata)
    resolvename = np.array(resolvedata.name)
    resolveg3grp = np.full(sz, -99.)
    resolveg3grpn = np.full(sz, -99.)
    resolveg3grpradeg = np.full(sz, -99.)
    resolveg3grpdedeg = np.full(sz, -99.)
    resolveg3grpcz = np.full(sz, -99.)
    resolveg3intmag = np.full(sz, -99.)
    resolveg3intmstar = np.full(sz, -99.)
    resolveg3logmh = np.full(sz, -99.)

    resbg3grpn = fof.multiplicity_function(resbg3grp, return_by_galaxy=True)
    resbg3grpradeg, resbg3grpdedeg, resbg3grpcz = fof.group_skycoords(resbradeg, resbdedeg, resbcz, resbg3grp)
    resbg3intmag = ic.get_int_mag(resbabsrmag, resbg3grp)
    resbg3intmstar = ic.get_int_mass(resblogmstar, resbg3grp)

    outofsample = (resbg3grp==99.)
    resbg3grpn[outofsample]=-99.
    resbg3grpradeg[outofsample]=-99.
    resbg3grpdedeg[outofsample]=-99.
    resbg3grpcz[outofsample]=-99.
    resbg3intmag[outofsample]=-99.
    resbg3intmstar[outofsample]=-99.
    resbg3logmh[outofsample]=-99.

    for i,nm in enumerate(resolvename):
        if nm.startswith('rs'):
            sel_in_eco = np.where(ecoresname==nm)
            resolveg3grp[i] = ecog3grp[sel_in_eco]
            resolveg3grpn[i] = ecog3grpn[sel_in_eco]
            resolveg3grpradeg[i] = ecog3grpradeg[sel_in_eco]
            resolveg3grpdedeg[i] = ecog3grpdedeg[sel_in_eco]
            resolveg3grpcz[i] = ecog3grpcz[sel_in_eco]
            resolveg3intmag[i] = ecog3intmag[sel_in_eco]
            resolveg3intmstar[i] = ecog3intmstar[sel_in_eco]
            resolveg3logmh[i] = ecog3logmh[sel_in_eco]
        elif nm.startswith('rf'):
            sel_in_resb = np.where(resbname==nm)
            resolveg3grp[i] = resbg3grp[sel_in_resb]
            resolveg3grpn[i] = resbg3grpn[sel_in_resb]
            resolveg3grpradeg[i] = resbg3grpradeg[sel_in_resb]
            resolveg3grpdedeg[i] = resbg3grpdedeg[sel_in_resb]
            resolveg3grpcz[i] = resbg3grpcz[sel_in_resb]
            resolveg3intmag[i] = resbg3intmag[sel_in_resb]
            resolveg3intmstar[i] = resbg3intmstar[sel_in_resb]
            resolveg3logmh[i] = resbg3logmh[sel_in_resb]
        else:
            assert False, nm+" not in RESOLVE"

    resolvedata['g3grp'] = resolveg3grp
    resolvedata['g3grpn'] = resolveg3grpn
    resolvedata['g3grpradeg'] = resolveg3grpradeg
    resolvedata['g3grpdedeg'] = resolveg3grpdedeg
    resolvedata['g3grpcz'] = resolveg3grpcz
    resolvedata['g3grpabsrmag'] = resolveg3intmag
    resolvedata['g3grpmstar'] = resolveg3intmstar
    resolvedata['g3logmh'] = resolveg3logmh
    resolvedata.to_csv("RESOLVEdata_G3catalog_luminosity.csv")

    


     

        
    

    


  


 
     
    

  


 

