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
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit
from center_binned_stats import center_binned_stats
import foftools as fof
import iterativecombination as ic
from smoothedbootstrap import smoothedbootstrap as sbs
from scipy.interpolate import UnivariateSpline
import sys

def sigmarange(x):
    q84, q16 = np.percentile(x, [84 ,16])
    return (q84-q16)/2.

#def giantmodel(x, a, b, c, d):
#    return a*np.log(np.abs(b)*x+c)+d

def giantmodel(x, a, b):
    return np.abs(a)*np.log(np.abs(b)*x+1)

def decayexp(x, a, b, c):
    return np.abs(a)*np.exp(-1*np.abs(b)*x + c)#+np.abs(d)

def sepmodel(x, a, b, c, d, e):
    #return np.abs(a)*np.exp(-1*np.abs(b)*x + c)+d
    #return a*(x**3)+b*(x**2)+c*x+d
    return a*(x**4)+b*(x**3)+c*(x**2)+(d*x)+e

if __name__=='__main__':
    ####################################
    # Step 1: Read in obs data
    ####################################
    ecodata = pd.read_csv("ECOdata_022521.csv")
    resolvedata = pd.read_csv("RESOLVEdata_022521.csv")
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
    ecogiantsel = (ecoabsrmag<=-19.4) & (ecocz>2530.) & (ecocz<8000.)
    # (a) compute sep values for eco giants
    ecovolume = 192351.36 # Mpc^3 with h=1 **
    meansep0 = (ecovolume/len(ecoabsrmag[ecogiantsel]))**(1/3.)
    ecogiantmags = ecoabsrmag[ecogiantsel]
    ecogiantsepdata = np.array([(192351.36/len(ecogiantmags[ecogiantmags<=Mr]))**(1/3.) for Mr in ecogiantmags])
    ecogiantsepdata = ecogiantsepdata*meansep0/np.median(ecogiantsepdata)
    #poptsfit, pcovsfit = curve_fit(sepmodel, ecogiantmags, ecogiantsepdata) 
    #meansepinterp = lambda x: sepmodel(x, *poptsfit) 
    #ecogiantsep = meansepinterp(ecogiantmags)
    meansepinterp = UnivariateSpline(np.sort(ecogiantmags), ecogiantsepdata[np.argsort(ecogiantmags)])
    meansepinterp.set_smoothing_factor(100)
    ecogiantsep = meansepinterp(ecogiantmags)
    print("Median Residual of Separation Fit: {} Mpc/h".format(np.median(np.abs(ecogiantsep-ecogiantsepdata))))

    # (b) make an interpolation function use this for RESOLVE-B  
    resbgiantsel = (resbabsrmag<=-19.4) & (resbcz>4250) & (resbcz<7300)
    resbgiantsep = meansepinterp(resbabsrmag[resbgiantsel])

    plt.figure()
    tx=np.linspace(-24,-19.3,100)
    plt.axhline(meansep0, label=r'Mean Separation of ECO Giant Galaxies, $\bar{s} = (V/N)^{1/3}$', color='k', linestyle='--')
    plt.plot(tx, meansepinterp(tx), label=r'$s_{i,\rm\, fit}$')
    plt.plot(ecogiantmags, ecogiantsepdata, 'k.', alpha=1, label=r'$s_i$ for ECO Giant Galaxies ($M_r \leq -19.4$)')
    plt.plot(resbabsrmag[resbgiantsel], resbgiantsep, 'r^', alpha=0.4, label=r'$s_{i,\rm\, fit}$ for RESOLVE-B Giant Galaxies (interpolated, $M_r \leq -19.4$)')
    plt.xlabel("Absolute $M_r$ of Giant Galaxy")
    plt.ylabel(r"$s_i$ - Separation used for Galaxy $i$ in Giant-Only FoF [Mpc/h]")
    plt.legend(loc='best')
    plt.xlim(-24,-19)
    plt.ylim(0,70)
    plt.gca().invert_xaxis()
    plt.savefig("images/meansep_M_r_plot.jpg")
    plt.savefig("paper1plots/meansep_M_r_plot.pdf")
    plt.show()

    # (c) perform giant-only FoF on ECO
    blos = 1.1
    bperp = 0.07 # from Duarte & Mamon 2014
    ADAPTIVE_OPTION=1
    ecogiantfofid = fof.fast_fof(ecoradeg[ecogiantsel], ecodedeg[ecogiantsel], ecocz[ecogiantsel], bperp, blos, (1-ADAPTIVE_OPTION)*meansep0+ADAPTIVE_OPTION*ecogiantsep) # meansep0 if fixed LL
    ecog3grp[ecogiantsel] = ecogiantfofid
    resbana_g3grp[ecogiantsel] = ecogiantfofid # RESOLVE-B analogue dataset
    # (d) perform giant-only FoF on RESOLVE-B
    resbgiantfofid = fof.fast_fof(resbradeg[resbgiantsel], resbdedeg[resbgiantsel], resbcz[resbgiantsel], bperp, blos, (1-ADAPTIVE_OPTION)*meansep0+ADAPTIVE_OPTION*resbgiantsep)
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
    plt.savefig("paper1plots/giantonlymult.pdf")
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
    poptrproj, jk = curve_fit(giantmodel, uniqecogiantgrpn[keepcalsel], median_relprojdist, sigma=rproj_median_error)#, p0=[0.1, -2, 3, -0.1])
    poptdvproj,jk = curve_fit(giantmodel, uniqecogiantgrpn[keepcalsel], median_relvel, sigma=dvproj_median_error)#, p0=[160,6.5,45,-600]) 
    rproj_boundary = lambda N: 3*giantmodel(N, *poptrproj) #3*(rprojslope*N+rprojint)
    vproj_boundary = lambda N: 4.5*giantmodel(N, *poptdvproj) #4.5*(dvprojslope*N+dvprojint)
    assert ((rproj_boundary(1)>0) and (vproj_boundary(1)>0)), "Cannot extrapolate Rproj or Vproj to N=1"
    
    # get virial radii from abundance matching to giant-only groups
    gihaloid, gilogmh, gir280b, gihalovdisp = ic.HAMwrapper(ecoradeg[ecogiantsel], ecodedeg[ecogiantsel], ecocz[ecogiantsel], ecoabsrmag[ecogiantsel], ecog3grp[ecogiantsel],\
                                                                ecovolume, inputfilename=None, outputfilename=None)
    gilogmh = np.log10(10**gilogmh / fof.getmhoffset(280,337,1,1,6)) # convert to 337b 
    gihalorvir = (3*(10**gilogmh) / (4*np.pi*337*0.3*2.77e11) )**(1/3.)
    gihalon = fof.multiplicity_function(np.sort(ecog3grp[ecogiantsel]), return_by_galaxy=False)
    plt.figure()
    plt.plot(gihalon, gihalorvir, 'k.')
    plt.show()

    fig, (ax,ax1) = plt.subplots(ncols=2, figsize=(13,5))
    sel = (ecogiantgrpn>1)
    ax1.scatter(gihalon, gihalovdisp, marker='D', color='purple', label=r'ECO HAM Velocity Dispersion')
    ax1.plot(ecogiantgrpn[sel], relvel[sel], 'r.', alpha=0.2, label='ECO Giant Galaxies')
    ax1.errorbar(uniqecogiantgrpn[keepcalsel], median_relvel, fmt='k^', label=r'$\Delta v_{\rm proj}$ (Median of $\Delta v_{\rm proj,\, gal}$)',yerr=dvproj_median_error)
    tx = np.linspace(1,max(ecogiantgrpn),1000)
    ax1.plot(tx, giantmodel(tx, *poptdvproj), label=r'$1\Delta v_{\rm proj}^{\rm fit}$')
    ax1.plot(tx, 4.5*giantmodel(tx, *poptdvproj), 'g',  label=r'$4.5\Delta v_{\rm proj}^{\rm fit}$', linestyle='-.')
    ax1.set_xlim(0,20)
    ax1.set_ylim(0,1000)
    ax1.set_xticks(np.arange(0,22,2))
    ax1.set_xlabel("Number of Giant Members")
    ax1.set_ylabel("Relative Velocity to Group Center [km/s]")
    ax1.legend(loc='best')

    ax.scatter(gihalon, gihalorvir, marker='D', color='purple', label=r'ECO Group Virial Radii')
    ax.plot(ecogiantgrpn[sel], relprojdist[sel], 'r.', alpha=0.2, label='ECO Giant Galaxies')
    ax.errorbar(uniqecogiantgrpn[keepcalsel], median_relprojdist, fmt='k^', label=r'$R_{\rm proj}$ (Median of $R_{\rm proj,\, gal}$)',yerr=rproj_median_error)
    ax.plot(tx, giantmodel(tx, *poptrproj), label=r'$1R_{\rm proj}^{\rm fit}$')
    ax.plot(tx, 3*giantmodel(tx, *poptrproj), 'g', label=r'$3R_{\rm proj}^{\rm fit}$', linestyle='-.')
    ax.set_xlabel("Number of Giant Members in Galaxy's Group")
    ax.set_ylabel("Projected Distance from Giant to Group Center [Mpc/h]")
    ax.legend(loc='best')
    ax.set_xlim(0,20)
    ax.set_ylim(0,1.5)
    ax.set_xticks(np.arange(0,22,2))
    plt.savefig("images/rproj_calibration_assoc.jpg")
    plt.savefig("paper1plots/rproj_calibration_assoc.pdf")
    plt.show()

    ####################################
    # Step 5: Association of Dwarfs
    ####################################
    ecodwarfsel = (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.33) & (ecocz>2530) & (ecocz<8000)
    resbdwarfsel = (resbabsrmag>-19.4) & (resbabsrmag<=-17.0) & (resbcz>4250) & (resbcz<7300)
    resbana_dwarfsel = (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.0) & (ecocz>2530) & (ecocz<8000)    

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
    ecogdsel = np.logical_not(np.logical_or(ecog3grp==-99., ((ecogdgrpn==1) & (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.33)))) #-17.33 not -17.0 (2/22/21)
    ecogdgrpra, ecogdgrpdec, ecogdgrpcz = fof.group_skycoords(ecoradeg[ecogdsel], ecodedeg[ecogdsel], ecocz[ecogdsel], ecog3grp[ecogdsel])
    ecogdrelvel = np.abs(ecogdgrpcz - ecocz[ecogdsel])
    ecogdrelprojdist = (ecogdgrpcz + ecocz[ecogdsel])/100. * ic.angular_separation(ecogdgrpra, ecogdgrpdec, ecoradeg[ecogdsel], ecodedeg[ecogdsel])/2.0
    ecogdn = ecogdgrpn[ecogdsel]
    ecogdtotalmag = ic.get_int_mag(ecoabsrmag[ecogdsel], ecog3grp[ecogdsel])
   
    magbins=np.arange(-24,-19,0.25)
    binsel = np.where(np.logical_and(ecogdn>1, ecogdtotalmag>-24))
    gdmedianrproj, magbincenters, agbinedges, jk = center_binned_stats(ecogdtotalmag[binsel], ecogdrelprojdist[binsel], np.median, bins=magbins)
    gdmedianrproj_err, jk, jk, jk = center_binned_stats(ecogdtotalmag[binsel], ecogdrelprojdist[binsel], sigmarange, bins=magbins)
    gdmedianrelvel, jk, jk, jk = center_binned_stats(ecogdtotalmag[binsel], ecogdrelvel[binsel], np.median, bins=magbins)
    gdmedianrelvel_err, jk, jk, jk = center_binned_stats(ecogdtotalmag[binsel], ecogdrelvel[binsel], sigmarange, bins=magbins)
    nansel = np.isnan(gdmedianrproj)
    if ADAPTIVE_OPTION: 
        guess=None
    else:
        guess=[1e-5, 0.4, 0.2]
    poptr, pcovr = curve_fit(decayexp, magbincenters[~nansel], gdmedianrproj[~nansel], p0=guess)
    print("guess:", poptr)
    poptv, pcovv = curve_fit(decayexp, magbincenters[~nansel], gdmedianrelvel[~nansel], p0=[3e-5,4e-1,5e-03])#,1])

    tx = np.linspace(-27,-17,100)
    fig, (ax,ax1) = plt.subplots(ncols=2, figsize=(12,6))
    ax.plot(ecogdtotalmag[binsel], ecogdrelprojdist[binsel], 'k.', alpha=0.2, label='ECO Galaxies in N>1 Giant+Dwarf Groups')
    ax.errorbar(magbincenters, gdmedianrproj, yerr=gdmedianrproj_err, fmt='r^', label='Medians')
    ax.plot(tx, 1*decayexp(tx,*poptr), label='Fit to Medians')
    ax.plot(tx, 3*decayexp(tx,*poptr), label=r'3 times Fit to Medians')
    ax.set_xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    ax.set_ylabel("Projected Distance from Galaxy to Group Center [Mpc/h]")
    ax.legend(loc='best')
    ax.set_xlim(-25,-19)
    ax.set_ylim(0,1.3)
    ax.invert_xaxis()

    ax1.plot(ecogdtotalmag[binsel], ecogdrelvel[binsel], 'k.', alpha=0.2, label='Mock Galaxies in N=2 Giant+Dwarf Groups')
    ax1.errorbar(magbincenters, gdmedianrelvel, yerr=gdmedianrelvel_err, fmt='r^',label='Medians')
    ax1.plot(tx, decayexp(tx, *poptv), label='Fit to Medians')
    ax1.plot(tx, 4.5*decayexp(tx, *poptv), label='4.5 times Fit to Medians')
    ax1.set_ylabel("Relative Velocity between Galaxy and Group Center")
    ax1.set_xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    ax1.set_xlim(-25,-19)
    ax1.set_ylim(0,1000)
    ax1.invert_xaxis()
    plt.savefig("paper1plots/itercombboundaries.pdf")
    plt.show()

    rproj_for_iteration = lambda M: 3*decayexp(M, *poptr)
    vproj_for_iteration = lambda M: 4.5*decayexp(M, *poptv)

    # --------------- now need to do this calibration for the RESOLVE-B analogue dataset, down to -17.0) -------------$
    resbana_gdgrpn = fof.multiplicity_function(resbana_g3grp, return_by_galaxy=True)
    #resbana_gdsel = np.logical_not((resbana_gdgrpn==1) & (ecoabsrmag>-19.4) & (resbana_g3grp!=-99.) & (resbana_g3grp>0)) # select galaxies that AREN'T ungrouped dwarfs
    resbana_gdsel = np.logical_not(np.logical_or(resbana_g3grp==-99., ((resbana_gdgrpn==1) & (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.0))))
    resbana_gdgrpra, resbana_gdgrpdec, resbana_gdgrpcz = fof.group_skycoords(ecoradeg[resbana_gdsel], ecodedeg[resbana_gdsel], ecocz[resbana_gdsel], resbana_g3grp[resbana_gdsel])
    resbana_gdrelvel = np.abs(resbana_gdgrpcz - ecocz[resbana_gdsel])
    resbana_gdrelprojdist = (resbana_gdgrpcz + ecocz[resbana_gdsel])/100. * ic.angular_separation(resbana_gdgrpra, resbana_gdgrpdec, ecoradeg[resbana_gdsel], ecodedeg[resbana_gdsel])/2.0

    resbana_gdn = resbana_gdgrpn[resbana_gdsel]
    resbana_gdtotalmag = ic.get_int_mag(ecoabsrmag[resbana_gdsel], resbana_g3grp[resbana_gdsel])

    magbins2=np.arange(-24,-19,0.25)
    binsel2 = np.where(np.logical_and(resbana_gdn>1, resbana_gdtotalmag>-24))
    gdmedianrproj, magbincenters, magbinedges, jk = center_binned_stats(resbana_gdtotalmag[binsel2], resbana_gdrelprojdist[binsel2], np.median, bins=magbins2)
    gdmedianrproj_err, jk, jk, jk = center_binned_stats(resbana_gdtotalmag[binsel2], resbana_gdrelprojdist[binsel2], sigmarange, bins=magbins2)
    gdmedianrelvel, jk, jk, jk = center_binned_stats(resbana_gdtotalmag[binsel2], resbana_gdrelvel[binsel2], np.median, bins=magbins2)
    gdmedianrelvel_err, jk, jk, jk = center_binned_stats(resbana_gdtotalmag[binsel2], resbana_gdrelvel[binsel2], sigmarange, bins=magbins2)
    nansel = np.isnan(gdmedianrproj)
    poptr_resbana, jk = curve_fit(decayexp, magbincenters[~nansel], gdmedianrproj[~nansel], p0=poptr)
    poptv_resbana, jk = curve_fit(decayexp, magbincenters[~nansel], gdmedianrelvel[~nansel], p0=poptv)

    tx = np.linspace(-27,-16,100)
    plt.figure()
    plt.plot(resbana_gdtotalmag[binsel2], resbana_gdrelprojdist[binsel2], 'k.', alpha=0.2, label='Mock Galaxies in N>1 Giant+Dwarf Groups')
    plt.errorbar(magbincenters, gdmedianrproj, yerr=gdmedianrproj_err, fmt='r^', label='Medians')
    plt.plot(tx, decayexp(tx,*poptr_resbana), label='Fit to Medians')
    plt.plot(tx, 3*decayexp(tx,*poptr_resbana), label='3 times Fit to Medians')
    plt.xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    plt.ylabel("Projected Distance from Galaxy to Group Center [Mpc/h]")
    plt.legend(loc='best')
    plt.xlim(-25,-19)
    plt.ylim(0,1.3)
    #plt.ylim(0,1.3)
    plt.gca().invert_xaxis()
    plt.show()

    plt.figure()
    plt.plot(resbana_gdtotalmag[binsel2], resbana_gdrelvel[binsel2], 'k.', alpha=0.2, label='Mock Galaxies in N=2 Giant+Dwarf Groups')
    plt.errorbar(magbincenters, gdmedianrelvel, yerr=gdmedianrelvel_err, fmt='r^',label='Medians')
    plt.plot(tx, decayexp(tx, *poptv_resbana), label='Fit to Medians')
    plt.plot(tx, 4.5*decayexp(tx, *poptv_resbana), label='4.5 times Fit to Medians')
    plt.ylabel("Relative Velocity between Galaxy and Group Center")
    plt.xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    plt.xlim(-25,-19)
    plt.ylim(0,1100)
    plt.gca().invert_xaxis()
    plt.show()

    rproj_for_iteration_resbana = lambda M: 3*decayexp(M, *poptr_resbana)
    vproj_for_iteration_resbana = lambda M: 4.5*decayexp(M, *poptv_resbana)


    ###########################################################
    # Step 7: Iterative Combination of Dwarf Galaxies
    ###########################################################
    assert (ecog3grp[(ecoabsrmag<=-19.4) & (ecocz<8000) & (ecocz>2530)]!=-99.).all(), "Not all giants are grouped."
    ecogrpnafterassoc = fof.multiplicity_function(ecog3grp, return_by_galaxy=True)
    resbgrpnafterassoc = fof.multiplicity_function(resbg3grp, return_by_galaxy=True)
    resbana_grpnafterassoc = fof.multiplicity_function(resbana_g3grp, return_by_galaxy=True)

    eco_ungroupeddwarf_sel = (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.33) & (ecocz<8000) & (ecocz>2530) & (ecogrpnafterassoc==1)
    ecoitassocid = ic.iterative_combination(ecoradeg[eco_ungroupeddwarf_sel], ecodedeg[eco_ungroupeddwarf_sel], ecocz[eco_ungroupeddwarf_sel], ecoabsrmag[eco_ungroupeddwarf_sel],\
                                           rproj_for_iteration, vproj_for_iteration, starting_id=np.max(ecog3grp)+1, centermethod='arithmetic')
    
    resb_ungroupeddwarf_sel = (resbabsrmag>-19.4) & (resbabsrmag<=-17.0) & (resbcz<7300) & (resbcz>4250) & (resbgrpnafterassoc==1)
    resbitassocid = ic.iterative_combination(resbradeg[resb_ungroupeddwarf_sel], resbdedeg[resb_ungroupeddwarf_sel], resbcz[resb_ungroupeddwarf_sel], resbabsrmag[resb_ungroupeddwarf_sel],\
                                            rproj_for_iteration, vproj_for_iteration, starting_id=np.max(resbg3grp)+1, centermethod='arithmetic')
    
    resbana_ungroupeddwarf_sel = (ecoabsrmag>-19.4) & (ecoabsrmag<=-17.0) & (ecocz<8000) & (ecocz>2530) & (resbana_grpnafterassoc==1)
    resbana_itassocid = ic.iterative_combination(ecoradeg[resbana_ungroupeddwarf_sel], ecodedeg[resbana_ungroupeddwarf_sel], ecocz[resbana_ungroupeddwarf_sel], ecoabsrmag[resbana_ungroupeddwarf_sel],\
                                                 rproj_for_iteration_resbana, vproj_for_iteration_resbana, starting_id=np.max(resbana_g3grp)+1, centermethod='arithmetic')

    ecog3grp[eco_ungroupeddwarf_sel] = ecoitassocid
    resbg3grp[resb_ungroupeddwarf_sel] = resbitassocid
    resbana_g3grp[resbana_ungroupeddwarf_sel] = resbana_itassocid
    #plt.figure()
    #plt.hist(fof.multiplicity_function(ecoitassocid, return_by_galaxy=False), log=True)
    #plt.hist(fof.multiplicity_function(resbitassocid, return_by_galaxy=False), log=True, histtype='step')
    #plt.show()
    
    fig, ax = plt.subplots(figsize=(8,8))
    binv = np.arange(0.5,1200.5,3)
    ax.hist(fof.multiplicity_function(ecog3grp[ecog3grp!=-99.], return_by_galaxy=False), bins=binv, log=True, label='ECO Groups', histtype='step', linewidth=3)
    ax.hist(fof.multiplicity_function(resbg3grp[resbg3grp!=-99.], return_by_galaxy=False), bins=binv, log=True, label='RESOLVE-B Groups', histtype='step', hatch='\\')
    ax.set_xlabel("Number of Giant + Dwarf Group Members")
    ax.set_ylabel("Number of Groups")
    ax.legend(loc='best')
    ax.set_xlim(0,100)

    ax2 = plt.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax, [0.4,0.2,0.5,0.5])
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')
   
    ax2.hist(fof.multiplicity_function(ecoitassocid, return_by_galaxy=False), bins=binv, log=True, label='ECO Groups', histtype='step', linewidth=3)
    ax2.hist(fof.multiplicity_function(resbitassocid, return_by_galaxy=False), bins=binv, log=True, label='RESOLVE-B Groups', histtype='step', hatch='\\')

    plt.show()
 
    ############################################################
    # Step 8: Halo Abundance Matching
    ###########################################################
    # --- for RESOLVE-B analogue ----#
    resbana_hamsel = (resbana_g3grp!=-99.)
    resbana_haloid, resbana_halomass, jk, jk = ic.HAMwrapper(ecoradeg[resbana_hamsel], ecodedeg[resbana_hamsel], ecocz[resbana_hamsel], ecoabsrmag[resbana_hamsel], resbana_g3grp[resbana_hamsel],\
                                                                ecovolume, inputfilename=None, outputfilename=None)
    resbana_halomass = np.log10(10**resbana_halomass/fof.getmhoffset(280,337,1,1,6)) # convert to 337b
    junk, uniqindex = np.unique(resbana_g3grp[resbana_hamsel], return_index=True)
    resbana_intmag = ic.get_int_mag(ecoabsrmag[resbana_hamsel], resbana_g3grp[resbana_hamsel])[uniqindex]
    sortind = np.argsort(resbana_intmag)
    sortedmag = resbana_intmag[sortind]
    resbcubicspline = interp1d(sortedmag, resbana_halomass[sortind], fill_value='extrapolate') 
    
    resbintmag = ic.get_int_mag(resbabsrmag[resbg3grp!=-99.], resbg3grp[resbg3grp!=-99.])
    resbg3logmh[resbg3grp!=-99.] = resbcubicspline(resbintmag)-np.log10(0.7)
 
    # ---- for ECO ----- #
    ecohamsel = (ecog3grp!=-99.)
    haloid, halomass, junk, junk = ic.HAMwrapper(ecoradeg[ecohamsel], ecodedeg[ecohamsel], ecocz[ecohamsel], ecoabsrmag[ecohamsel], ecog3grp[ecohamsel],\
                                                     ecovolume, inputfilename=None, outputfilename=None)
    junk, uniqindex = np.unique(ecog3grp[ecohamsel], return_index=True)
    halomass = np.log10(10**halomass/fof.getmhoffset(280,337,1,1,6)) # convert to 337b
    halomass = halomass-np.log10(0.7)
    for i,idv in enumerate(haloid):
        sel = np.where(ecog3grp==idv)
        ecog3logmh[sel] = halomass[i] # m337b
    
    # calculate Rvir in arcsec
    ecog3rvir = (3*(10**ecog3logmh) / (4*np.pi*337*0.3*1.36e11) )**(1/3.)#/(ecog3grpcz/70.) * 206265
    resbg3rvir = (3*(10**resbg3logmh) / (4*np.pi*337*0.3*1.36e11))**(1/3.)#/(resbg3grpcz/70.) * 206265

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
    plt.savefig("paper1plots/hamLrrelationG3.pdf")
    plt.show()

    ########################################
    # (9) Output arrays     
    ########################################
    # ---- first get the quantities for ECO ---- #
    #eco_in_gf = np.where(ecog3grp!=-99.)
    ecog3grpn = fof.multiplicity_function(ecog3grp, return_by_galaxy=True)
    ecog3grpngi = np.zeros(len(ecog3grpn))
    ecog3grpndw = np.zeros(len(ecog3grpn))
    for uid in np.unique(ecog3grp):
        grpsel = np.where(ecog3grp==uid)
        gisel = np.where(np.logical_and((ecog3grp==uid),(ecoabsrmag<=-19.4)))
        dwsel = np.where(np.logical_and((ecog3grp==uid), (ecoabsrmag>-19.4)))
        if len(gisel[0])>0.:
            ecog3grpngi[grpsel] = len(gisel[0])
        if len(dwsel[0])>0.:
            ecog3grpndw[grpsel] = len(dwsel[0])

    ecog3grpradeg, ecog3grpdedeg, ecog3grpcz = fof.group_skycoords(ecoradeg, ecodedeg, ecocz, ecog3grp)
    ecog3rproj = fof.get_grprproj_e17(ecoradeg, ecodedeg, ecocz, ecog3grp, h=0.7) / (ecog3grpcz/70.) * 206265 # in arcsec
    ecog3fc = fof.get_central_flag(ecoabsrmag, ecog3grp) 
    ecog3router = fof.get_outermost_galradius(ecoradeg, ecodedeg, ecocz, ecog3grp) # in arcsec
    ecog3router[(ecog3grpngi+ecog3grpndw)==1] = 0.
    junk, ecog3vdisp = fof.get_rproj_czdisp(ecoradeg, ecodedeg, ecocz, ecog3grp)
    ecog3rvir = ecog3rvir*206265/(ecog3grpcz/70.) 

    outofsample = (ecog3grp==-99.)
    ecog3grpn[outofsample]=-99.
    ecog3grpngi[outofsample]=-99.
    ecog3grpndw[outofsample]=-99.
    ecog3grpradeg[outofsample]=-99.
    ecog3grpdedeg[outofsample]=-99.
    ecog3grpcz[outofsample]=-99.
    ecog3logmh[outofsample]=-99.
    ecog3rvir[outofsample]=-99.
    ecog3rproj[outofsample]=-99.
    ecog3fc[outofsample]=-99.
    ecog3router[outofsample]=-99.
    ecog3vdisp[outofsample]=-99.
    insample = ecog3grpn!=-99.

    ecodata['g3grp_l'] = ecog3grp
    ecodata['g3grpradeg_l'] = ecog3grpradeg
    ecodata['g3grpdedeg_l'] = ecog3grpdedeg
    ecodata['g3grpcz_l'] = ecog3grpcz
    ecodata['g3grpndw_l'] = ecog3grpndw
    ecodata['g3grpngi_l'] = ecog3grpngi
    ecodata['g3logmh_l'] = ecog3logmh
    ecodata['g3rvir_l'] = ecog3rvir
    ecodata['g3rproj_l'] = ecog3rproj
    ecodata['g3router_l'] = ecog3router
    ecodata['g3fc_l'] = ecog3fc
    ecodata['g3vdisp_l'] = ecog3vdisp
    ecodata.to_csv("ECOdata_G3catalog_luminosity.csv", index=False)    

    # ------ now do RESOLVE
    sz = len(resolvedata)
    resolvename = np.array(resolvedata.name)
    resolveg3grp = np.full(sz, -99.)
    resolveg3grpngi = np.full(sz, -99.)
    resolveg3grpndw = np.full(sz, -99.)
    resolveg3grpradeg = np.full(sz, -99.)
    resolveg3grpdedeg = np.full(sz, -99.)
    resolveg3grpcz = np.full(sz, -99.)
    resolveg3intmag = np.full(sz, -99.)
    resolveg3intmstar = np.full(sz, -99.)
    resolveg3logmh = np.full(sz, -99.)
    resolveg3rvir = np.full(sz, -99.)
    resolveg3rproj = np.full(sz,-99.)
    resolveg3fc = np.full(sz,-99.)
    resolveg3router = np.full(sz,-99.)
    resolveg3vdisp = np.full(sz,-99.)

    resbg3grpngi = np.full(len(resbg3grp), 0)
    resbg3grpndw = np.full(len(resbg3grp), 0) # was originally filled with -99 (ZH edited 7/8/21)
    for uid in np.unique(resbg3grp):
        grpsel = np.where(resbg3grp==uid)
        gisel = np.where(np.logical_and((resbg3grp==uid),(resbabsrmag<=-19.4)))
        dwsel = np.where(np.logical_and((resbg3grp==uid), (resbabsrmag>-19.4)))
        if len(gisel[0])>0.:
            resbg3grpngi[grpsel] = len(gisel[0])
        if len(dwsel[0])>0.:
            resbg3grpndw[grpsel] = len(dwsel[0])

    resbg3grpradeg, resbg3grpdedeg, resbg3grpcz = fof.group_skycoords(resbradeg, resbdedeg, resbcz, resbg3grp)
    resbg3intmag = ic.get_int_mag(resbabsrmag, resbg3grp)
    resbg3intmstar = ic.get_int_mass(resblogmstar, resbg3grp)
    resbg3rproj = fof.get_grprproj_e17(resbradeg, resbdedeg, resbcz, resbg3grp, h=0.7) / (resbg3grpcz/70.) * 206265 # in arcsec 
    resbg3fc = fof.get_central_flag(resbabsrmag, resbg3grp)
    resbg3router = fof.get_outermost_galradius(resbradeg, resbdedeg, resbcz, resbg3grp) # in arcsec
    resbg3router[(resbg3grpngi+resbg3grpndw)==1] = 0.
    junk, resbg3vdisp = fof.get_rproj_czdisp(resbradeg, resbdedeg, resbcz, resbg3grp)
    resbg3rvir = resbg3rvir*206265/(resbg3grpcz/70.)
    print(resbg3rvir)  

    outofsample = (resbg3grp==-99.)
    resbg3grpngi[outofsample]=-99.
    resbg3grpndw[outofsample]=-99.
    resbg3grpradeg[outofsample]=-99.
    resbg3grpdedeg[outofsample]=-99.
    resbg3grpcz[outofsample]=-99.
    resbg3intmag[outofsample]=-99.
    resbg3intmstar[outofsample]=-99.
    resbg3logmh[outofsample]=-99.
    resbg3rvir[outofsample]=-99.
    resbg3rproj[outofsample]=-99.
    resbg3router[outofsample]=-99.
    resbg3fc[outofsample]=-99.
    resbg3vdisp[outofsample]=-99.
    for i,nm in enumerate(resolvename):
        if nm.startswith('rs'):
            sel_in_eco = np.where(ecoresname==nm)
            resolveg3grp[i] = ecog3grp[sel_in_eco]
            resolveg3grpngi[i] = ecog3grpngi[sel_in_eco]
            resolveg3grpndw[i] = ecog3grpndw[sel_in_eco]
            resolveg3grpradeg[i] = ecog3grpradeg[sel_in_eco]
            resolveg3grpdedeg[i] = ecog3grpdedeg[sel_in_eco]
            resolveg3grpcz[i] = ecog3grpcz[sel_in_eco]
            resolveg3intmag[i] = ecog3intmag[sel_in_eco]
            resolveg3intmstar[i] = ecog3intmstar[sel_in_eco]
            resolveg3logmh[i] = ecog3logmh[sel_in_eco]
            resolveg3rvir[i] = ecog3rvir[sel_in_eco]
            resolveg3rproj[i] = ecog3rproj[sel_in_eco]
            resolveg3fc[i] = ecog3fc[sel_in_eco]
            resolveg3router[i]=ecog3router[sel_in_eco]
            resolveg3vdisp[i]=ecog3vdisp[sel_in_eco]
        elif nm.startswith('rf'):
            sel_in_resb = np.where(resbname==nm)
            resolveg3grp[i] = resbg3grp[sel_in_resb]
            resolveg3grpngi[i] = resbg3grpngi[sel_in_resb]
            resolveg3grpndw[i] = resbg3grpndw[sel_in_resb]
            resolveg3grpradeg[i] = resbg3grpradeg[sel_in_resb]
            resolveg3grpdedeg[i] = resbg3grpdedeg[sel_in_resb]
            resolveg3grpcz[i] = resbg3grpcz[sel_in_resb]
            resolveg3intmag[i] = resbg3intmag[sel_in_resb]
            resolveg3intmstar[i] = resbg3intmstar[sel_in_resb]
            resolveg3logmh[i] = resbg3logmh[sel_in_resb]
            resolveg3rvir[i] = resbg3rvir[sel_in_resb]
            resolveg3rproj[i] = resbg3rproj[sel_in_resb]
            resolveg3fc[i] = resbg3fc[sel_in_resb]
            resolveg3router[i] = resbg3router[sel_in_resb]
            resolveg3vdisp[i] = resbg3vdisp[sel_in_resb]
        else:
            assert False, nm+" not in RESOLVE"


    resolveg3router[np.where((resolveg3grpngi+resolveg3grpndw)==0)]=0.
    resolvedata['g3grp_l'] = resolveg3grp
    resolvedata['g3grpngi_l'] = resolveg3grpngi
    resolvedata['g3grpndw_l'] = resolveg3grpndw
    resolvedata['g3grpradeg_l'] = resolveg3grpradeg
    resolvedata['g3grpdedeg_l'] = resolveg3grpdedeg
    resolvedata['g3grpcz_l'] = resolveg3grpcz
    resolvedata['g3logmh_l'] = resolveg3logmh
    resolvedata['g3rvir_l'] = resolveg3rvir
    resolvedata['g3rproj_l'] = resolveg3rproj
    resolvedata['g3router_l'] = resolveg3router
    resolvedata['g3fc_l'] = resolveg3fc
    resolvedata['g3vdisp_l'] = resolveg3vdisp
    resolvedata.to_csv("RESOLVEdata_G3catalog_luminosity.csv", index=False)
