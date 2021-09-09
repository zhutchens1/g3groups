"""
Zackary Hutchens - November 2020

This program creates baryonic mass-selected  group catalogs for ECO/RESOLVE-G3 using the new algorithm, described in the readme markdown.

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
(6) Based on giant+dwarf groups, calibrate boundaries (as function of giant+dwarf integrated baryonic mass) for iterative combination
(7) Iterative combination on remaining ungrouped dwarf galaxies
(8) halo mass assignment
(9) Finalize arrays + output
"""
import virtools as vz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit
from center_binned_stats import center_binned_stats
import foftools as fof
import iterativecombination as ic
from smoothedbootstrap import smoothedbootstrap as sbs
import sys
from lss_dens import lss_dens_by_galaxy

def giantmodel(x, a, b):
    return np.abs(a)*np.log(np.abs(b)*x+1)

def exp(x, a, b, c):
    return np.abs(a)*np.exp(1*np.abs(b)*x + c)

def sepmodel(x, a, b, c, d, e):
    #return np.abs(a)*np.exp(-1*np.abs(b)*x + c)+d
    #return a*(x**3)+b*(x**2)+c*x+d
    return a*(x**4)+b*(x**3)+c*(x**2)+(d*x)+e

def sigmarange(x):
    q84, q16 = np.percentile(x, [84 ,16])
    return (q84-q16)/2

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
    ecologmstar = np.array(ecodata.logmstar)
    ecologmgas = np.array(ecodata.logmgas)
    ecologmbary = np.log10(10.**ecologmstar+10.**ecologmgas)
    ecourcolor = np.array(ecodata.modelu_rcorr)
    ecog3grp = np.full(ecosz, -99.) # id number of g3 group
    ecog3grpn = np.full(ecosz, -99.) # multiplicity of g3 group
    ecog3grpradeg = np.full(ecosz,-99.) # ra of group center
    ecog3grpdedeg = np.full(ecosz,-99.) # dec of group center
    ecog3grpcz = np.full(ecosz,-99.) # cz of group center
    ecog3logmh = np.full(ecosz,-99.) # abundance-matched halo mass

    resbana_g3grp = np.full(ecosz,-99.) # for RESOLVE-B analogue dataset

    resbsz = int(len(resolvebdata))
    resbname = np.array(resolvebdata.name)
    resbradeg = np.array(resolvebdata.radeg)
    resbdedeg = np.array(resolvebdata.dedeg)
    resbcz = np.array(resolvebdata.cz)
    resblogmstar = np.array(resolvebdata.logmstar)
    resblogmgas = np.array(resolvebdata.logmgas)
    resblogmbary = np.log10(10.**resblogmstar+10.**resblogmgas)
    resburcolor = np.array(resolvebdata.modelu_rcorr)
    resbg3grp = np.full(resbsz, -99.)
    resbg3grpn = np.full(resbsz, -99.)
    resbg3grpradeg = np.full(resbsz, -99.)
    resbg3grpdedeg = np.full(resbsz, -99.)
    resbg3grpcz = np.full(resbsz, -99.)
    resbg3logmh = np.full(resbsz, -99.)

    ####################################
    # Step 3: Giant-Only FOF
    ####################################
    ecogiantsel = (ecologmbary>=9.9) & (ecocz>2530.) & (ecocz<8000.)
    # (a) compute sep values for eco giants
    ecovolume = 192351.36 # Mpc^3 with h=1 **
    meansep0 = (ecovolume/len(ecologmbary[ecogiantsel]))**(1/3.)

    # (c) perform giant-only FoF on ECO
    blos = 1.1
    bperp = 0.07 # from Duarte & Mamon 2014
    ecogiantfofid = fof.fast_fof(ecoradeg[ecogiantsel], ecodedeg[ecogiantsel], ecocz[ecogiantsel], bperp, blos, meansep0) # meansep0 if fixed LL
    ecog3grp[ecogiantsel] = ecogiantfofid
    resbana_g3grp[ecogiantsel] = ecogiantfofid # RESOLVE-B analogue dataset
    # (d) perform giant-only FoF on RESOLVE-B
    resbgiantsel = (resblogmbary>=9.9) & (resbcz>4250) & (resbcz<7300)
    resbgiantfofid = fof.fast_fof(resbradeg[resbgiantsel], resbdedeg[resbgiantsel], resbcz[resbgiantsel], bperp, blos, meansep0)
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
    plt.show()

    ##########################################
    # Step 4: Compute Association Boundaries
    ##########################################
    ecogiantgrpra, ecogiantgrpdec, ecogiantgrpcz = fof.group_skycoords(ecoradeg[ecogiantsel], ecodedeg[ecogiantsel], ecocz[ecogiantsel], ecogiantfofid)
    relvel = np.abs(ecogiantgrpcz - ecocz[ecogiantsel])
    relprojdist = (ecogiantgrpcz + ecocz[ecogiantsel])/100. * np.sin(ic.angular_separation(ecogiantgrpra, ecogiantgrpdec, ecoradeg[ecogiantsel], ecodedeg[ecogiantsel])/2.0)
    ecogiantgrpn = fof.multiplicity_function(ecogiantfofid, return_by_galaxy=True)
    uniqecogiantgrpn, uniqindex = np.unique(ecogiantgrpn, return_index=True)
    keepcalsel = np.where(uniqecogiantgrpn>1)

    median_relprojdist = np.array([np.median(relprojdist[np.where(ecogiantgrpn==sz)]) for sz in uniqecogiantgrpn[keepcalsel]])
    median_relvel = np.array([np.median(relvel[np.where(ecogiantgrpn==sz)]) for sz in uniqecogiantgrpn[keepcalsel]])

    rproj_median_error = np.std(np.array([sbs(relprojdist[np.where(ecogiantgrpn==sz)], 100000, np.median, kwargs=dict({'axis':1 })) for sz in uniqecogiantgrpn[keepcalsel]]), axis=1)
    dvproj_median_error = np.std(np.array([sbs(relvel[np.where(ecogiantgrpn==sz)], 100000, np.median, kwargs=dict({'axis':1})) for sz in uniqecogiantgrpn[keepcalsel]]), axis=1)

    #rprojslope, rprojint = np.polyfit(uniqecogiantgrpn[keepcalsel], median_relprojdist, deg=1, w=1/rproj_median_error)
    #dvprojslope, dvprojint = np.polyfit(uniqecogiantgrpn[keepcalsel], median_relvel, deg=1, w=1/dvproj_median_error)
    poptrproj, jk = curve_fit(giantmodel, uniqecogiantgrpn[keepcalsel], median_relprojdist, sigma=rproj_median_error)
    poptdvproj,jk = curve_fit(giantmodel, uniqecogiantgrpn[keepcalsel], median_relvel, sigma=dvproj_median_error)#, p0=[160,6.5,45,-600])
    rproj_boundary = lambda N: 3*giantmodel(N, *poptrproj) #3*(rprojslope*N+rprojint)
    vproj_boundary = lambda N: 4.5*giantmodel(N, *poptdvproj) #4.5*(dvprojslope*N+dvprojint)
    assert rproj_boundary(1)>0, "Rproj boundary fit can't be extrapolated to N=1"
    assert vproj_boundary(1)>0, "dv proj boundary fit can't be extrapolated to N=1"

    # get virial radii from abundance matching to giant-only groups
    gihaloid, gilogmh, gir280, gihalovdisp = ic.HAMwrapper(ecoradeg[ecogiantsel], ecodedeg[ecogiantsel], ecocz[ecogiantsel], ecologmbary[ecogiantsel], ecog3grp[ecogiantsel],\
                                                                ecovolume, inputfilename=None, outputfilename=None)
    gilogmh = np.log10(10**gilogmh)#no longer needed as of 7/29: /fof.getmhoffset(280,337,1,1,6))
    gihalorvir = (3*(10**gilogmh) / (4*np.pi*337*0.3*2.77e11) )**(1/3.)
    gihalon = fof.multiplicity_function(np.sort(ecog3grp[ecogiantsel]), return_by_galaxy=False)
    plt.figure()
    plt.plot(gihalon, gihalorvir, 'k.')
    plt.show()

    plt.figure()
    sel = (ecogiantgrpn>1)
    plt.scatter(gihalon, gihalovdisp, marker='D', color='purple', label=r'ECO HAM Velocity Dispersion')
    plt.plot(ecogiantgrpn[sel], relvel[sel], 'r.', alpha=0.2, label='ECO Giant Galaxies')
    plt.errorbar(uniqecogiantgrpn[keepcalsel], median_relvel, fmt='k^', label=r'$\Delta v_{\rm proj}$ (Median of $\Delta v_{\rm proj,\, gal}$)',yerr=dvproj_median_error)
    tx = np.linspace(1,max(ecogiantgrpn),1000)
    plt.plot(tx, giantmodel(tx, *poptdvproj), label=r'$1\Delta v_{\rm proj}^{\rm fit}$')
    plt.plot(tx, 4.5*giantmodel(tx, *poptdvproj), 'g',  label=r'$4.5\Delta v_{\rm proj}^{\rm fit}$', linestyle='-.')
    plt.xlabel("Number of Giant Members")
    plt.ylabel("Relative Velocity to Group Center [km/s]")
    plt.legend(loc='best')
    plt.show()

    plt.clf()
    plt.scatter(gihalon, gihalorvir, marker='D', color='purple', label=r'ECO Group Virial Radii')
    plt.plot(ecogiantgrpn[sel], relprojdist[sel], 'r.', alpha=0.2, label='ECO Giant Galaxies')
    plt.errorbar(uniqecogiantgrpn[keepcalsel], median_relprojdist, fmt='k^', label=r'$R_{\rm proj}$ (Median of $R_{\rm proj,\, gal}$)',yerr=rproj_median_error)
    plt.plot(tx, giantmodel(tx, *poptrproj), label=r'$1R_{\rm proj}^{\rm fit}$')
    plt.plot(tx, 3*giantmodel(tx, *poptrproj), 'g', label=r'$3R_{\rm proj}^{\rm fit}$', linestyle='-.')
    plt.xlabel("Number of Giant Members in Galaxy's Group")
    plt.ylabel("Projected Distance from Giant to Group Center [Mpc/h]")
    plt.legend(loc='best')
    #plt.xlim(0,20)
    #plt.ylim(0,2.5)
    #plt.xticks(np.arange(0,22,2))
    plt.show()

    ####################################
    # Step 5: Association of Dwarfs
    ####################################
    ecodwarfsel = (ecologmbary<9.9) & (ecologmbary>=9.4) & (ecocz>2530) & (ecocz<8000)
    resbdwarfsel = (resblogmbary<9.9) & (resblogmbary>=9.1) & (resbcz>4250) & (resbcz<7300)
    resbana_dwarfsel = (ecologmbary<9.9) & (ecologmbary>=9.1) & (ecocz>2530) & (ecocz<8000)

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
    #ecogdsel = np.logical_not((ecogdgrpn==1) & (ecologmbary>-19.4) & (ecog3grp>0)) # select galaxies that AREN'T ungrouped dwarfs
    ecogdsel = np.logical_not(np.logical_or(ecog3grp==-99., ((ecogdgrpn==1) & (ecologmbary<9.9) & (ecologmbary>=9.4))))
    ecogdgrpra, ecogdgrpdec, ecogdgrpcz = fof.group_skycoords(ecoradeg[ecogdsel], ecodedeg[ecogdsel], ecocz[ecogdsel], ecog3grp[ecogdsel])
    ecogdrelvel = np.abs(ecogdgrpcz - ecocz[ecogdsel])
    ecogdrelprojdist = (ecogdgrpcz + ecocz[ecogdsel])/100. * np.sin(ic.angular_separation(ecogdgrpra, ecogdgrpdec, ecoradeg[ecogdsel], ecodedeg[ecogdsel])/2.0)
    ecogdn = ecogdgrpn[ecogdsel]
    ecogdtotalmass = ic.get_int_mass(ecologmbary[ecogdsel], ecog3grp[ecogdsel])

    massbins=np.arange(9.9,14,0.15)
    binsel = np.where(np.logical_and(ecogdn>1, ecogdtotalmass<14))
    gdmedianrproj, massbincenters, massbinedges, jk = center_binned_stats(ecogdtotalmass[binsel], ecogdrelprojdist[binsel], np.median, bins=massbins)
    gdmedianrproj_err, jk, jk, jk = center_binned_stats(ecogdtotalmass[binsel], ecogdrelprojdist[binsel], sigmarange, bins=massbins)
    gdmedianrelvel, jk, jk, jk = center_binned_stats(ecogdtotalmass[binsel], ecogdrelvel[binsel], np.median, bins=massbins)
    gdmedianrelvel_err, jk, jk, jk = center_binned_stats(ecogdtotalmass[binsel], ecogdrelvel[binsel], sigmarange, bins=massbins)
    nansel = np.isnan(gdmedianrproj)
    if 0:
        guess=None
    else:
        guess=None# [-1,0.5,-6]#None#[1e-5, 0.4, 0.2, 1]
    poptr, pcovr = curve_fit(exp, massbincenters[~nansel], gdmedianrproj[~nansel], p0=guess, sigma=gdmedianrproj_err[~nansel])
    poptv, pcovv = curve_fit(exp, massbincenters[~nansel], gdmedianrelvel[~nansel], p0=[3e-5,4e-1,5e-03], sigma=gdmedianrelvel_err[~nansel])

    tx = np.linspace(7,15,100)
    plt.figure()
    plt.plot(ecogdtotalmass[binsel], ecogdrelprojdist[binsel], 'k.', alpha=0.2, label='ECO Galaxies in N>1 Giant+Dwarf Groups')
    plt.errorbar(massbincenters, gdmedianrproj, yerr=gdmedianrproj_err, fmt='r^', label='Medians')
    plt.plot(tx, exp(tx,*poptr), label='Fit to Medians')
    plt.plot(tx, 3*exp(tx,*poptr), label='3 times Fit to Medians')
    plt.xlabel(r"Integrated baryonic Mass of Giant + Dwarf Members")
    plt.ylabel("Projected Distance from Galaxy to Group Center [Mpc/h]")
    plt.legend(loc='best')
    plt.xlim(9.9,13)
    plt.ylim(0,3)
    plt.show()

    plt.figure()
    plt.plot(ecogdtotalmass[binsel], ecogdrelvel[binsel], 'k.', alpha=0.2, label='Mock Galaxies in N=2 Giant+Dwarf Groups')
    plt.errorbar(massbincenters, gdmedianrelvel, yerr=gdmedianrelvel_err, fmt='r^',label='Medians')
    plt.plot(tx, exp(tx, *poptv), label='Fit to Medians')
    plt.plot(tx, 4.5*exp(tx, *poptv), label='4.5 times Fit to Medians')
    plt.ylabel("Relative Velocity between Galaxy and Group Center")
    plt.xlabel(r"Integrated baryonic Mass of Giant + Dwarf Members")
    plt.xlim(9.9,13.2)
    plt.ylim(0,2000)
    plt.show()

    rproj_for_iteration = lambda M: 3*exp(M, *poptr)
    vproj_for_iteration = lambda M: 4.5*exp(M, *poptv)

    # --------------- now need to do this calibration for the RESOLVE-B analogue dataset, down to 9.1 baryonic mass) -------------$
    resbana_gdgrpn = fof.multiplicity_function(resbana_g3grp, return_by_galaxy=True)
    #resbana_gdsel = np.logical_not((resbana_gdgrpn==1) & (ecologmbary>-19.4) & (resbana_g3grp!=-99.) & (resbana_g3grp>0)) # select galaxies that AREN'T ungrouped dwarfs
    resbana_gdsel = np.logical_not(np.logical_or(resbana_g3grp==-99., ((resbana_gdgrpn==1) & (ecologmbary<9.9) & (ecologmbary>=9.1))))
    resbana_gdgrpra, resbana_gdgrpdec, resbana_gdgrpcz = fof.group_skycoords(ecoradeg[resbana_gdsel], ecodedeg[resbana_gdsel], ecocz[resbana_gdsel], resbana_g3grp[resbana_gdsel])
    resbana_gdrelvel = np.abs(resbana_gdgrpcz - ecocz[resbana_gdsel])
    resbana_gdrelprojdist = (resbana_gdgrpcz + ecocz[resbana_gdsel])/100. *np.sin(ic.angular_separation(resbana_gdgrpra, resbana_gdgrpdec, ecoradeg[resbana_gdsel], ecodedeg[resbana_gdsel])/2.0)

    resbana_gdn = resbana_gdgrpn[resbana_gdsel]
    resbana_gdtotalmass = ic.get_int_mass(ecologmbary[resbana_gdsel], resbana_g3grp[resbana_gdsel])

    massbins2=np.arange(9.9,14,0.15)
    binsel2 = np.where(np.logical_and(resbana_gdn>1, resbana_gdtotalmass>-24))
    gdmedianrproj, massbincenters, massbinedges, jk = center_binned_stats(resbana_gdtotalmass[binsel2], resbana_gdrelprojdist[binsel2], np.median, bins=massbins2)
    gdmedianrproj_err, jk, jk, jk = center_binned_stats(resbana_gdtotalmass[binsel2], resbana_gdrelprojdist[binsel2], sigmarange, bins=massbins2)
    gdmedianrelvel, jk, jk, jk = center_binned_stats(resbana_gdtotalmass[binsel2], resbana_gdrelvel[binsel2], np.median, bins=massbins2)
    gdmedianrelvel_err, jk, jk, jk = center_binned_stats(resbana_gdtotalmass[binsel2], resbana_gdrelvel[binsel2], sigmarange, bins=massbins2)
    nansel = np.isnan(gdmedianrproj)
    poptr_resbana, jk = curve_fit(exp, massbincenters[~nansel], gdmedianrproj[~nansel], p0=poptr)#, sigma=2**massbincenters[~nansel])
    poptv_resbana, jk = curve_fit(exp, massbincenters[~nansel], gdmedianrelvel[~nansel], p0=poptv)

    tx = np.linspace(7,15)
    plt.figure()
    plt.plot(resbana_gdtotalmass[binsel2], resbana_gdrelprojdist[binsel2], 'k.', alpha=0.2, label='Mock Galaxies in N>1 Giant+Dwarf Groups')
    plt.errorbar(massbincenters, gdmedianrproj, yerr=gdmedianrproj_err, fmt='r^', label='Medians')
    plt.plot(tx, exp(tx,*poptr_resbana), label='Fit to Medians')
    plt.plot(tx, 3*exp(tx,*poptr_resbana), label='3 times Fit to Medians')
    plt.xlabel(r"Integrated baryonic Mass of Giant + Dwarf Members")
    plt.ylabel("Projected Distance from Galaxy to Group Center [Mpc/h]")
    plt.legend(loc='best')
    plt.xlim(9.9,13.2)
    plt.ylim(0,3)
    plt.show()

    plt.figure()
    plt.plot(resbana_gdtotalmass[binsel2], resbana_gdrelvel[binsel2], 'k.', alpha=0.2, label='Mock Galaxies in N=2 Giant+Dwarf Groups')
    plt.errorbar(massbincenters, gdmedianrelvel,yerr=gdmedianrelvel_err, fmt='r^',label='Medians')
    plt.plot(tx, exp(tx, *poptv_resbana), label='Fit to Medians')
    plt.plot(tx, 4.5*exp(tx, *poptv_resbana), label='4.5 times Fit to Medians')
    plt.ylabel("Relative Velocity between Galaxy and Group Center")
    plt.xlabel(r"Integrated baryonic Mass of Giant + Dwarf Members")
    plt.xlim(9.9,13.2)
    plt.ylim(0,2000)
    plt.show()

    rproj_for_iteration_resbana = lambda M: 3*exp(M, *poptr_resbana)
    vproj_for_iteration_resbana = lambda M: 4.5*exp(M, *poptv_resbana)


    ###########################################################
    # Step 7: Iterative Combination of Dwarf Galaxies
    ###########################################################
    assert (ecog3grp[(ecologmbary>9.9) & (ecocz<8000) & (ecocz>2530)]!=-99.).all(), "Not all giants are grouped."
    ecogrpnafterassoc = fof.multiplicity_function(ecog3grp, return_by_galaxy=True)
    resbgrpnafterassoc = fof.multiplicity_function(resbg3grp, return_by_galaxy=True)
    resbana_grpnafterassoc = fof.multiplicity_function(resbana_g3grp, return_by_galaxy=True)

    eco_ungroupeddwarf_sel = (ecologmbary<9.9) & (ecologmbary>=9.4) & (ecocz<8000) & (ecocz>2530) & (ecogrpnafterassoc==1)
    ecoitassocid = ic.iterative_combination(ecoradeg[eco_ungroupeddwarf_sel], ecodedeg[eco_ungroupeddwarf_sel], ecocz[eco_ungroupeddwarf_sel], ecologmbary[eco_ungroupeddwarf_sel],\
                                           rproj_for_iteration, vproj_for_iteration, starting_id=np.max(ecog3grp)+1, centermethod='arithmetic')

    resb_ungroupeddwarf_sel = (resblogmbary<9.9) & (resblogmbary>=9.1) & (resbcz<7300) & (resbcz>4250) & (resbgrpnafterassoc==1)
    resbitassocid = ic.iterative_combination(resbradeg[resb_ungroupeddwarf_sel], resbdedeg[resb_ungroupeddwarf_sel], resbcz[resb_ungroupeddwarf_sel], resblogmbary[resb_ungroupeddwarf_sel],\
                                            rproj_for_iteration, vproj_for_iteration, starting_id=np.max(resbg3grp)+1, centermethod='arithmetic')

    resbana_ungroupeddwarf_sel = (ecologmbary<9.9) & (ecologmbary>=9.1) & (ecocz<8000) & (ecocz>2530) & (resbana_grpnafterassoc==1)
    resbana_itassocid = ic.iterative_combination(ecoradeg[resbana_ungroupeddwarf_sel], ecodedeg[resbana_ungroupeddwarf_sel], ecocz[resbana_ungroupeddwarf_sel], ecologmbary[resbana_ungroupeddwarf_sel],\
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
    resbana_haloid, resbana_halomass, jk, jk = ic.HAMwrapper(ecoradeg[resbana_hamsel], ecodedeg[resbana_hamsel], ecocz[resbana_hamsel], ecologmbary[resbana_hamsel], resbana_g3grp[resbana_hamsel],\
                                                                ecovolume, inputfilename=None, outputfilename=None)
    resbana_halomass = np.log10(10**resbana_halomass)# no longer needed as of 7/29: fof.getmhoffset(280,337,1,1,6))
    junk, uniqindex = np.unique(resbana_g3grp[resbana_hamsel], return_index=True)
    resbana_intmass = ic.get_int_mass(ecologmbary[resbana_hamsel], resbana_g3grp[resbana_hamsel])[uniqindex]
    sortind = np.argsort(resbana_intmass)
    sortedmass = resbana_intmass[sortind]
    resbcubicspline = interp1d(sortedmass, resbana_halomass[sortind], fill_value='extrapolate')

    resbintmass = ic.get_int_mass(resblogmbary[resbg3grp!=-99.], resbg3grp[resbg3grp!=-99.])
    resbg3logmh[resbg3grp!=-99.] = resbcubicspline(resbintmass)-np.log10(0.7)

    # ---- for ECO ----- #
    ecohamsel = (ecog3grp!=-99.)
    haloid, halomass, junk, junk = ic.HAMwrapper(ecoradeg[ecohamsel], ecodedeg[ecohamsel], ecocz[ecohamsel], ecologmbary[ecohamsel], ecog3grp[ecohamsel],\
                                                     ecovolume, inputfilename=None, outputfilename=None)
    junk, uniqindex = np.unique(ecog3grp[ecohamsel], return_index=True)
    halomass = halomass-np.log10(0.7)
    for i,idv in enumerate(haloid):
        sel = np.where(ecog3grp==idv)
        ecog3logmh[sel] = halomass[i] # m337b

    # calculate Rvir in arcsec
    ecog3rvir = (3*(10**ecog3logmh) / (4*np.pi*337*0.3*1.36e11) )**(1/3.)
    resbg3rvir = (3*(10**resbg3logmh) / (4*np.pi*337*0.3*1.36e11))**(1/3.)

    ecointmass = ic.get_int_mass(ecologmbary[ecohamsel], ecog3grp[ecohamsel])
    plt.figure()
    plt.plot(ecointmass, ecog3logmh[ecog3grp!=-99.], '.', color='palegreen', alpha=0.6, label='ECO', markersize=11)
    plt.plot(resbintmass, resbg3logmh[resbg3grp!=-99.], 'k.', alpha=1, label='RESOLVE-B', markersize=3)
    plt.plot
    plt.xlabel("group-integrated log baryonic mass")
    plt.ylabel(r"group halo mass (log$M_\odot$)")
    plt.legend(loc='best')
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
        gisel = np.where(np.logical_and((ecog3grp==uid),(ecologmbary>=9.9)))
        dwsel = np.where(np.logical_and((ecog3grp==uid), (ecologmbary<9.9)))
        if len(gisel[0])>0.:
            ecog3grpngi[grpsel] = len(gisel[0])
        if len(dwsel[0])>0.:
            ecog3grpndw[grpsel] = len(dwsel[0])

    ecog3grpradeg, ecog3grpdedeg, ecog3grpcz = fof.group_skycoords(ecoradeg, ecodedeg, ecocz, ecog3grp)
    ecog3rproj = fof.get_grprproj_e17(ecoradeg, ecodedeg, ecocz, ecog3grp, h=0.7) / (ecog3grpcz/70.) * 206265 # in arcsec
    ecog3fc = fof.get_central_flag(ecologmbary, ecog3grp)
    ecog3router = fof.get_outermost_galradius(ecoradeg, ecodedeg, ecocz, ecog3grp) # in arcsec
    ecog3router[(ecog3grpngi+ecog3grpndw)==1] = 0.
    junk, ecog3vdisp = fof.get_rproj_czdisp(ecoradeg, ecodedeg, ecocz, ecog3grp)
    ecog3rvir = ecog3rvir*206265/(ecog3grpcz/70.)
    ecog3grpgas = ic.get_int_mass(ecologmgas, ecog3grp)
    ecog3grpstars = ic.get_int_mass(ecologmstar, ecog3grp)
    ecog3ADtest = vz.AD_test(ecocz, ecog3grp)
    ecog3tcross = vz.group_crossing_time(ecoradeg, ecodedeg, ecocz, ecog3grp)
    ecog3colorgap = vz.group_color_gap(ecog3grp, ecologmbary, ecourcolor)
    ecog3dsprob = vz.fast_DS_test(ecoradeg,ecodedeg,ecocz,ecog3grp,niter=2500)
    ecog3nndens, ecog3edgeflag, ecog3nndens2d, ecog3edgeflag2d, ecog3edgescale2d = lss_dens_by_galaxy(ecog3grp,\
        ecoradeg, ecodedeg, ecocz, ecog3logmh, Nnn=3, rarange=(130.05,237.45), decrange=(-1,50), czrange=(2530,7470))

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
    ecog3grpgas[outofsample]=-99.
    ecog3grpstars[outofsample]=-99.
    ecog3ADtest[outofsample]=-99.
    ecog3tcross[outofsample]=-99.
    ecog3colorgap[outofsample]=-99.
    ecog3dsprob[outofsample]=-99.
    ecog3nndens[outofsample]=-99.
    ecog3edgeflag[outofsample]=-99.
    ecog3nndens2d[outofsample]=-99.
    ecog3edgeflag2d[outofsample]=-99.
    ecog3edgescale2d[outofsample]=-99.

    insample = ecog3grpn!=-99.
    ecodata['g3grp_b'] = ecog3grp
    ecodata['g3grpradeg_b'] = ecog3grpradeg
    ecodata['g3grpdedeg_b'] = ecog3grpdedeg
    ecodata['g3grpcz_b'] = ecog3grpcz
    ecodata['g3grpndw_b'] = ecog3grpndw
    ecodata['g3grpngi_b'] = ecog3grpngi
    ecodata['g3logmh_b'] = ecog3logmh
    ecodata['g3r337_b'] = ecog3rvir
    ecodata['g3rproj_b'] = ecog3rproj
    ecodata['g3router_b'] = ecog3router
    ecodata['g3fc_b'] = ecog3fc
    ecodata['g3vdisp_b'] = ecog3vdisp
    ecodata['g3grplogG_b'] = ecog3grpgas
    ecodata['g3grplogS_b'] = ecog3grpstars
    ecodata['g3grpadAlpha_b'] = ecog3ADtest
    ecodata['g3grptcross_b'] = ecog3tcross
    ecodata['g3grpcolorgap_b'] = ecog3colorgap
    ecodata['g3grpdsProb_b'] = ecog3dsprob
    ecodata['g3grpnndens_b'] = ecog3nndens
    ecodata['g3grpedgeflag_b'] = ecog3edgeflag
    ecodata['g3grpnndens2d_b'] = ecog3nndens2d
    ecodata['g3grpedgeflag2d_b'] = ecog3edgeflag2d
    ecodata['g3grpedgescale2d_b'] = ecog3edgescale2d
    ecodata.to_csv("ECOdata_G3catalog_baryonic.csv", index=False)

    # ------ now do RESOLVE
    sz = len(resolvedata)
    resolvename = np.array(resolvedata.name)
    resolveg3grp = np.full(sz, -99.)
    resolveg3grpngi = np.full(sz, -99.)
    resolveg3grpndw = np.full(sz, -99.)
    resolveg3grpradeg = np.full(sz, -99.)
    resolveg3grpdedeg = np.full(sz, -99.)
    resolveg3grpcz = np.full(sz, -99.)
    resolveg3logmh = np.full(sz, -99.)
    resolveg3rvir = np.full(sz, -99.)
    resolveg3rproj = np.full(sz,-99.)
    resolveg3fc = np.full(sz,-99.)
    resolveg3router = np.full(sz,-99.)
    resolveg3vdisp = np.full(sz,-99.)
    resolveg3grpgas = np.full(sz, -99.)
    resolveg3grpstars = np.full(sz, -99.)
    resolveg3ADtest = np.full(sz, -99.)
    resolveg3tcross = np.full(sz, -99.)
    resolveg3colorgap = np.full(sz, -99.)
    resolveg3dsprob = np.full(sz,-99.)
    resolveg3nndens = np.full(sz, -99.)
    resolveg3edgeflag = np.full(sz, -99.)
    resolveg3nndens2d = np.full(sz, -99.)
    resolveg3edgeflag2d = np.full(sz, -99.)
    resolveg3edgescale2d = np.full(sz, -99.)

    resbg3grpngi = np.full(len(resbg3grp), 0.)
    resbg3grpndw = np.full(len(resbg3grp), 0.)
    for uid in np.unique(resbg3grp):
        grpsel = np.where(resbg3grp==uid)
        gisel = np.where(np.logical_and((resbg3grp==uid),(resblogmbary>=9.9)))
        dwsel = np.where(np.logical_and((resbg3grp==uid), (resblogmbary<9.9)))
        if len(gisel[0])>0.:
            resbg3grpngi[grpsel] = len(gisel[0])
        if len(dwsel[0])>0.:
            resbg3grpndw[grpsel] = len(dwsel[0])

    resbg3grpradeg, resbg3grpdedeg, resbg3grpcz = fof.group_skycoords(resbradeg, resbdedeg, resbcz, resbg3grp)
    resbg3intmbary = ic.get_int_mass(resblogmbary, resbg3grp)
    resbg3rproj = fof.get_grprproj_e17(resbradeg, resbdedeg, resbcz, resbg3grp, h=0.7) / (resbg3grpcz/70.) * 206265 # in arcsec
    resbg3fc = fof.get_central_flag(resblogmbary, resbg3grp)
    resbg3router = fof.get_outermost_galradius(resbradeg, resbdedeg, resbcz, resbg3grp) # in arcsec
    resbg3router[(resbg3grpngi+resbg3grpndw)==1] = 0.

    junk, resbg3vdisp = fof.get_rproj_czdisp(resbradeg, resbdedeg, resbcz, resbg3grp)
    resbg3rvir = resbg3rvir*206265/(resbg3grpcz/70.)
    resbg3rvir = resbg3rvir*206265/(resbg3grpcz/70.)
    resbg3grpgas = ic.get_int_mass(resblogmgas, resbg3grp)
    resbg3grpstars = ic.get_int_mass(resblogmstar, resbg3grp)
    resbg3ADtest = vz.AD_test(resbcz, resbg3grp)
    resbg3tcross = vz.group_crossing_time(resbradeg, resbdedeg, resbcz, resbg3grp)
    resbg3colorgap = vz.group_color_gap(resbg3grp, resblogmstar, resburcolor)
    resbg3dsprob = vz.fast_DS_test(resbradeg,resbdedeg,resbcz,resbg3grp,niter=2500)
    RESB_RADEG_REMAPPED = np.copy(resbradeg)
    REMAPSEL = np.where(resbradeg>18*15.)
    RESB_RADEG_REMAPPED[REMAPSEL] = resbradeg[REMAPSEL]-360.
    resbg3nndens, resbg3edgeflag, resbg3nndens2d, resbg3edgeflag2d, resbg3edgescale2d  = lss_dens_by_galaxy(resbg3grp,\
        RESB_RADEG_REMAPPED, resbdedeg, resbcz, resbg3logmh, Nnn=3, rarange=(-2*15.,3*15.), decrange=(-1.25,1.25),\
         czrange=(4250,7250)) # must use remapped RESOLVE-B RA because of 0/360 wraparound

    outofsample = (resbg3grp==-99.)
    resbg3grpngi[outofsample]=-99.
    resbg3grpndw[outofsample]=-99.
    resbg3grpradeg[outofsample]=-99.
    resbg3grpdedeg[outofsample]=-99.
    resbg3grpcz[outofsample]=-99.
    resbg3logmh[outofsample]=-99.
    resbg3rvir[outofsample]=-99.
    resbg3rproj[outofsample]=-99.
    resbg3router[outofsample]=-99.
    resbg3fc[outofsample]=-99.
    resbg3vdisp[outofsample]=-99.
    resbg3grpgas[outofsample]=-99.
    resbg3grpstars[outofsample]=-99.
    resbg3ADtest[outofsample]=-99.
    resbg3tcross[outofsample]=-99.
    resbg3colorgap[outofsample]=-99.
    resbg3dsprob[outofsample]=-99.
    resbg3nndens[outofsample]=-99.
    resbg3edgeflag[outofsample]=-99.
    resbg3nndens2d[outofsample]=-99.
    resbg3edgeflag2d[outofsample]=-99.
    resbg3edgescale2d[outofsample]=-99.
    for i,nm in enumerate(resolvename):
        if nm.startswith('rs'):
            sel_in_eco = np.where(ecoresname==nm)
            resolveg3grp[i] = ecog3grp[sel_in_eco]
            resolveg3grpngi[i] = ecog3grpngi[sel_in_eco]
            resolveg3grpndw[i] = ecog3grpndw[sel_in_eco]
            resolveg3grpradeg[i] = ecog3grpradeg[sel_in_eco]
            resolveg3grpdedeg[i] = ecog3grpdedeg[sel_in_eco]
            resolveg3grpcz[i] = ecog3grpcz[sel_in_eco]
            resolveg3logmh[i] = ecog3logmh[sel_in_eco]
            resolveg3rvir[i] = ecog3rvir[sel_in_eco]
            resolveg3rproj[i] = ecog3rproj[sel_in_eco]
            resolveg3fc[i] = ecog3fc[sel_in_eco]
            resolveg3router[i]=ecog3router[sel_in_eco]
            resolveg3vdisp[i]=ecog3vdisp[sel_in_eco]
            resolveg3grpstars[i] = ecog3grpstars[sel_in_eco]
            resolveg3grpgas[i] = ecog3grpgas[sel_in_eco]
            resolveg3ADtest[i] = ecog3ADtest[sel_in_eco]
            resolveg3tcross[i] = ecog3tcross[sel_in_eco]
            resolveg3colorgap[i] = ecog3colorgap[sel_in_eco]
            resolveg3dsprob[i] = ecog3dsprob[sel_in_eco]
            resolveg3nndens[i] = ecog3nndens[sel_in_eco]
            resolveg3edgeflag[i] = ecog3edgeflag[sel_in_eco]
            resolveg3nndens2d[i] = ecog3nndens2d[sel_in_eco]
            resolveg3edgeflag2d[i] = ecog3edgeflag2d[sel_in_eco]
            resolveg3edgescale2d[i] = ecog3edgescale2d[sel_in_eco]
        elif nm.startswith('rf'):
            sel_in_resb = np.where(resbname==nm)
            resolveg3grp[i] = resbg3grp[sel_in_resb]
            resolveg3grpngi[i] = resbg3grpngi[sel_in_resb]
            resolveg3grpndw[i] = resbg3grpndw[sel_in_resb]
            resolveg3grpradeg[i] = resbg3grpradeg[sel_in_resb]
            resolveg3grpdedeg[i] = resbg3grpdedeg[sel_in_resb]
            resolveg3grpcz[i] = resbg3grpcz[sel_in_resb]
            resolveg3logmh[i] = resbg3logmh[sel_in_resb]
            resolveg3rvir[i] = resbg3rvir[sel_in_resb]
            resolveg3rproj[i] = resbg3rproj[sel_in_resb]
            resolveg3fc[i] = resbg3fc[sel_in_resb]
            resolveg3router[i] = resbg3router[sel_in_resb]
            resolveg3vdisp[i] = resbg3vdisp[sel_in_resb]
            resolveg3grpgas[i] = resbg3grpgas[sel_in_resb]
            resolveg3grpstars[i] = resbg3grpstars[sel_in_resb]
            resolveg3ADtest[i] = resbg3ADtest[sel_in_resb]
            resolveg3tcross[i] = resbg3tcross[sel_in_resb]
            resolveg3colorgap[i] = resbg3colorgap[sel_in_resb]
            resolveg3nndens[i] = resbg3nndens[sel_in_resb]
            resolveg3edgeflag[i] = resbg3edgeflag[sel_in_resb]
            resolveg3nndens2d[i] = resbg3nndens2d[sel_in_resb]
            resolveg3edgeflag2d[i] = resbg3edgeflag2d[sel_in_resb]
            resolveg3edgescale2d[i] = resbg3edgescale2d[sel_in_resb]
        else:
            assert False, nm+" not in RESOLVE"

    resolvedata['g3grp_b'] = resolveg3grp
    resolvedata['g3grpngi_b'] = resolveg3grpngi
    resolvedata['g3grpndw_b'] = resolveg3grpndw
    resolvedata['g3grpradeg_b'] = resolveg3grpradeg
    resolvedata['g3grpdedeg_b'] = resolveg3grpdedeg
    resolvedata['g3grpcz_b'] = resolveg3grpcz
    resolvedata['g3logmh_b'] = resolveg3logmh
    resolvedata['g3r337_b'] = resolveg3rvir
    resolvedata['g3rproj_b'] = resolveg3rproj
    resolvedata['g3router_b'] = resolveg3router
    resolvedata['g3fc_b'] = resolveg3fc
    resolvedata['g3vdisp_b'] = resolveg3vdisp
    resolvedata['g3grplogG_b'] = resolveg3grpgas
    resolvedata['g3grplogS_b'] = resolveg3grpstars
    resolvedata['g3grpadAlpha_b'] = resolveg3ADtest
    resolvedata['g3grptcross_b'] = resolveg3tcross
    resolvedata['g3grpcolorgap_b'] = resolveg3colorgap
    resolvedata['g3grpnndens_b'] = resolveg3nndens
    resolvedata['g3grpedgeflag_b'] = resolveg3edgeflag
    resolvedata['g3grpnndens2d_b'] = resolveg3nndens2d
    resolvedata['g3grpedgeflag2d_b'] = resolveg3edgeflag2d
    resolvedata['g3grpedgescale2d_b'] = resolveg3edgescale2d
    resolvedata.to_csv("RESOLVEdata_G3catalog_baryonic.csv", index=False)
