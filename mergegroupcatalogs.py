import pandas as pd

lumgrps = pd.read_csv("ECOdata_G3catalog_luminosity.csv")
lumgrps = lumgrps.set_index('name')

stellargrps = pd.read_csv("ECOdata_G3catalog_stellar.csv")
stellargrps = stellargrps.set_index('name')
stellargrps = stellargrps[['g3grp_s', 'g3grpngi_s', 'g3grpndw_s', 'g3grpradeg_s', 'g3grpdedeg_s', 'g3grpcz_s', 'g3logmh_s', 'g3r337_s', 'g3rproj_s', 'g3router_s', 'g3fc_s',\
                            'g3grplogG_s', 'g3grplogS_s', 'g3grpadAlpha_s', 'g3grptcross_s', 'g3grpcolorgap_s', 'g3grpdsProb_s', 'g3grpnndens_s', 'g3grpedgeflag_s', 'g3grpnndens2d_s',\
                            'g3grpedgeflag2d_s', 'g3grpedgescale2d_s']]

barygrps = pd.read_csv("ECOdata_G3catalog_baryonic.csv")
barygrps = barygrps.set_index('name')
barygrps = barygrps[['g3grp_b', 'g3grpngi_b', 'g3grpndw_b', 'g3grpradeg_b', 'g3grpdedeg_b', 'g3grpcz_b', 'g3logmh_b', 'g3r337_b', 'g3rproj_b', 'g3router_b', 'g3fc_b',\
                    'g3grplogG_b', 'g3grplogS_b', 'g3grpadAlpha_b', 'g3grptcross_b', 'g3grpcolorgap_b','g3grpdsProb_b', 'g3grpnndens_b', 'g3grpedgeflag_b', 'g3grpnndens2d_b',\
                            'g3grpedgeflag2d_b', 'g3grpedgescale2d_b']]


eco = lumgrps.join(stellargrps)
eco = eco.join(barygrps)

eco=eco[['radeg','dedeg','cz','absrmag','logmstar','dup','resname',\
         'g3grp_l', 'g3grpngi_l', 'g3grpndw_l', 'g3grpradeg_l', 'g3grpdedeg_l', 'g3grpcz_l', 'g3logmh_l', 'g3r337_l',\
         'g3rproj_l', 'g3router_l', 'g3fc_l', 'g3grplogG_l', 'g3grplogS_l', 'g3grpadAlpha_l', 'g3grptcross_l', 'g3grpcolorgap_l',\
         'g3grpdsProb_l', 'g3grpnndens_l', 'g3grpedgeflag_l', 'g3grpnndens2d_l','g3grpedgeflag2d_l', 'g3grpedgescale2d_l',\
         'g3grp_s', 'g3grpngi_s', 'g3grpndw_s', 'g3grpradeg_s', 'g3grpdedeg_s', 'g3grpcz_s', 'g3logmh_s', 'g3r337_s',\
         'g3rproj_s', 'g3router_s', 'g3fc_s', 'g3grplogG_s', 'g3grplogS_s', 'g3grpadAlpha_s', 'g3grptcross_s', 'g3grpcolorgap_s',\
         'g3grpdsProb_s', 'g3grpnndens_s', 'g3grpedgeflag_s', 'g3grpnndens2d_s','g3grpedgeflag2d_s', 'g3grpedgescale2d_s',\
         'g3grp_b', 'g3grpngi_b', 'g3grpndw_b', 'g3grpradeg_b', 'g3grpdedeg_b', 'g3grpcz_b', 'g3logmh_b', 'g3r337_b',\
         'g3rproj_b', 'g3router_b', 'g3fc_b','g3grplogG_b', 'g3grplogS_b', 'g3grpadAlpha_b', 'g3grptcross_b', 'g3grpcolorgap_b',\
         'g3grpdsProb_b', 'g3grpnndens_b', 'g3grpedgeflag_b', 'g3grpnndens2d_b','g3grpedgeflag2d_b', 'g3grpedgescale2d_b',\
         'grp', 'grpn', 'logmh', 'grpsig', 'grprproj','grpcz','fc',\
         'mhidet_a100', 'emhidet_a100', 'confused_a100', 'mhilim_a100', 'limflag_a100', 'w20_a100', 'w50_a100',\
         'ew50_a100', 'peaksnhi_a100','logmgas_a100','agcnr_a100',\
         'mhidet', 'emhidet', 'confused', 'mhilim', 'limflag', 'limsigma', 'limmult', 'w20', 'w50', 'ew50',\
         'peaksnhi', 'logmgas','hitelescope','mhi_corr','emhi_corr_rand', 'emhi_corr_sys',\
         'mhidet_e16', 'emhidet_e16', 'mhilim_e16', 'limflag_e16', 'hitelescope_e16', 'confused_e16','logmgas_e16']]
         
eco.to_csv("ECO_G3galaxycatalog_090921.csv")

eco.index.rename('central_name', inplace=True)
# output just groups (luminosity)
eco[(eco.g3fc_l==1)][['g3grp_l', 'g3grpngi_l', 'g3grpndw_l', 'g3grpradeg_l', 'g3grpdedeg_l', 'g3grpcz_l', 'g3logmh_l', 'g3r337_l', 'g3rproj_l',\
         'g3router_l','g3grplogG_l', 'g3grplogS_l', 'g3grpadAlpha_l', 'g3grptcross_l', 'g3grpcolorgap_l', 'g3grpdsProb_l', 'g3grpnndens_l',\
         'g3grpedgeflag_l', 'g3grpnndens2d_l','g3grpedgeflag2d_l', 'g3grpedgescale2d_l']].to_csv("ECO_G3groupcatalog_luminosity_090921.csv")
# output just groups (stellar mass)
eco[(eco.g3fc_s==1)][['g3grp_s', 'g3grpngi_s', 'g3grpndw_s', 'g3grpradeg_s', 'g3grpdedeg_s', 'g3grpcz_s', 'g3logmh_s', 'g3r337_s', 'g3rproj_s',\
        'g3router_s', 'g3grplogG_s', 'g3grplogS_s', 'g3grpadAlpha_s', 'g3grptcross_s', 'g3grpcolorgap_s','g3grpdsProb_s', 'g3grpnndens_s', 
         'g3grpedgeflag_s', 'g3grpnndens2d_s','g3grpedgeflag2d_s', 'g3grpedgescale2d_s']].to_csv("ECO_G3groupcatalog_stellar_090921.csv")
# output just groups (baryonic mass)
eco[(eco.g3fc_b==1)][['g3grp_b', 'g3grpngi_b', 'g3grpndw_b', 'g3grpradeg_b', 'g3grpdedeg_b', 'g3grpcz_b', 'g3logmh_b', 'g3r337_b', 'g3rproj_b',\
        'g3router_b', 'g3grplogG_b', 'g3grplogS_b', 'g3grpadAlpha_b', 'g3grptcross_b', 'g3grpcolorgap_b','g3grpdsProb_b', 'g3grpnndens_b',\
         'g3grpedgeflag_b', 'g3grpnndens2d_b','g3grpedgeflag2d_b', 'g3grpedgescale2d_b']].to_csv("ECO_G3groupcatalog_baryonic_090921.csv")



###############################################################################
# output RESOLVE
lumgrps = pd.read_csv("RESOLVEdata_G3catalog_luminosity.csv")
lumgrps = lumgrps.set_index('name')

stellargrps = pd.read_csv("RESOLVEdata_G3catalog_stellar.csv")
stellargrps = stellargrps.set_index('name')
stellargrps = stellargrps[['g3grp_s', 'g3grpngi_s', 'g3grpndw_s', 'g3grpradeg_s', 'g3grpdedeg_s', 'g3grpcz_s', 'g3logmh_s', 'g3r337_s', 'g3rproj_s', 'g3router_s', 'g3fc_s',\
                            'g3grplogG_s', 'g3grplogS_s', 'g3grpadAlpha_s', 'g3grptcross_s', 'g3grpcolorgap_s', 'g3grpdsProb_s', 'g3grpnndens_s', 'g3grpedgeflag_s', 'g3grpnndens2d_s',\
                            'g3grpedgeflag2d_s', 'g3grpedgescale2d_s']]

barygrps = pd.read_csv("RESOLVEdata_G3catalog_baryonic.csv")
barygrps = barygrps.set_index('name')
barygrps = barygrps[['g3grp_b', 'g3grpngi_b', 'g3grpndw_b', 'g3grpradeg_b', 'g3grpdedeg_b', 'g3grpcz_b', 'g3logmh_b', 'g3r337_b', 'g3rproj_b', 'g3router_b', 'g3fc_b',\
                    'g3grplogG_b', 'g3grplogS_b', 'g3grpadAlpha_b', 'g3grptcross_b', 'g3grpcolorgap_b','g3grpdsProb_b', 'g3grpnndens_b', 'g3grpedgeflag_b', 'g3grpnndens2d_b',\
                            'g3grpedgeflag2d_b', 'g3grpedgescale2d_b']]

resolve = lumgrps.join(stellargrps)
resolve = resolve.join(barygrps)

resolve=resolve[['radeg','dedeg','cz','absrmag','logmstar','f_a','f_b','fl_insample','econame',\
         'g3grp_l', 'g3grpngi_l', 'g3grpndw_l', 'g3grpradeg_l', 'g3grpdedeg_l', 'g3grpcz_l', 'g3logmh_l', 'g3r337_l',\
         'g3rproj_l', 'g3router_l', 'g3fc_l', 'g3grplogG_l', 'g3grplogS_l', 'g3grpadAlpha_l', 'g3grptcross_l', 'g3grpcolorgap_l',\
         'g3grpdsProb_l', 'g3grpnndens_l', 'g3grpedgeflag_l', 'g3grpnndens2d_l','g3grpedgeflag2d_l', 'g3grpedgescale2d_l',\
         'g3grp_s', 'g3grpngi_s', 'g3grpndw_s', 'g3grpradeg_s', 'g3grpdedeg_s', 'g3grpcz_s', 'g3logmh_s', 'g3r337_s',\
         'g3rproj_s', 'g3router_s', 'g3fc_s', 'g3grplogG_s', 'g3grplogS_s', 'g3grpadAlpha_s', 'g3grptcross_s', 'g3grpcolorgap_s',\
         'g3grpdsProb_s', 'g3grpnndens_s', 'g3grpedgeflag_s', 'g3grpnndens2d_s','g3grpedgeflag2d_s', 'g3grpedgescale2d_s',\
         'g3grp_b', 'g3grpngi_b', 'g3grpndw_b', 'g3grpradeg_b', 'g3grpdedeg_b', 'g3grpcz_b', 'g3logmh_b', 'g3r337_b',\
         'g3rproj_b', 'g3router_b', 'g3fc_b','g3grplogG_b', 'g3grplogS_b', 'g3grpadAlpha_b', 'g3grptcross_b', 'g3grpcolorgap_b',\
         'g3grpdsProb_b', 'g3grpnndens_b', 'g3grpedgeflag_b', 'g3grpnndens2d_b','g3grpedgeflag2d_b', 'g3grpedgescale2d_b',\
         'grp', 'grpn', 'logmh', 'grpsig', 'grprproj','grpcz','fc',\
         'mhidet', 'emhidet', 'confused', 'mhilim', 'limflag', 'w20', 'w50', 'ew50','mhi_corr','emhi_corr_rand','emhi_corr_sys',\
         'peaksnhi', 'logmgas','hitelescope','logmgas_e16']]


resolve.to_csv("RESOLVE_G3galaxycatalog_090921.csv")


resolve.index.rename('central_name', inplace=True)
# output just groups (luminosity)
resolve[(resolve.g3fc_l==1)][['g3grp_l', 'g3grpngi_l', 'g3grpndw_l', 'g3grpradeg_l', 'g3grpdedeg_l', 'g3grpcz_l', 'g3logmh_l', 'g3r337_l', 'g3rproj_l',\
         'g3router_l','g3grplogG_l', 'g3grplogS_l', 'g3grpadAlpha_l', 'g3grptcross_l', 'g3grpcolorgap_l', 'g3grpdsProb_l', 'g3grpnndens_l',\
         'g3grpedgeflag_l', 'g3grpnndens2d_l','g3grpedgeflag2d_l', 'g3grpedgescale2d_l']].to_csv("RESOLVE_G3groupcatalog_luminosity_090921.csv")
# output just groups (stellar mass)
resolve[(resolve.g3fc_s==1)][['g3grp_s', 'g3grpngi_s', 'g3grpndw_s', 'g3grpradeg_s', 'g3grpdedeg_s', 'g3grpcz_s', 'g3logmh_s', 'g3r337_s', 'g3rproj_s',\
        'g3router_s', 'g3grplogG_s', 'g3grplogS_s', 'g3grpadAlpha_s', 'g3grptcross_s', 'g3grpcolorgap_s','g3grpdsProb_s', 'g3grpnndens_s',\
         'g3grpedgeflag_s', 'g3grpnndens2d_s','g3grpedgeflag2d_s', 'g3grpedgescale2d_s']].to_csv("RESOLVE_G3groupcatalog_stellar_090921.csv")
# output just groups (baryonic mass)
resolve[(resolve.g3fc_b==1)][['g3grp_b', 'g3grpngi_b', 'g3grpndw_b', 'g3grpradeg_b', 'g3grpdedeg_b', 'g3grpcz_b', 'g3logmh_b', 'g3r337_b', 'g3rproj_b',\
        'g3router_b', 'g3grplogG_b', 'g3grplogS_b', 'g3grpadAlpha_b', 'g3grptcross_b', 'g3grpcolorgap_b','g3grpdsProb_b', 'g3grpnndens_b',\
         'g3grpedgeflag_b', 'g3grpnndens2d_b','g3grpedgeflag2d_b', 'g3grpedgescale2d_b']].to_csv("RESOLVE_G3groupcatalog_baryonic_090921.csv")

