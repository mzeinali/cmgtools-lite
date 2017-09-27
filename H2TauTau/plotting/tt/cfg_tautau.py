from collections import namedtuple
from operator import itemgetter

from CMGTools.H2TauTau.proto.samples.summer16.htt_common import lumi
from CMGTools.H2TauTau.proto.plotter.PlotConfigs import SampleCfg, HistogramCfg, VariableCfg
from CMGTools.H2TauTau.proto.plotter.categories_TauTau import inc_sig_tau1_iso, inc_sig_tau2_iso, inc_sig_no_iso
from CMGTools.H2TauTau.proto.plotter.categories_common import cat_J1, cat_VBF
from CMGTools.H2TauTau.proto.plotter.HistCreator import createHistograms, createTrees
from CMGTools.H2TauTau.proto.plotter.HistDrawer import HistDrawer
from CMGTools.H2TauTau.proto.plotter.Variables import tautau_vars, getVars
from CMGTools.H2TauTau.proto.plotter.Samples import createSampleLists
from CMGTools.H2TauTau.proto.plotter.qcdEstimation import qcd_estimation
from CMGTools.H2TauTau.proto.plotter.cut import Cut
from CMGTools.H2TauTau.proto.plotter.metrics import ams_hists_rebin

int_lumi = lumi
analysis_dir = '/afs/cern.ch/work/m/mzeinali/SUSY_diTau_fullData/CMSSW_8_0_25/src/CMGTools/H2TauTau/plotting/tt/200317/DiTauMC'
verbose = True
total_weight = 'weight'
optimisation = False
make_plots = True
mode = 'susy'

import os
from ROOT import gSystem, gROOT
if "/sHTTEfficiencies_cc.so" not in gSystem.GetLibraries(): 
    gROOT.ProcessLine(".L %s/src/CMGTools/H2TauTau/python/proto/plotter/HTTEfficiencies.cc+" % os.environ['CMSSW_BASE']);
    from ROOT import getTauWeight

total_weight = 'weight*getTauWeight(l1_gen_match, l1_pt, l1_eta, l1_decayMode)*getTauWeight(l2_gen_match, l2_pt, l2_eta, l2_decayMode)'

samples_mc, samples_data, samples, all_samples, sampleDict = createSampleLists(analysis_dir=analysis_dir, channel='tt', mode='mssm' if mssm else 'susy', ztt_cut='(l2_gen_match == 5 && l1_gen_match == 5)', zl_cut='(l1_gen_match < 6 && l2_gen_match < 6 && !(l1_gen_match == 5 && l2_gen_match == 5))',

MyCut = namedtuple('MyCut', ['name', 'cut'])

cuts = []

inc_sig_no_iso = inc_sig_no_iso & Cut('Flag_HBHENoiseFilter && Flag_HBHENoiseIsoFilter && Flag_EcalDeadCellTriggerPrimitiveFilter && Flag_goodVertices && Flag_eeBadScFilter && Flag_globalTightHalo2016Filter && passBadMuonFilter && passBadChargedHadronFilter && badMuonMoriond2017 && badCloneMuonMoriond2017')

# categories, do not include charge and iso cuts
met_cut = Cut('pfmet_pt > 30')
ZVeto_cut = Cut('mvis > 85 || mvis < 55')
mt2_cut = Cut('mt2 > 20')
jet_cut = Cut('n_jets > 0')

lowmet_cut = Cut('pfmet_pt < 30')

baseline_cut = Cut('pfmet_pt > 30 && (mvis > 85 || mvis < 55) && mt2 > 20')

mt2upper_cut = Cut('mt2 < 90')

nbVeto_cut = Cut('n_bjets == 0')

#summt_cut = Cut('(pfmet_mt1+pfmet_mt2) > 250')
summt_cut = Cut('(pfmet_mt1+pfmet_mt2) < 250 && (pfmet_mt1+pfmet_mt2) > 200')

SR1_cut = Cut('mt2 > 90')
SR2_cut = Cut('mt2 < 90 && (pfmet_mt1+pfmet_mt2) > 250 && n_bjets == 0')
SR3_cut = Cut('mt2 < 90 && (pfmet_mt1+pfmet_mt2) < 250 && (pfmet_mt1+pfmet_mt2) > 200 && n_bjets == 0')

#inc_cut = inc_sig_no_iso & baseline_cut & mt2upper_cut & nbVeto_cut & summt_cut
#inc_cut = inc_sig_no_iso & met_cut & ZVeto_cut & mt2upper_cut & nbVeto_cut & summt_cut

#inc_cut = inc_sig_no_iso & met_cut & ZVeto_cut & mt2_cut & SR1_cut
#inc_cut = inc_sig_no_iso & met_cut & ZVeto_cut & mt2_cut & SR2_cut
#inc_cut = inc_sig_no_iso & charge_cut

# iso and charge cuts, need to have them explicitly for the QCD estimation
iso_cut = inc_sig_tau1_iso & inc_sig_tau2_iso

charge_cut = Cut('l1_charge != l2_charge')

### 0.5 is replace with 1.5 to check the fake estimation by going from VLoose to Loose
both_loose_nontight = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 1.5 && l2_byIsolationMVArun2v1DBoldDMwLT > 1.5 && l1_byIsolationMVArun2v1DBoldDMwLT < 4.5 && l2_byIsolationMVArun2v1DBoldDMwLT < 4.5')
one_loose_nontight_one_tight = Cut('(l1_byIsolationMVArun2v1DBoldDMwLT > 1.5 && l1_byIsolationMVArun2v1DBoldDMwLT < 4.5 && l2_byIsolationMVArun2v1DBoldDMwLT > 4.5) || (l1_byIsolationMVArun2v1DBoldDMwLT > 4.5 && l2_byIsolationMVArun2v1DBoldDMwLT > 1.5 && l2_byIsolationMVArun2v1DBoldDMwLT < 4.5)')
both_tight = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 4.5 && l2_byIsolationMVArun2v1DBoldDMwLT > 4.5')

inc_cut = inc_sig_no_iso & charge_cut# & lowmet_cut

atleast1loose_iso_cut = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 0.5 || l2_byIsolationMVArun2v1DBoldDMwLT > 0.5')
atleast1tight_iso_cut = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 4.5 || l2_byIsolationMVArun2v1DBoldDMwLT > 4.5')
# append categories to plot

#cuts.append(myCut('atLeast1loose_OS_pfmet30', inc_cut ))
#cuts.append(myCut('atLeast1tight_OS_pfmet30', inc_cut ))
cuts.append(myCut('test', inc_cut ))


    # if optimisation:
    #     cuts = []
    #     met_sig_cuts = [2, 3]
    #     # met_sig_cuts = [1]
    #     sum_mt_cuts = [0, 50, 100, 150, 200, 250]
    #     # pzeta_disc_cuts = [-40, 0, 1000]
    #     pzeta_disc_cuts = [-40, 1000]

    #     for met_sig_cut in met_sig_cuts:
    #         for sum_mt_cut in sum_mt_cuts:
    #             for pzeta_cut in pzeta_disc_cuts:
    #                 cut_name = 'susy_jan_{c1}_{c2}_{c3}'.format(c1=met_sig_cut, c2=sum_mt_cut, c3=pzeta_cut)
    #                 cut = 'met_pt/sqrt(met_cov00 + met_cov11) > {met_sig_cut} && mvis>100 && mt + mt_leg2 > {sum_mt_cut} && n_bjets==0 && pzeta_disc < {pzeta_disc_cut}'.format(met_sig_cut=met_sig_cut, sum_mt_cut=sum_mt_cut, pzeta_disc_cut=pzeta_cut)
    #                 cuts.append(MyCut(cut_name, inc_cut & cut))


    # cuts.append(MyCut('susy_jan_SS', inc_cut & Cut('met_pt/sqrt(met_cov00 + met_cov11) > 1. && mvis>100 && mt + mt_leg2 > 150. && n_bjets==0 && pzeta_disc < -40.')))

    

def getVariables(mode):
    # Taken from Variables.py, can get subset with e.g. getVars(['mt', 'mvis'])
    # variables = tautau_vars
    if mode == 'control':
        variables = getVars(['_norm_', 'mvis', 'mt2', 'l1_pt', 'l2_pt', 'delta_phi_l1_l2', 'delta_eta_l1_l2', 'met_pt', 'mt_total', 'mt_total_mssm', 'mt_sum', 'pzeta_met', 'l2_mt', 'mt', 'pzeta_vis', 'pzeta_disc', 'pthiggs', 'jet1_pt', 'n_jets', 'dil_pt'], channel='tautau')
    if mode == 'mssm':
        variables = getVars(['mt_total', 'mt_total_mssm', 'mt_total_mssm_fine', 'mvis_extended', 'l1_pt'], channel='tautau')
    # variables += [
    #     VariableCfg(name='mt2', binning={'nbinsx':15, 'xmin':0., 'xmax':150.}, unit='GeV', xtitle='m_{T2}')
    # ]
    if mode == 'mva':
        variables += getVars(['_norm_'])
        variables += [
            VariableCfg(name='mva1', binning={'nbinsx':10, 'xmin':0., 'xmax':1.}, unit='', xtitle='Stau MVA')
        ]

    if mode == 'susy':
        variables = getVars(['l1_pt', '_norm_', 'l2_pt', 'mt2', 'mt', 'mt_leg2', 'mt_total_mssm'])

    return variables



def createSamples(mode, analysis_dir, optimisation=False):
    samples_mc, samples_data, samples, all_samples, sampleDict = createSampleLists(analysis_dir=analysis_dir, channel='tt', mode=mode, ztt_cut='(l2_gen_match == 5 && l1_gen_match == 5)', zl_cut='(l1_gen_match < 6 && l2_gen_match < 6 && !(l1_gen_match == 5 && l2_gen_match == 5))',
                                                                                   zj_cut='(l2_gen_match == 6 || l1_gen_match == 6)', signal_scale=1. if optimisation else 20.)
    return all_samples, samples


def makePlots(variables, cuts, total_weight, all_samples, samples, friend_func, mode='control', dc_postfix='', make_plots=True, optimisation=False):
    sample_names = set()
    ams_dict = {}

    from CMGTools.H2TauTau.proto.plotter.cut import Cut

    # def_iso_cut = inc_sig_tau1_iso & inc_sig_tau2_iso
    iso_cuts = {
        'vvtight':(Cut('l1_byIsolationMVArun2v1DBoldDMwLT>5.5') & Cut('l2_byIsolationMVArun2v1DBoldDMwLT>5.5'), Cut('l1_byIsolationMVArun2v1DBoldDMwLT>3.5') & Cut('l2_byIsolationMVArun2v1DBoldDMwLT>3.5')),
        'vtight':(Cut('l1_byIsolationMVArun2v1DBoldDMwLT>4.5') & Cut('l2_byIsolationMVArun2v1DBoldDMwLT>4.5'), Cut('l1_byIsolationMVArun2v1DBoldDMwLT>2.5') & Cut('l2_byIsolationMVArun2v1DBoldDMwLT>2.5')),
        'tight':(Cut('l1_byIsolationMVArun2v1DBoldDMwLT>3.5') & Cut('l2_byIsolationMVArun2v1DBoldDMwLT>3.5'), Cut('l1_byIsolationMVArun2v1DBoldDMwLT>3.5') & Cut('l2_byIsolationMVArun2v1DBoldDMwLT>1.5')),
        'medium':(Cut('l1_byIsolationMVArun2v1DBoldDMwLT>2.5') & Cut('l2_byIsolationMVArun2v1DBoldDMwLT>2.5'), Cut('l1_byIsolationMVArun2v1DBoldDMwLT>0.5') & Cut('l2_byIsolationMVArun2v1DBoldDMwLT>0.5')),
        'loose':(Cut('l1_byIsolationMVArun2v1DBoldDMwLT>1.5') & Cut('l2_byIsolationMVArun2v1DBoldDMwLT>1.5'), Cut('1')),
        'vloose':(Cut('l1_byIsolationMVArun2v1DBoldDMwLT>0.5') & Cut('l2_byIsolationMVArun2v1DBoldDMwLT>0.5'), Cut('1')),
    }

    for cut in cuts:
        for iso_cut_name, (iso_cut, max_iso_cut) in iso_cuts.items():
            
            # iso and charge cuts, need to have them explicitly for the QCD estimation
            # max_iso_cut = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 2.5 && l2_byIsolationMVArun2v1DBoldDMwLT > 2.5')
            iso_sideband_cut = (~iso_cut) & max_iso_cut
            charge_cut = Cut('l1_charge != l2_charge')
            isSS = 'SS' in cut.name
            all_samples_qcd = qcd_estimation(
                cut.cut & iso_sideband_cut & (charge_cut if not isSS else ~charge_cut),  # shape sideband
                cut.cut & iso_cut & (~charge_cut),  # norm sideband 1
                cut.cut & iso_sideband_cut & (~charge_cut),  # norm sideband 2
                all_samples if mode in ['mssm'] else samples,
                int_lumi,
                total_weight,
                verbose=verbose,
                friend_func=friend_func
            )

            # now include charge and isolation too
            the_cut = MyCut(cut.name+iso_cut_name, cut.cut & iso_cut & (charge_cut if not isSS else ~charge_cut))

            # for variable in variables:
            cfg_total = HistogramCfg(name=the_cut.name, vars=variables, cfgs=all_samples_qcd, cut=str(the_cut.cut), lumi=int_lumi, weight=total_weight)
            # all_samples_qcd[-1].vars = variables

            if mode == 'mva_train':
                createTrees(cfg_total, '/data1/steggema/tt/MVATrees', verbose=True)
                continue

            plots = createHistograms(cfg_total, verbose=True, friend_func=friend_func)


            for variable in variables:
                plot = plots[variable.name]
                plot.Group('Single t', ['T_tWch', 'TBar_tWch', 'TToLeptons_tch_powheg', 'TBarToLeptons_tch_powheg'])  # 'TToLeptons_sch',
                plot.Group('VV', ['VVTo2L2Nu', 'ZZTo2L2Q', 'WWTo1L1Nu2Q', 'WZTo1L3Nu', 'ZZTo4L',  'WZTo2L2Q', 'WZTo1L1Nu2Q', 'Single t'])  # 'WZTo3L',
                plot.Group('ZTT', ['ZTT', 'ZTT1Jets', 'ZTT2Jets', 'ZTT3Jets', 'ZTT4Jets'])
                plot.Group('ZJ', ['ZJ', 'ZJ1Jets', 'ZJ2Jets', 'ZJ3Jets', 'ZJ4Jets'])
                plot.Group('ZL', ['ZL', 'ZL1Jets', 'ZL2Jets', 'ZL3Jets', 'ZL4Jets'])
                plot.Group('W', ['WJetsToLNu', 'W1Jets', 'W2Jets', 'W3Jets', 'W4Jets'])
                plot.Group('Electroweak', ['W', 'VV', 'Single t', 'ZJ'])

                if optimisation:
                    plot.DrawStack('HIST')
                    print plot
                    for signal_hist in plot.SignalHists():
                        sample_names.add(signal_hist.name)
                        ams = ams_hists_rebin(signal_hist.weighted, plot.BGHist().weighted)
                        if variable.name == 'mt_total_mssm' and signal_hist.name == 'ggH1800':
                            print ams_hists_rebin(signal_hist.weighted, plot.BGHist().weighted, debug=True)
                            # import pdb; pdb.set_trace()
                        ams_dict[variable.name + '__' + the_cut.name + '__' + signal_hist.name + '_'] = ams
                
                if not make_plots:
                    continue

                blindxmin = 0.7 if 'mva' in variable.name else None
                blindxmax = 1.00001 if 'mva' in variable.name else None

                if variable.name == 'mt2':
                    blindxmin = 60.
                    blindxmax = variable.binning['xmax']

                if variable.name == 'mt_sum':
                    blindxmin = 250.
                    blindxmax = variable.binning['xmax']

                if variable.name == 'mt_total':
                    blindxmin = 200.
                    blindxmax = variable.binning['xmax']

                plot_dir = 'plot_' + the_cut.name
                HistDrawer.draw(plot, channel='#tau_{h}#tau_{h}', plot_dir=plot_dir, blindxmin=blindxmin, blindxmax=blindxmax)
                # HistDrawer.drawRatio(plot, channel='#tau_{h}#tau_{h}')

                plot.UnGroup('Electroweak')#, ['W', 'VV', 'Single t', 'ZJ'])
                plot.Group('VV', ['VV', 'Single t'])
                if variable.name in ['mt_total', 'svfit_mass', 'mt_total_mssm', 'mt_total_mssm_fine']:
                    plot.WriteDataCard(filename=plot_dir+'/htt_tt.inputs-sm-13TeV_{var}{postfix}.root'.format(var=variable.name, postfix=dc_postfix), dir='tt_' + cut.name, mode='UPDATE')

            # Save AMS dict
            import pickle
            pickle.dump(ams_dict, open('opt.pkl', 'wb'))
            

    if optimisation:
        print '\nOptimisation results:'
        all_vals = ams_dict.items()
        for sample_name in sample_names:
            vals = [v for v in all_vals if sample_name + '_' in v[0]]
            vals.sort(key=itemgetter(1))
            for key, item in vals:
                print item, key

            print '\nBy variable'
            for variable in variables:
                name = variable.name
                print '\nResults for variable', name
                for key, item in vals:
                    if key.startswith(name + '__'):
                        print item, key


if __name__ == '__main__':
    mode = 'mssm' # 'control' 'mssm' 'mva_train' 'susy' 'sm'

    all_samples, samples = createSamples(mode, analysis_dir, optimisation)
    variables = getVariables(mode)
makePlots(variables, cuts, total_weight, all_samples, samples, friend_func, mode=mode, optimisation=optimisation)
