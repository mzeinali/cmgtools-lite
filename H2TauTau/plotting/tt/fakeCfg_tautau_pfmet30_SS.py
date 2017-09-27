from collections import namedtuple
from operator import itemgetter

from CMGTools.H2TauTau.proto.samples.summer16.htt_common import lumi
from CMGTools.H2TauTau.proto.plotter.PlotConfigs import SampleCfg, HistogramCfg, VariableCfg
from CMGTools.H2TauTau.proto.plotter.categories_TauTau import inc_sig_tau1_iso, inc_sig_tau2_iso, inc_sig_no_iso, inc_trigger
from CMGTools.H2TauTau.proto.plotter.categories_common import cat_J1, cat_VBF
from CMGTools.H2TauTau.proto.plotter.HistCreator import createHistograms, createTrees
from CMGTools.H2TauTau.proto.plotter.HistDrawer import HistDrawer
from CMGTools.H2TauTau.proto.plotter.Variables import tautau_vars, getVars
from CMGTools.H2TauTau.proto.plotter.Samples import createSampleLists
from CMGTools.H2TauTau.proto.plotter.fakeEstimation import fake_estimation
from CMGTools.H2TauTau.proto.plotter.cut import Cut
from CMGTools.H2TauTau.proto.plotter.metrics import ams_hists

int_lumi = lumi
analysis_dir = '/afs/cern.ch/work/m/mzeinali/SUSY_diTau_fullData/CMSSW_8_0_25/src/CMGTools/H2TauTau/plotting/tt/200317/DiTauMC'
#analysis_dir = '/tmp/mzeinali/tes_down'
verbose = True
#total_weight = 'weight'
optimisation = False
make_plots = True
mode = 'susy'

import os
from ROOT import gSystem, gROOT
if "/sHTTEfficiencies_cc.so" not in gSystem.GetLibraries(): 
    gROOT.ProcessLine(".L %s/src/CMGTools/H2TauTau/python/proto/plotter/HTTEfficiencies.cc+" % os.environ['CMSSW_BASE']);
    from ROOT import getTauWeight

total_weight = 'weight*getTauWeight(l1_gen_match, l1_pt, l1_eta, l1_decayMode)*getTauWeight(l2_gen_match, l2_pt, l2_eta, l2_decayMode)'

# Infer whether this is mssm
mssm = True
if mode != 'mssm':
    mssm = False

# Check whether friend trees need to be added
friend_func = None
if mode == 'mva':
    # friend_func = lambda f: f.replace('MC', 'MCMVAmt200')
    friend_func = lambda f: f.replace('MC', 'MCMVAmt200_7Vars')



samples_mc, samples_data, samples, all_samples, sampleDict = createSampleLists(analysis_dir=analysis_dir, channel='tt', mode='mssm' if mssm else 'susy', ztt_cut='(l2_gen_match == 5 && l1_gen_match == 5)', zl_cut='(l1_gen_match < 6 && l2_gen_match < 6 && !(l1_gen_match == 5 && l2_gen_match == 5))',
                                                                               zj_cut='(l2_gen_match == 6 || l1_gen_match == 6)', signal_scale=1.)

myCut = namedtuple('myCut', ['name', 'cut'])
cuts = []

inc_sig_no_iso = inc_sig_no_iso & Cut('Flag_HBHENoiseFilter && Flag_HBHENoiseIsoFilter && Flag_EcalDeadCellTriggerPrimitiveFilter && Flag_goodVertices && Flag_eeBadScFilter && Flag_globalTightHalo2016Filter && passBadMuonFilter && passBadChargedHadronFilter && badMuonMoriond2017 && badCloneMuonMoriond2017')

# categories, do not include charge and iso cuts
ditau_pt_cut = Cut('dil_pt > 100')
met_cut = Cut('pfmet_pt > 30')
ZVeto_cut = Cut('mvis > 85 || mvis < 55')
mt2_cut = Cut('mt2 > 20')
tau1Pt_cut = Cut('l1_pt > 100')
tau1Pt_Uppercut = Cut('l1_pt < 100')

lowmet_cut = Cut('pfmet_pt < 30')

baseline_cut = Cut('pfmet_pt > 30 && (mvis > 85 || mvis < 55) && mt2 > 20')
baseline_cut_modified = Cut('pfmet_pt > 30 && mvis > 100 && mt2 > 20')

nbVeto_cut = Cut('n_bjets == 0')
nbjets_cut = Cut('n_bjets > 1')
mvis_cut = Cut('mvis > 100')

summt_cut = Cut('(pfmet_mt1+pfmet_mt2) > 300')
#summt_cut = Cut('(pfmet_mt1+pfmet_mt2) < 250 && (pfmet_mt1+pfmet_mt2) > 200')

SR1_cut = Cut('mt2 > 90')
#SR2_cut = Cut('mt2 < 90 && (pfmet_mt1+pfmet_mt2) > 250 && n_bjets == 0')
SR2_cut = Cut('mt2 < 90 && (pfmet_mt1+pfmet_mt2) > 300 && n_bjets == 0')
#SR2_cut = Cut('mt2 < 90 && (pfmet_mt1+pfmet_mt2) > 250 && n_bjets == 0 && (abs(TVector2::Phi_mpi_pi(l1_phi - l2_phi)) > 2.)')
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

mt2upper_cut = Cut('mt2 < 90')
mt2lower_cut = Cut('mt2 > 90')

eq0_jet = Cut('n_jets == 0')
gt0_jet = Cut('n_jets > 0')

atleast1loose_iso_cut = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 1.5 || l2_byIsolationMVArun2v1DBoldDMwLT > 1.5')
atleast1tight_iso_cut = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 4.5 || l2_byIsolationMVArun2v1DBoldDMwLT > 4.5')

# append categories to plot
#inc_cut = inc_sig_no_iso & charge_cut & baseline_cut & mt2upper_cut & gt0_jet & summt_cut & tau1Pt_cut # & met_cut & ZVeto_cut & ~mt2_cut
#inc_cut = inc_sig_no_iso & charge_cut & baseline_cut & mt2lower_cut & gt0_jet & tau1Pt_cut # & tau1Pt_Uppercut # & met_cut & ZVeto_cut & ~mt2_cut
inc_cut = inc_sig_no_iso & ~charge_cut & atleast1loose_iso_cut & lowmet_cut # check contamination of non-qcd in qcd control region

#inc_cut = inc_sig_no_iso & charge_cut # & gt0_jet # check the DY shape and show it's different in njets bins
#inc_cut = inc_sig_no_iso & charge_cut & baseline_cut_modified & mt2upper_cut
#inc_cut = inc_sig_no_iso & charge_cut & baseline_cut_modified & mt2lower_cut & eq0_jet # SR I
#inc_cut = inc_sig_no_iso & charge_cut & baseline_cut_modified & mt2lower_cut & gt0_jet # SR II
#inc_cut = inc_sig_no_iso & charge_cut & baseline_cut_modified & mt2upper_cut & summt_cut & eq0_jet # SR III
#inc_cut = inc_sig_no_iso & charge_cut & baseline_cut_modified & mt2upper_cut & summt_cut & gt0_jet # SR IV
#inc_cut = inc_sig_no_iso & charge_cut & baseline_cut_modified & mt2upper_cut & summt_cut & gt0_jet & nbVeto_cut # SR IV, check the comment by hannsjorg


#cuts.append(myCut('atLeast1loose_OS_pfmet30', inc_cut ))
#cuts.append(myCut('atLeast1tight_OS_pfmet30', inc_cut ))
#cuts.append(myCut('preSelection_pfmet30_zveto', inc_cut ))

#cuts.append(myCut('sumMT_mt2LT90_eq0Jet', inc_cut ))
#cuts.append(myCut('sumMT_mt2LT90_eq0Jet_summtGT300', inc_cut ))
#cuts.append(myCut('sumMT_mt2LT90_gt0Jet', inc_cut ))
#cuts.append(myCut('sumMT_mt2LT90_gt0Jet_summtGT300_tau1Pt100', inc_cut ))
#cuts.append(myCut('mt2_mt2GT90_eq0Jet', inc_cut ))
#cuts.append(myCut('mt2_mt2GT90_gt0Jet_overflow', inc_cut ))
#cuts.append(myCut('SRIV', inc_cut ))
#cuts.append(myCut('mvis_njetsGt0', inc_cut ))

#inc_cut = inc_sig_no_iso & charge_cut & nbjets_cut & mvis_cut 
cuts.append(myCut('qcdCR_SS_pfmet30_1Loose', inc_cut ))

variables = getVars(['pfmet_pt'])

#inc_cut = inc_sig_no_iso & charge_cut & ditau_pt_cut
#cuts.append(myCut('DY_controlRegion', inc_cut ))
#
#variables = getVars(['l1_byIsolationMVArun2v1DBoldDMwLT','l2_byIsolationMVArun2v1DBoldDMwLT'])

#variables = getVars(['n_bjets'])

#variables = getVars(['mt2'])
#variables = getVars(['pfmet_sumMT'])

#variables = getVars(['mt2'])
#variables = getVars(['pfmet_sumMT'])

#variables = getVars(['l1_pt','l2_pt','pfmet_pt','mvis','mt2','pfmet_sumMT','n_jets'])
#variables = getVars(['pfmet_sumMT','n_bjets','n_jets'])
#variables = getVars(['pfmet_sumMT','min_delta_phi_tau1tau2_met','min_delta_phi_j1j2_met','minDphiMETJets','delta_phi_l1l2_met','delta_phi_l1_l2'])

#cuts.append(myCut('preSelection', inc_cut ))
#variables = getVars(['l1_gen_match','l2_gen_match','l1_byIsolationMVArun2v1DBoldDMwLT','l2_byIsolationMVArun2v1DBoldDMwLT'])
#variables = getVars(['l1_byIsolationMVArun2v1DBoldDMwLT','l2_byIsolationMVArun2v1DBoldDMwLT'])
#variables = getVars(['pfmet_pt','mvis','mt2','pfmet_sumMT','l1_pt','l2_pt','n_vertices'])
#variables = getVars(['l1_pt','l2_pt'])

ams_dict = {}
sample_names = set()

for cut in cuts:
    isSS = 'SS' in cut.name
    all_samples_qcd = fake_estimation(
        cut.cut & both_loose_nontight,  # N_0t
        cut.cut & one_loose_nontight_one_tight,  # N_1t
        cut.cut & both_tight,  # N_2t
        all_samples if mssm else samples,
        int_lumi,
        total_weight,
        verbose=verbose,
        friend_func=friend_func
    )

    # now include charge and isolation too
    #cut = myCut(cut.name, cut.cut & iso_cut & (charge_cut if not isSS else ~charge_cut)) # default
#    cut = myCut(cut.name, cut.cut & iso_cut)  # comment this on 15 Aug to update the QCD CR plot
    #cut = myCut(cut.name, cut.cut & atleast1loose_iso_cut & charge_cut)
    #cut = myCut(cut.name, cut.cut & atleast1tight_iso_cut & charge_cut)

    # for variable in variables:
    print '********************** ', cut.cut
    cfg_total = HistogramCfg(name=cut.name, vars=variables, cfgs=all_samples_qcd, cut=str(cut.cut), lumi=int_lumi, weight=total_weight)
    # all_samples_qcd[-1].vars = variables

    plots = createHistograms(cfg_total, verbose=True, friend_func=friend_func)

    for variable in variables:
        plot = plots[variable.name]
        plot.Group('TT', ['TT', 'T_tWch', 'TBar_tWch', 'TToLeptons_tch_powheg', 'TBarToLeptons_tch_powheg'])  # 'TToLeptons_sch',
        # plot.Group('Single T', ['T_tWch', 'TBar_tWch', 'TToLeptons_tch_powheg', 'TBarToLeptons_tch_powheg'])  # 'TToLeptons_sch',
        plot.Group('VV', ['VVTo2L2Nu', 'ZZTo2L2Q', 'WWTo1L1Nu2Q', 'WZTo1L3Nu', 'ZZTo4L',  'WZTo2L2Q', 'WZTo1L1Nu2Q'])  # 'WZTo3L',
        plot.Group('ZTT', ['ZTT', 'ZTT1Jets', 'ZTT2Jets', 'ZTT3Jets', 'ZTT4Jets'])
        plot.Group('ZJ', ['ZJ', 'ZJ1Jets', 'ZJ2Jets', 'ZJ3Jets', 'ZJ4Jets'])
        plot.Group('ZL', ['ZL', 'ZL1Jets', 'ZL2Jets', 'ZL3Jets', 'ZL4Jets'])
	plot.Group('DYJets', ['ZTT','ZJ','ZL']) # when splitDY set to False
	#plot.Group('DYJets', ['ZTT', 'ZTT1Jets', 'ZTT2Jets', 'ZTT3Jets', 'ZTT4Jets'])
	plot.Group('FakeEstimation', ['N_0t_region','N_1t_region','N_2t_region'])
        # plot.Group('W', ['WJetsToLNu', 'W1Jets', 'W2Jets', 'W3Jets', 'W4Jets']) ## commented on 11 Feb
        plot.Group('W', ['W1Jets', 'W2Jets', 'W3Jets', 'W4Jets']) ## commented on 11 Feb
        # plot.Group('Electroweak', ['W', 'VV', 'Single t', 'ZJ'])
        plot.Group('SMS_400_1', ['SMS_TChipmStauSnuMStau400MChi1'])
        plot.Group('SMS_250_50', ['SMS_TChipmStauSnuMStau250MChi50'])
        plot.Group('SMS_200_100', ['SMS_TChipmStauSnuMStau200MChi100'])
        # plot.Group('SMS_400_1', ['SMSDM400_MStau400MChi1','SMSDM400_MStau425MChi25','SMSDM400_MStau450MChi50','SMSDM400_MStau475MChi75'])
        # plot.Group('SMS_250_50', ['SMSDM200_MStau250MChi50','SMSDM200_MStau275MChi75','SMSDM200_MStau300MChi100','SMSDM200_MStau325MChi125'])
        # plot.Group('SMS_200_100', ['SMSDM100_MStau200MChi100','SMSDM100_MStau225MChi125','SMSDM100_MStau250MChi150','SMSDM100_MStau275MChi175'])

        if optimisation:
            plot.DrawStack('HIST')
            print plot
            for signal_hist in plot.SignalHists():
                sample_names.add(signal_hist.name)
                ams_dict[variable.name + '__' + cut.name + '__' + signal_hist.name + '_'] = ams_hists(signal_hist.weighted, plot.BGHist().weighted)
        
        if not make_plots:
            continue

        blindxmin = 0.7 if 'mva' in variable.name else None
        blindxmax = 1.00001 if 'mva' in variable.name else None

        if (variable.name == 'mt2' or variable.name == 'mt2_lep' ):
            blindxmin = 90.
            blindxmax = variable.binning['xmax']

        if variable.name == 'pfmet_sumMT':
            blindxmin = 300.
            blindxmax = variable.binning['xmax']

        HistDrawer.draw(plot, channel='#tau_{h}#tau_{h}', plot_dir='plot_%s' % cut.name, blindxmin=blindxmin, blindxmax=blindxmax)
            # blindxmin=variable.binning[
                        # 'xmin'] if optimisation and 'xmin' in variable.binning else None, blindxmax=variable.binning['xmax'] if optimisation and 'xmax' in variable.binning else None)
        # HistDrawer.drawRatio(plot, channel='#tau_{h}#tau_{h}')

        # if variable.name == 'mvis':
        #     plot.WriteDataCard(filename='plot_%s/htt_tt.inputs-sm-13TeV.root' %cut.name, dir='tt_' + cut.name, mode='UPDATE')
        if variable.name == 'svfit_mass':
            plot.WriteDataCard(filename='plot_%s/htt_tt.inputs-sm-13TeV_svFit.root' % cut.name, dir='tt_' + cut.name, mode='UPDATE')
