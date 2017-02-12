from collections import namedtuple
from operator import itemgetter

from CMGTools.H2TauTau.proto.samples.spring16.htt_common import lumi_tt as lumi
from CMGTools.H2TauTau.proto.plotter.PlotConfigs import SampleCfg, HistogramCfg, VariableCfg
from CMGTools.H2TauTau.proto.plotter.categories_TauTau import inc_sig_tau1_iso, inc_sig_tau2_iso, inc_sig_no_iso
from CMGTools.H2TauTau.proto.plotter.categories_common import cat_J1, cat_VBF
from CMGTools.H2TauTau.proto.plotter.HistCreator import createHistograms, createTrees
from CMGTools.H2TauTau.proto.plotter.HistDrawer import HistDrawer
from CMGTools.H2TauTau.proto.plotter.Variables import tautau_vars, getVars
from CMGTools.H2TauTau.proto.plotter.Samples import createSampleLists
from CMGTools.H2TauTau.proto.plotter.fakeEstimation import fake_estimation
from CMGTools.H2TauTau.proto.plotter.cut import Cut
from CMGTools.H2TauTau.proto.plotter.metrics import ams_hists

int_lumi = lumi
analysis_dir = '/afs/cern.ch/work/m/mzeinali/SUSY_diTau/CMSSW_8_0_21/src/CMGTools/H2TauTau/plotting/tt/251116/DiTauMC'
verbose = True
total_weight = 'weight'
optimisation = True
make_plots = True
mode = 'susy'

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
                                                                               zj_cut='(l2_gen_match == 6 || l1_gen_match == 6)', signal_scale=1. if optimisation else 20.)

myCut = namedtuple('myCut', ['name', 'cut'])
cuts = []

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

both_loose_nontight = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 0.5 && l2_byIsolationMVArun2v1DBoldDMwLT > 0.5 && l1_byIsolationMVArun2v1DBoldDMwLT < 4.5 && l2_byIsolationMVArun2v1DBoldDMwLT < 4.5')
one_loose_nontight_one_tight = Cut('(l1_byIsolationMVArun2v1DBoldDMwLT > 0.5 && l1_byIsolationMVArun2v1DBoldDMwLT < 4.5 && l2_byIsolationMVArun2v1DBoldDMwLT > 4.5) || (l1_byIsolationMVArun2v1DBoldDMwLT > 4.5 && l2_byIsolationMVArun2v1DBoldDMwLT > 0.5 && l2_byIsolationMVArun2v1DBoldDMwLT < 4.5)')
both_tight = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 4.5 && l2_byIsolationMVArun2v1DBoldDMwLT > 4.5')

inc_cut = inc_sig_no_iso & charge_cut

atleast1loose_iso_cut = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 0.5 || l2_byIsolationMVArun2v1DBoldDMwLT > 0.5')
atleast1tight_iso_cut = Cut('l1_byIsolationMVArun2v1DBoldDMwLT > 4.5 || l2_byIsolationMVArun2v1DBoldDMwLT > 4.5')
# append categories to plot

#cuts.append(myCut('atLeast1loose_OS_pfmet30', inc_cut ))
#cuts.append(myCut('atLeast1tight_OS_pfmet30', inc_cut ))
cuts.append(myCut('test', inc_cut ))
#variables = getVars(['pfmet_pt'])
#variables = getVars(['mvis'])
#variables = getVars(['mt2'])
#variables = getVars(['pfmet_sumMT','n_bjets','n_jets'])
#variables = getVars(['pfmet_sumMT'])

#cuts.append(myCut('preSelection', inc_cut ))
#variables = getVars(['mt2','mt2_lep','min_delta_phi_j1j2_met','minDphiMETJets'])
#variables = getVars(['l1_gen_match','l2_gen_match','l1_byIsolationMVArun2v1DBoldDMwLT','l2_byIsolationMVArun2v1DBoldDMwLT'])
#variables = getVars(['l1_byIsolationMVArun2v1DBoldDMwLT','l2_byIsolationMVArun2v1DBoldDMwLT'])
variables = getVars(['pfmet_pt'])

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
    cut = myCut(cut.name, cut.cut & iso_cut & (charge_cut if not isSS else ~charge_cut)) # default
    #cut = myCut(cut.name, cut.cut & atleast1loose_iso_cut & charge_cut)
    #cut = myCut(cut.name, cut.cut & atleast1tight_iso_cut & charge_cut)

    # for variable in variables:
    print '********************** ', cut.cut
    cfg_total = HistogramCfg(name=cut.name, vars=variables, cfgs=all_samples_qcd, cut=str(cut.cut), lumi=int_lumi, weight=total_weight)
    # all_samples_qcd[-1].vars = variables

    plots = createHistograms(cfg_total, verbose=True, friend_func=friend_func)

    for variable in variables:
        plot = plots[variable.name]
        plot.Group('Single t', ['T_tWch', 'TBar_tWch', 'TToLeptons_tch_powheg', 'TBarToLeptons_tch_powheg'])  # 'TToLeptons_sch',
        plot.Group('VV', ['VVTo2L2Nu', 'ZZTo2L2Q', 'WWTo1L1Nu2Q', 'WZTo1L3Nu', 'ZZTo4L',  'WZTo2L2Q', 'WZTo1L1Nu2Q', 'Single t'])  # 'WZTo3L',
        # plot.Group('ZTT', ['ZTT', 'ZTT1Jets', 'ZTT2Jets', 'ZTT3Jets', 'ZTT4Jets'])
        # plot.Group('ZJ', ['ZJ', 'ZJ1Jets', 'ZJ2Jets', 'ZJ3Jets', 'ZJ4Jets'])
        # plot.Group('ZL', ['ZL', 'ZL1Jets', 'ZL2Jets', 'ZL3Jets', 'ZL4Jets'])
        # plot.Group('W', ['WJetsToLNu', 'W1Jets', 'W2Jets', 'W3Jets', 'W4Jets']) ## commented on 11 Feb
	plot.Group('DYJets', ['ZTT'])
	#plot.Group('DYJets', ['ZTT','ZJ','ZL'])
	plot.Group('FakeEstimation', ['N_0t_region','N_1t_region','N_2t_region'])
        #plot.Group('Electroweak', ['W', 'VV', 'Single t', 'ZJ'])
        plot.Group('SMS_400_1', ['SMSDM400_MStau400MChi1','SMSDM400_MStau425MChi25','SMSDM400_MStau450MChi50','SMSDM400_MStau475MChi75'])
        plot.Group('SMS_250_50', ['SMSDM200_MStau250MChi50','SMSDM200_MStau275MChi75','SMSDM200_MStau300MChi100','SMSDM200_MStau325MChi125'])
        plot.Group('SMS_200_100', ['SMSDM100_MStau200MChi100','SMSDM100_MStau225MChi125','SMSDM100_MStau250MChi150','SMSDM100_MStau275MChi175'])

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
            blindxmin = 250.
            blindxmax = variable.binning['xmax']

        HistDrawer.draw(plot, channel='#tau_{h}#tau_{h}', plot_dir='plot_%s' % cut.name, blindxmin=blindxmin, blindxmax=blindxmax)
            # blindxmin=variable.binning[
                        # 'xmin'] if optimisation and 'xmin' in variable.binning else None, blindxmax=variable.binning['xmax'] if optimisation and 'xmax' in variable.binning else None)
        # HistDrawer.drawRatio(plot, channel='#tau_{h}#tau_{h}')

        # if variable.name == 'mvis':
        #     plot.WriteDataCard(filename='plot_%s/htt_tt.inputs-sm-13TeV.root' %cut.name, dir='tt_' + cut.name, mode='UPDATE')
        if variable.name == 'svfit_mass':
            plot.WriteDataCard(filename='plot_%s/htt_tt.inputs-sm-13TeV_svFit.root' % cut.name, dir='tt_' + cut.name, mode='UPDATE')
