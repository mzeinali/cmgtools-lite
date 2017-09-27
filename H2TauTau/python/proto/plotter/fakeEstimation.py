import copy

from CMGTools.H2TauTau.proto.plotter.Variables import dict_all_vars
from CMGTools.H2TauTau.proto.plotter.HistCreator import createHistogram
from CMGTools.H2TauTau.proto.plotter.PlotConfigs import HistogramCfg

def fake_estimation(N_0t, N_1t, N_2t, all_samples, int_lumi, total_weight, scale=1., verbose=True, friend_func=None):
    '''Fake method.
    
    Returns an updated list of samples that includes the QCD HistgramCfg.
    '''
    
    norm_var = dict_all_vars['_norm_']

    bothLooseNonTight_cut = N_0t
    oneLooseNonTight_oneTight_cut = N_1t
    bothTight_cut = N_2t
        
    samples_qcd_copy = copy.deepcopy( [s for s in all_samples if s.name != 'QCD' and not s.is_signal and s.name == 'data_obs'] )
    samples_qcd_copy = [s for s in samples_qcd_copy if not s.is_signal and s.name == 'data_obs']
    
    for sample in samples_qcd_copy:
        sample.scale = scale if sample.name == 'data_obs' else -scale
    
    ''' 
    qcd_c_region = HistogramCfg(name='QCD_C_region', var=norm_var, cfgs=samples_qcd_copy, cut=str(QCD_C_region_cut), lumi=int_lumi, weight=total_weight)
    qcd_d_region = HistogramCfg(name='QCD_D_region', var=norm_var, cfgs=samples_qcd_copy, cut=str(QCD_D_region_cut), lumi=int_lumi, weight=total_weight)

    plot_qcd_c = createHistogram(qcd_c_region, all_stack=True, friend_func=friend_func)
    plot_qcd_d = createHistogram(qcd_d_region, all_stack=True, friend_func=friend_func)

    if verbose:
        print 'Histogram C region'
        print plot_qcd_c
        print 'Histogram D region'
        print plot_qcd_d

    yield_c = plot_qcd_c.GetStack().totalHist.Yield()
    yield_d = plot_qcd_d.GetStack().totalHist.Yield()

    if yield_d == 0.:
        print 'WARNING: no events left for the QCD estimation. Set to 0'
        qcd_scale = 0.
    else:
        qcd_scale = yield_c / yield_d

    if qcd_scale < 0.:
        print 'WARNING: negative QCD scaling; set it to zero'
        qcd_scale = 0.
        verbose = True

    if verbose:
        print 'QCD estimation: '
        print '  Yield C:', yield_c, ' yield D:', yield_d
        print '  Ratio C/D', qcd_scale
    '''

    p = 0.710724 # 0.639389 -> this number obtained from moriond dataset (12.9 fb-1) and also from VLoose/VTight isolations
	# 0.674774 +- 0.00686021 obtained from DY_ext sample, no trigger and cleaning cuts applied
	# 0.698851 +- 0.0116294 obtained from DY_ext sample, when trigger and cleaning cuts applied
	# 0.686516 +- 0.00495757 obtained from DY_ext2 sample, no trigger and cleaning cuts applied
	# 0.710724 +- 0.00830105 obtained from DY_ext2 sample, when trigger and cleaning cuts applied
    #### these fake rates are obtained based on WNJets with wrong trigger weight. Those obtained from reproduced samples are given below
    '''
    f = 0.287557 # this is obtained by Loose/VTight instead of VLoose/VTight # 0.13478 # 0.161887 	
	# FR: 0.252357 +- 0.012095 obtained from WNJets samples, no trigger and cleaning cuts applied
	# FR: 0.287036 +- 0.0218686 obtained from WNJets samples, when trigger but no cleaning cuts applied
	# FR: 0.287557 +- 0.0221594 obtained from WNJets samples, when trigger and cleaning cuts applied
    '''
    #f = 0.286389 # 0.286389 +- 0.0220926, obtained from MC WJets, here to do the MC closure test !!!!
    f = 0.276816 # this is obtained by Loose/VTight instead of VLoose/VTight # 0.13478 # 0.161887 	
	# FR: ??? obtained from WNJets samples, no trigger and cleaning cuts applied
	# FR: ??? obtained from WNJets samples, when trigger but no cleaning cuts applied
	# FR: 0.28725 +- 0.0221622 obtained from WNJets samples, when trigger and cleaning cuts applied
	# FR: 0.276816 +- 0.00069182, this is obtained from data (QCD enriched control region, SS, pfmet<30)
    #n0t_scale = f*f*p*p/(p-f)**2 # double fake
    n0t_scale = -1.*f*f*p*p/(p-f)**2 # single + double fake
    #n1t_scale = -1.*f*f*p*(1-p)/(p-f)**2 # double fake
    n1t_scale = p*p*f*(1-f)/(p-f)**2 # single + double fake
    #n2t_scale = f*f*(1-p)*(1-p)/(p-f)**2 # double fake
    n2t_scale = (f*f*(1-p)*(1-p)-2.*f*p*(1-p)*(1-f))/(p-f)**2 # single + double fake

    bothLooseNonTight_hist = HistogramCfg(name='N_0t_region', var=None, cfgs=samples_qcd_copy, cut=str(bothLooseNonTight_cut), lumi=int_lumi, weight=total_weight, total_scale=n0t_scale)
    oneLooseNonTight_oneTight_hist = HistogramCfg(name='N_1t_region', var=None, cfgs=samples_qcd_copy, cut=str(oneLooseNonTight_oneTight_cut), lumi=int_lumi, weight=total_weight, total_scale=n1t_scale)
    bothTight_hist = HistogramCfg(name='N_2t_region', var=None, cfgs=samples_qcd_copy, cut=str(bothTight_cut), lumi=int_lumi, weight=total_weight, total_scale=n2t_scale)
    
    all_samples_qcd = copy.deepcopy(all_samples)
    all_samples_qcd = [bothLooseNonTight_hist] + [oneLooseNonTight_oneTight_hist] + [bothTight_hist] + all_samples_qcd
    
    return all_samples_qcd

