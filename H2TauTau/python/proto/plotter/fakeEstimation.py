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

    p = 0.639389
    f = 0.13478 # 0.161887 	
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

