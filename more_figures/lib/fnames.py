

def load_fnames_response(obj,model_pars=''):
        
    # coupling parameters
    sim_pars = '_TN='+str(obj.TN)

    obj.lc_fname = obj.dir1+'/lc_TN={}_'+model_pars+'.txt'
    obj.m_fname = obj.dir1+'/m_TN={}_'+model_pars+'.txt'

    obj.lc_fname = obj.lc_fname.format(obj.TN)
    obj.m_fname = obj.m_fname.format(obj.TN)

    obj.A_fname = '{}A.d'.format(obj.dir1)
    
    for key in obj.var_names:
        v = '{}g{}_sym_{}.d'
        obj.g['sym_fnames_'+key] = [v.format(obj.dir1,key,k)
                                    for k in range(obj.miter)]

        v = '{}z{}_sym_{}.d'
        obj.z['sym_fnames_'+key] = [v.format(obj.dir1,key,k)
                                    for k in range(obj.miter)]
        
        v = '{}i{}_sym_{}.d'
        obj.i['sym_fnames_'+key] = [v.format(obj.dir1,key,k)
                                    for k in range(obj.miter)]

    v = '{}g_data_{}{}{}.txt'
    obj.g['dat_fnames'] = [v.format(obj.dir1,k,model_pars,sim_pars)
                           for k in range(obj.miter)]

    v = '{}z_data_{}{}{}.txt'
    obj.z['dat_fnames'] = [v.format(obj.dir1,k,model_pars,sim_pars)
                           for k in range(obj.miter)]
    
    v = '{}i_data_{}{}{}.txt'
    obj.i['dat_fnames'] = [v.format(obj.dir1,k,model_pars,sim_pars)
                           for k in range(obj.miter)]

        
    for key in obj.var_names:
        obj.g[key+'_eps_fname'] = (obj.dir1+'g_'+key+'_miter='
                                      +str(obj.miter)+'.d')

        obj.z[key+'_eps_fname'] = (obj.dir1+'z_'+key+'_miter='
                                      +str(obj.miter)+'.d')

        obj.i[key+'_eps_fname'] = (obj.dir1+'i_'+key+'_miter='
                                      +str(obj.miter)+'.d')


def load_fnames_nm(system,obj,model_pars='',coupling_pars=''):
    """
    system.dir1 is unique to the model name, so a model index in the filename is not needed.
    """
    
    pars = (system.dir1,system.miter,obj.forcing)
    system.G['fname'] = '{}G_{}_f={}.d'.format(*pars)
    system.K['fname'] = '{}K_{}_f={}.d'.format(*pars)
    
    system.G['fname_psi'] = '{}G_psi_{}_f={}.d'.format(*pars)
    system.K['fname_psi'] = '{}K_psi_{}_f={}.d'.format(*pars)
    
    system.p['fname'] = '{}p_{}_f={}.d'.format(*pars)
    system.h['fname'] = '{}h_{}_f={}.d'.format(*pars)

    system.G['fname_gz'] = '{}gz_{}_f={}.d'.format(*pars)
    system.G['fname_gi'] = '{}gi_{}_f={}.d'.format(*pars)

    fname_pars = (obj.NP,obj.NH,obj.pfactor,obj._n[1],obj._m[1],
                  obj.forcing)
    fname_pars2 = (obj.NP,obj.NH,obj.pfactor,obj._n[1],obj._m[1],
                  obj.forcing)
    

    val = '{}h_data_idx={}_NH={}.txt'
    system.h['fnames_data'] = [val.format(system.dir1,k,obj.NH) for k in range(8)]
    
    val = '{}p_sym.d'
    system.p['fname_sym'] = val.format(system.dir1)


    nn = system.miter
    fname_pars3 = (system.dir1,obj.NH,obj._n[1],obj._m[1])
    

    val = '{}p_data_hom_NH={}_n={}_m={}_ord={}.txt'
    system.p['fnames_data_hom'] = [val.format(*fname_pars3,k) for k in range(nn)]

    val = '{}p_data_het_NH={}_n={}_m={}_ord={}_pow={}.txt'
    system.p['fnames_data_het'] = [[val.format(*fname_pars3,k,j+1) for j in range(k)] for k in range(nn)]
    system.p['fnames_data_het'][0].append(val.format(*fname_pars3,0,0))

    #val = '{}p_sym_het_NH={}_n={}_m={}_ord={}_pow={}.txt'
    #system.p['fnames_sym_het'] = [val.format(*fname_pars3,k,j+1) for j in range(10)]

    val = '{}h_data_hom_NH={}_n={}_m={}_ord={}.txt'
    system.h['fnames_data_hom'] = [val.format(*fname_pars3,k) for k in range(nn)]    
    
    val = '{}h_data_het_NH={}_n={}_m={}_ord={}_pow={}.txt'
    system.h['fnames_data_het'] = [[val.format(*fname_pars3,k,j) for j in range(k+1)] for k in range(nn)]

    
    val = '{}p_data_homb_NH={}_n={}_m={}_ord={}.txt'
    system.p['fnames_data_homb'] = [val.format(*fname_pars3,k) for k in range(nn)]

    val = '{}p_data_hetb_NH={}_n={}_m={}_ord={}_pow={}.txt'
    system.p['fnames_data_hetb'] = [[val.format(*fname_pars3,k,j+1) for j in range(k)] for k in range(nn)]
    system.p['fnames_data_hetb'][0].append(val.format(*fname_pars3,0,0))

    val = '{}h_data_homb_NH={}_n={}_m={}_ord={}.txt'
    system.h['fnames_data_homb'] = [val.format(*fname_pars3,k) for k in range(nn)]    
    
    val = '{}h_data_hetb_NH={}_n={}_m={}_ord={}_pow={}.txt'
    system.h['fnames_data_hetb'] = [[val.format(*fname_pars3,k,j) for j in range(k+1)] for k in range(nn)]
    #system.h['fnames_data_het'][0].append(val.format(*fname_pars3,0,0))
