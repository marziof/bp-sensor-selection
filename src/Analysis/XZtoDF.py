import pandas as pd
import numpy as np
import random
import argparse

from measures import *

import sys
import os
import time
import lzma
import pickle

def main():
    parser = argparse.ArgumentParser(description="Compute DataFrame of observables from data files")
    parser.add_argument(
        "--load_dir",
        type=str,
        default="../data/check/",
        help="load directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/check/",
        help="saving directory",
    )
    parser.add_argument(
        "--print_it",
        action="store_true",
        help="If false, load just the marginals at convergence. If true, load also iteration-marginals",
    )
    parser.add_argument(
        "--init_DF",
        action="store_true",
        help="If false, creates Data Frame just from the XZ data. If true, load also the Data Frames contained in load_dir",
    )
    args = parser.parse_args()
    load_dir=args.load_dir
    save_dir = args.save_dir
    print_it = args.print_it

    flag = 0
    if args.init_DF == True:
        for filename in os.listdir(load_dir):
            path = os.path.join(load_dir, filename)
            if not os.path.isdir(path):
                if ( (print_it == True) and (filename.startswith( "DF_IT_" ))) or ( (print_it == False) and (filename.startswith( "DF_" )) and not (filename.startswith( "DF_IT_" )) ):
                    with lzma.open(load_dir + filename, "rb") as f:
                        if flag == 0 : 
                            data_frame = pickle.load(f)
                            flag = 1
                        else: data_frame = pd.concat([data_frame,pickle.load(f)],ignore_index=True)

    dict_list = []
    for filename in os.listdir(load_dir):
        path = os.path.join(load_dir, filename)
        if not os.path.isdir(path):
            if not filename.startswith('.'):
                if ( (print_it == True) and (filename.startswith('IT_'))) or ( (print_it == False) and not (filename.startswith('DF_')) ):
                    with lzma.open(load_dir + filename, "rb") as f:
                        data = pickle.load(f)
                        dict_list = dict_list + data_to_dict(data)
                                    
    if flag == 0: data_frame = pd.DataFrame(dict_list)
    else : data_frame = pd.concat([data_frame,pd.DataFrame(dict_list)],ignore_index=True)
    timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(random.randint(1,1000))
    if print_it : file_name = "DF_IT_" + timestr + ".xz"  
    else : file_name = "DF_" + timestr + ".xz" 
    with lzma.open(save_dir + file_name, "wb") as f:
        pickle.dump(data_frame, f)

def data_to_dict(data1,data2,init_type): #TO BE CHANGED FOR THE CASE DIFFERENT FROM RND+INF
    
    if init_type== 0:
        single_dict_list = DtoD(data1,data2,init="rnd")
        #single_dict_list.append(dict(zip(keys, values))) 
    elif init_type== 1:
        single_dict_list = DtoD(data1,data2,init="inf")
    elif init_type== 2:
        [SEmess_list,
        marg_list_rnd, 
        marg_list_inf,
        eR_list,
        eI_list,
        itR_list,
        itI_list,
        logLR_list,
        logLI_list,
        errR_list,
        errI_list
        ] = data2
        data2R = [
        SEmess_list,
        marg_list_rnd, 
        eR_list,
        itR_list,
        logLR_list,
        errR_list]
        single_dict_list = DtoD(data1,data2R,init="rnd") 
        data2I = [
        SEmess_list,
        marg_list_inf, 
        eI_list,
        itI_list,
        logLI_list,
        errI_list]
        single_dict_list = single_dict_list + DtoD(data1,data2I,init="inf") 
    else:
        single_dict_list = DtoD(data1,data2,init="unif")
    
    return single_dict_list

if __name__ == "__main__":
    main()


def DtoD(data1,data2,init):

    keys = [
        r"init",
        r"graph_type",
        r"$N$",
        r"$d$",
        r"$\lambda$",
        r"s_type",
        r"S",
        r"o_type",
        r"M",
        r"iter_space",
        r"seed",
        r"tol",
        r"n_iter",
        r"obs_type",
        r"snap_time",
        r"T_max",
        r"mask_type",
        r"$\mu$",
        r"tol2",
        r"it_max",
        r"$T$",
        r"$f_S$",
        r"$f_I$",
        r"$T_O$",
        r"$T_{BP}$",
        r"$infer_up_to$",
        r"$\Delta$",
        r"damping",
        r"error",
        r"iteration",
        r"it_final",
        r"logL",
        r"$O_{t=0}$",
        r"$O_{t=0,RND}$",
        r"$MO_{t=0}$",
        r"$MO_{t=0,RND}$",
        r"$\delta O_{t=0}$",
        r"$\widetilde{O}_{t=0}$",
        r"$\widetilde{MO}_{t=0}$",
        r"$\widetilde{\delta O}_{t=0}$",
        r"$O_{t=T}$",
        r"$O_{t=T,RND}$",
        r"$MO_{t=T}$",
        r"$MO_{t=T,RND}$",
        r"$\delta O_{t=T}$",
        r"$\widetilde{O}_{t=T}$",
        r"$\widetilde{MO}_{t=T}$",
        r"$\widetilde{\delta O}_{t=T}$",
        r"SE",
        r"MSE",
        r"$SE_{RND}$",
        r"$MSE_{RND}$",
        r"$\delta SE$",
        r"$R_{SE}$",
        r"$R_{MSE}$",
        r"$\delta R_{SE}$",
        r"ConvChecks",
        r"$SE_{mess}$"
    ]
    [graph, 
    N,
    d,
    lam,
    s_type,
    S,
    o_type,
    M,
    iter_space,
    seed,
    tol,
    n_iter,
    obs_type,
    snap_time,
    T_max,
    mask,
    mask_type,
    mu,
    tol2,
    it_max,
    ground_truth,
    G,
    T,
    list_obs,
    fS,
    fI,
    TO,
    T_BP,
    infer_up_to,
    Delta,
    damp
    ] = data1  
    [SEmess_list,
    marg_list, 
    e_list,
    it_list,
    logL_list,
    err_list
    ] = data2

    dict_list=[]
    for it_idx, B in enumerate(marg_list):   
                   
        MI0 = B[:, 0]
        MS0 = 1-MI0
        MR0 = np.zeros_like(MS0)
        M0 = np.array([ MS0, MI0, MR0])
        #Compute general marginals:
        pS = 1 - np.cumsum(B,axis=1)
        if mask == ["SI"] : pI = np.cumsum(B,axis=1)
        elif mask_type == "SIR" : pI = np.array([ np.array([ sum(np.array([ mask[t-ti-1] if ((t>ti) and (t-ti <= len(mask))) else 0 for ti in np.arange(-1,T_BP+1) ]) * b ) for t in range(T_BP+2)]) for b in B])
        else : pI = np.array([ np.array([ sum(b[1+t-Delta:1+t]) if t>=Delta else sum(b[:t+1]) for t in range(T_BP+2)]) for b in B])
        pR = 1 - pS - pI
        #Take time T
        MST = pS[:,T_BP]
        MIT = pI[:,T_BP]
        MRT = pR[:,T_BP]
        MT = np.array([ MST, MIT, MRT])
        x0_inf = np.argmax(M0,axis=0)
        xT_inf = np.argmax(MT,axis=0)
        ti_str = ti_star(ground_truth[:T_BP+1],T_BP)

        ov0 = OV(ground_truth[0], x0_inf)
        ov0_rnd = OV_rnd(ground_truth[0], M0)
        mov0 = MOV(M0)
        mov0_rnd = MOV_rnd(M0)
        ovT = OV(ground_truth[min(T,T_BP)], xT_inf)
        ovT_rnd = OV_rnd(ground_truth[min(T,T_BP)], MT)
        movT = MOV(MT)
        movT_rnd = MOV_rnd(MT)
        ti_inf = ti_inferred(B)
        ti_rnd = ti_random(B)
        se = SE(ti_str, ti_inf)
        mse = MSE(B, ti_inf)
        se_rnd = SE(ti_str, ti_rnd)
        mse_rnd = MSE(B, ti_rnd)
        e = e_list[it_idx]
        it = it_list[it_idx]
        it_final = it_list[-1]
        logL = logL_list[it_idx]
        ov0t = (ov0 - ov0_rnd) / (1 - ov0_rnd)
        mov0t = (mov0 - mov0_rnd) / (1 - mov0_rnd)
        ovTt = (ovT - ovT_rnd) / (1 + 1e-12 - ovT_rnd)
        movTt = (movT - movT_rnd) / (1 + 1e-12 - movT_rnd)
        Rse = (se_rnd - se) / se_rnd
        Rmse = (mse_rnd - mse) / mse_rnd
        SEmess = SEmess_list[it_idx]
        
        values = [
            init,
            graph, 
            N,
            d,
            lam,
            s_type,
            S,
            o_type,
            M,
            iter_space,
            seed,
            tol,
            n_iter,
            obs_type,
            snap_time,
            T_max,
            mask_type,
            mu,
            tol2,
            it_max,
            T,
            fS,
            fI,
            TO,
            T_BP,
            infer_up_to,
            Delta,
            damp,
            e,
            it,
            it_final,
            float(logL),
            ov0,
            ov0_rnd,
            mov0,
            mov0_rnd,
            ov0 - mov0,
            ov0t,
            mov0t,
            ov0t - mov0t,
            ovT,
            ovT_rnd,
            movT,
            movT_rnd,
            ovT - movT,
            ovTt,
            movTt,
            ovTt - movTt,
            se,
            mse,
            se_rnd,
            mse_rnd,
            se - mse,
            Rse,
            Rmse,
            Rse - Rmse,
            err_list,
            SEmess
        ]
        dict_list.append(dict(zip(keys, values))) 

    return dict_list