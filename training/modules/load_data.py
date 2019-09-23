import pandas as pd
import uproot


def drop_branches(channel, branches, whatToApply):
    branches_drop = []
    if "type" in whatToApply:
        branches_drop.append(b'pairType')
        if channel == 0 or channel == 1:
            branches_drop.append(b'dau1_MVAisoNew')
            branches_drop.append(b'dau2_MVAisoNew')
    if "trigger" in whatToApply:
        branches_drop.apppend(b'pass_trigger')
        if channel == 0 or channel == 1:
            branches_drop.apppend(b'isVBFtrigger')
    if "baseline" in whatToApply:
        branches_drop.append(b'nleps')
    if "looseVBF" in whatToApply:
        branches_drop.append(b'isVBF')
        branches_drop.append(b'isVBFtrigger')
    if "tightVBF" in whatToApply:
        branches_drop.append(b'isVBF')
        branches_drop.append(b'isVBFtrigger')
    return branches_drop
    

def load_chain(ntuples, tree, branch, channel, whatToApply):
    df = []
    chain = uproot.tree.iterate(ntuples, tree, branches = branch)
    for block in chain:
        df_b = pd.DataFrame(block)
        df_b = make_selection(df_b, channel, whatToApply)
        branches_drop = drop_branches(channel,branch, whatToApply)
        df_b = df_b.drop(columns = branches_drop)
        df.append(df_b)

    df = pd.concat(df, ignore_index=True)
    df.head()    
    return df


def make_selection(df_b, channel, whatToApply):
        if "type" in whatToApply:
            df_b = df_b[(df_b[b'pairType'] == channel)]

        if "baseline" in whatToApply:
            if channel == 0:
                df_b = df_b[(df_b[b'dau1_pt'] > 10) 
                            & (df_b[b'dau2_pt'] > 20) 
                            & (df_b[b'nleps'] == 0) 
                            & (df_b[b'nbjetscand'] > 1) 
                            & (df_b[b'dau1_iso'] > 0.5) 
                            & (df_b[b'dau2_MVAisoNew'] > 2)]
            if channel == 1:
                df_b = df_b[(df_b[b'dau1_pt'] > 10) 
                            & (df_b[b'dau2_pt'] > 20) 
                            & (df_b[b'nleps'] == 0) 
                            & (df_b[b'nbjetscand'] > 1) 
                            & (df_b[b'dau1_iso'] > 0.5) 
                            & (df_b[b'dau2_MVAisoNew'] > 2)]
            if channel == 2:
                df_b = df_b[(df_b[b'dau1_pt'] > 20) 
                            & (df_b[b'dau2_pt'] > 20) 
                            & (df_b[b'nleps'] == 0) 
                            & (df_b[b'nbjetscand'] > 1) 
                            & (df_b[b'dau1_MVAisoNew'] > 2) 
                            & (df_b[b'dau2_MVAisoNew'] > 2)]

        if "trigger" in whatToApply:
            df_b = df_b[(df_b[b'pass_trigger'] == True)]
        if "looseVBF" in whatToApply:
            df_b = df_b[(df_b[b'isVBF'] == True) 
                        & (df_b[b'isVBFtrigger'] == False) ]
        if "tightVBF" in whatToApply:
            df_b = df_b[(df_b[b'isVBF'] == True) 
                        & (df_b[b'isVBFtrigger'] == True) ]

        return df_b
