import numpy as np
import os
import argparse

from sklearn.model_selection import train_test_split, StratifiedKFold
import modules.load_data as load_data
import modules.methods as methods


from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, make_scorer

branches_NN = [
    "pairType",
    "nleps",  # general
    "dau1_MVAisoNew",
    "dau2_MVAisoNew",
    "dau1_iso",
    "dau2_iso",
    "dau1_pt",
    "dau2_pt",
    "dau1_eta",
    "dau2_eta",  # tau candidates
    "bjet1_pt",
    "bjet2_pt",
    "bjet1_eta",
    "bjet2_eta",
    "dib_dEtaSign",
    "nbjetscand",  # b jet candidates
    "VBFjj_deltaEta",
    "VBFjj_mass",
    "VBFjet1_pt",
    "VBFjet2_pt",
    "VBFjet1_eta",
    "VBFjet2_btag",
    "VBFjj_HT",
    "VBFjj_dEtaSign",  # VBF candidates
    "jj_deltaEta",
    "jet5_VBF_eta",
    "jet5_VBF_pt",
    "jet3_pt",
    "jet3_eta",
    "jet4_pt",
    "jet4_eta",
    "jj_mass",  # additional jets
    "tauH_mass",
    "tauH_pt",
    "tauH_SVFIT_mass",
    "tauH_SVFIT_pt",  # Htt
    "bH_mass",
    "bH_pt",
    "bH_VBF1_deltaEta",  # Hbb
    "isBoosted",
    "isVBF",
    "isVBFtrigger",  # flags
    "dau1_z",
    "dau2_z",
    "bjet1_z",
    "bjet2_z",
    "tauH_z",
    "bH_z",
    "HH_z",
    "jet5_VBF_z",
    "HH_zV",
    "HH_A",  # centrality
]


# sigDir = '/data_CMS/cms/amendola/HH2017Skim_Jan2019/SKIMS_6June2019_Run2017_0jets_VBFtrigger/SKIM_VBFSM/output_*.root'
# bkgDir = '/data_CMS/cms/amendola/HH2017Skim_Jan2019/SKIMS_6June2019_Run2017_0jets_VBFtrigger/SKIM_GGHSM/output_*.root'

sigDir = "/data_CMS/cms/amendola/HH2017Skim_Jan2019/SKIMS_20Sept2019_Run2017_0jets_privatesignals/SKIM_VBF*/output_*.root"
bkgDir = "/data_CMS/cms/amendola/HH2017Skim_Jan2019/SKIMS_20Sept2019_Run2017_0jets_privatesignals/SKIM_GGHSM_private/output_*.root"


parser = argparse.ArgumentParser(description="Command line parser of plotting options")

parser.add_argument(
    "--channel", type=int, dest="channel", help="which channel", default=2
)
parser.add_argument(
    "--selection", dest="selection", help="type-baseline-looseVBF-tightVBF", default=""
)


parser.add_argument("--name", dest="name", help="selection name for plot", default=None)
parser.add_argument(
    "--dir", dest="dir", help="analysis output folder name", default="./"
)
parser.add_argument(
    "--cv",
    dest="cv",
    help="use cross validation to estimate the performance",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--train",
    dest="train",
    help="train on full dataset",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--cv_BDT",
    dest="cv_BDT",
    help="use cross validation to estimate the performance",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--optimize",
    dest="optimize",
    help="optimize neurons",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--optimize_BDT",
    dest="optimize_BDT",
    help="optimize BDT",
    action="store_true",
    default=False,
)
parser.add_argument("--neurons", dest="neurons", nargs="+", type=int, default=False)

parser.add_argument(
    "--epochs",
    dest="epochs",
    help="choose the best model with early stopping",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--show", dest="show", help="show plot", action="store_true", default=False
)


args = parser.parse_args()


# fill the array

selection = args.selection

sigTree = load_data.load_chain(
    sigDir, "HTauTauTree", branches_NN, args.channel, selection
)
bkgTree = load_data.load_chain(
    bkgDir, "HTauTauTree", branches_NN, args.channel, selection
)

print("VBF entries = %d" % sigTree.shape[0])
print("ggF entries = %d" % bkgTree.shape[0])

sigT = sigTree[: min(sigTree.shape[0], bkgTree.shape[0])]
bkgT = bkgTree[: min(sigTree.shape[0], bkgTree.shape[0])]

# sigT = sigTree[:5000]
# bkgT = bkgTree[:5000]

# 0s for ggF, 1s for VBFs
output = np.append(np.ones(sigT.shape[0]), np.zeros(bkgT.shape[0]))
input_raw = np.append(sigT, bkgT, axis=0)

scaler = StandardScaler()
scaler.fit(input_raw)
input = scaler.transform(input_raw)

print("VBF entries = %d" % sigT.shape[0])
print("ggF entries = %d" % bkgT.shape[0])


branches_drop = load_data.drop_branches(args.channel, branches_NN, selection)
branches_cleaned = [i for i in branches_NN if not i in branches_drop]
print(branches_cleaned)
print(len(branches_NN))
print(len(branches_cleaned))
branches_size = len(branches_NN) - len(branches_drop)
print(branches_size)
if args.optimize:
    # neurons= [[600, 400, 10], [400,400, 10], [400, 200, 10], [100, 100, 10], [100,10, 10] , [400, 400 , 100, 10], [400, 300, 200, 100], [50,10], [20, 10]]
    neurons = [[100, 100, 10], [100, 10], [50, 10]]
    # neurons = [[400, 200, 10], [100,50, 10], [100, 100, 10]]#, [400, 100] , [100,100], [100, 10],
    # neurons = [50,10], [20, 10]
    methods.opt_neurons(input, output, neurons, branches_size, args.channel, selection)

elif args.cv:
    neurons = [100, 100, 10]
    methods.cv_NN(
        input, output, neurons, branches_size, args.channel, selection, args.show
    )

elif args.optimize_BDT:
    params = {
        "learning_rate": 0.1,
        "n_estimators": 200000,
        "max_depth": 5,
        "min_child_weight": 1,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "scale_pos_weight": 1,
        "seed": 27,
    }
    methods.opt_BDT(input, output, params, args.show, branches_cleaned)


elif args.cv_BDT:
    params = {
        "learning_rate": 0.1,
        "n_estimators": 10,
        "max_depth": 3,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:logistic",
        "scale_pos_weight": 1,
        "seed": 29,
    }
    methods.cv_BDT(
        input, output, params, args.show, args.channel, selection, branches_cleaned
    )

elif args.epochs:
    neurons = [100, 100, 10]
    methods.best_early_stopping(
        input, output, args, neurons, branches_size, selection, args.show
    )

elif args.train:
    if not args.neurons:
        neurons = [100, 100, 10]
    else:
        neurons = args.neurons
    methods.train_NN(
        input,
        output,
        neurons,
        branches_size,
        args.channel,
        selection,
        args.show,
        scaler,
    )


#
#    results_signal = saved_model.predict(X_test[y_test==1])
#    results_bkg = saved_model.predict(X_test[y_test==0])
#
#    mp.rc('figure', figsize=(5,5))
#
#    plt.hist(results_signal, bins = 20, range =(0,1), label="signal")
#    plt.hist(results_bkg, bins = 20, range = (0,1), label="bkg", alpha = .5)
#    plt.legend()
#    plt.savefig('separation.png')
#    plt.show()
#
#
#    prediction = saved_model.predict(X_test)
#    mp.rc('figure', figsize=(5,5),  dpi=140)
#    fp , tp, th = roc_curve(y_test, prediction)
#    plt.plot(fp, tp, 'r')
#    plt.xlabel('false positive')
#    plt.ylabel('true positive')
#    plt.savefig('roc.png')
#    plt.show()
