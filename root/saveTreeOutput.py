from keras.models import load_model
import sys, os
from array import array
import argparse
from tqdm import tqdm
import numpy as np
import uproot
import pandas as pd
from glob import glob
from ROOT import TFile, TTree, TCanvas
import ROOT
import joblib
from uproot.write.TFile import TFileUpdate as update

# And now to load...


parser = argparse.ArgumentParser()

parser.add_argument(
    "-n", "--name", type=str, required=False, default="name", help="Output branch name"
)

args = parser.parse_args()
batchsize = 4096
branchname = "DNN_VBFvsGGF"
newtree = True

model_names = {
    "MuTau": "models/channel_0_neurons_100_50_10_type-baseline-looseVBF.h5",
    "ETau": "models/channel_1_neurons_100_100_10_type-baseline-looseVBF.h5",
    "TauTauLoose": "models/channel_2_neurons_100_100_10_type-baseline-looseVBF.h5",
    "TauTauTight": "models/channel_2_neurons_100_10_type-baseline-tightVBF.h5",
}

scaler_names = {
    "MuTau": "models/scaler_channel_0_neurons_100_50_10_type-baseline-looseVBF.pkl",
    "ETau": "models/scaler_channel_1_neurons_100_100_10_type-baseline-looseVBF.pkl",
    "TauTauLoose": "models/scaler_channel_2_neurons_100_100_10_type-baseline-looseVBF.pkl",
    "TauTauTight": "models/scaler_channel_2_neurons_100_10_type-baseline-tightVBF.pkl",
}

# skimLocation='/data_CMS/cms/amendola/HH2017Skim_Jan2019/SKIMS_8Oct2019_Run2017_0jets_VBFtrigger_mh_BDT_Thesis/'
# skimLocation='/eos/home-c/camendol/SKIMS_Legacy2017/SKIMMED_Legacy2017_27mar2020_DNN_VBFvsGGF/'
skimLocation = "/data_CMS/cms/amendola/SKIMMED_Legacy2017_27mar2020_DNN/"
print("Skim directory set to:", skimLocation)
inputDirs = glob(skimLocation + "/SKIM*/")


branches_etau = [
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
    "isBoosted",  # flags
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

branches_mutau = [
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
    "isBoosted",  # flags
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

branches_tautau = [
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
    "isBoosted",  # flags
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

branches_tautautight = [
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


models = {}
scalers = {}
branches = {}
for i in model_names:
    models[i] = load_model(model_names[i])
    scalers[i] = joblib.load(scaler_names[i])
    if i == "ETau":
        branches[i] = branches_etau
    if i == "MuTau":
        branches[i] = branches_mutau
    if i == "TauTauLoose":
        branches[i] = branches_tautau
    if i == "TauTauTight":
        branches[i] = branches_tautautight

print(models)


for k, inputDir in enumerate(inputDirs):
    print("Transforming directory:", inputDir)
    if inputDir[-1] == "/":
        inputDir = inputDir[:-1]
    # if "lambdarew" in inputDir: continue
    # if "C2V_" in inputDir: continue

    inputList = inputDir + "/" + "goodfiles.txt"

    if not os.path.isfile(inputList):
        print("Could not find file:", inputList)
        sys.exit()
    if skimLocation[-1] == "/":
        skimLocation = skimLocation[:-1]

    inputListFile = open(inputList)
    for idx, line in enumerate(inputListFile):
        line = line.strip()
        fileName = line

        print(fileName)
        tfile = TFile(fileName, "update")
        data_tree = tfile.Get("HTauTauTree")
        name = args.name
        if data_tree.GetListOfBranches().Contains(branchname + "_" + name):
            continue
        tree = uproot.open(fileName)["HTauTauTree"]

        # score_branch = {}

        # print(name)
        # print(branches[name])
        score = array("f", [0.0])
        thisbranch = data_tree.Branch(
            branchname + "_" + name, score, "{0}/F".format(branchname + "_" + name)
        )
        with tqdm(total=data_tree.GetEntries()) as pbar:
            for events in tree.iterate(
                branches[name], entrysteps=batchsize, outputtype=pd.DataFrame
            ):
                events_rescaled = scalers[name].transform(events)
                scores = models[name].predict(events_rescaled, batch_size=batchsize)

                for i, s in enumerate(scores):
                    data_tree.GetEntry(i)
                    score[0] = s
                    thisbranch.Fill()
                    # print(score, i)
                    pbar.update()
        ROOT.gDirectory.Purge()
        # data_tree.Write()
        data_tree.Write("", ROOT.TObject.kOverwrite)
        tfile.Close()
