import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp

mp.rc("figure", figsize=(10, 10), dpi=100)


colors = {
    "qcdww": (1, 0.4, 0),
    "singleTop": (0.4, 0.8, 1),
    "ttbar": (1, 0.8, 0.2),
    "wjets": (0.8, 0.2, 0.2),
    "ewk": (0.0, 0.4, 1),
}


rocDNN = pd.read_csv(
    "plots/DNN_channel_1_neurons_100_100_10_type-baseline-looseVBF_ROC.txt", header=0
)
rocBDT = pd.read_csv("plots/BDT_channel_2_BDT_type-baseline-looseVBF_ROC.txt", header=0)

aucDNN = float(0.89)
aucBDT = float(0.86)

print(rocDNN)

plt.figure(figsize=(5, 5), dpi=300)
plt.plot(
    rocDNN["se"], rocDNN["br"], label="AUC={:.2f} - DNN".format(aucDNN), c=(0.0, 0.4, 1)
)
plt.plot(
    rocBDT["se"],
    rocBDT["br"],
    label="AUC={:.2f} - BDT".format(aucBDT),
    c=(0.8, 0.2, 0.2),
)

plt.legend()
# plt.title("ROC - {} classifier".format(title))
plt.xlabel("VBF eff.")
plt.ylabel("ggF rej.")
plt.savefig("ROC_BDT_DNN.pdf")
plt.clf()
