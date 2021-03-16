import matplotlib as mp
import matplotlib.pyplot as plt


from sklearn.metrics import roc_curve


def plot_separation(model, X_test, y_test, name, isNN):
    if isNN:
        name = "DNN_%s" % name
        results_signal = model.predict(X_test[y_test == 1])
        results_bkg = model.predict(X_test[y_test == 0])
    else:
        name = "BDT_%s" % name
        results_signal = model.predict_proba(X_test[y_test == 1])[:, 1]
        results_bkg = model.predict_proba(X_test[y_test == 0])[:, 1]

    mp.rc("figure", figsize=(5, 5))

    plt.hist(results_signal, bins=20, range=(0, 1), label="signal")
    plt.hist(results_bkg, bins=20, range=(0, 1), label="bkg", alpha=0.5)
    plt.legend()
    plt.savefig("plots/%s_separation.png" % name)
    plt.show()


def plot_ROC(model, X_test, y_test, name, isNN):
    if isNN:
        name = "DNN_%s" % name
        prediction = model.predict(X_test)
    else:
        name = "BDT_%s" % name
        prediction = model.predict_proba(X_test)[:, 1]

    mp.rc("figure", figsize=(5, 5), dpi=140)
    fp, tp, th = roc_curve(y_test, prediction)
    plt.plot(fp, tp, "r")
    plt.xlabel("false positive")
    plt.ylabel("true positive")
    plt.savefig("plots/%s_ROC.png" % name)
    plt.show()
    txt = open("plots/%s_ROC.txt" % name, "w")
    # txt.write("#AUC={0}\n".format(auc))
    txt.write("br,se,th\n")
    bkg_rej = 1 - fp
    for i in range(len(fp)):
        txt.write("{0},{1},{2}\n".format(bkg_rej[i], tp[i], th[i]))
    txt.close()
