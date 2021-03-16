from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import *
import numpy as np
import uproot
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, make_scorer
import matplotlib as mp
import argparse

from sklearn import metrics
from keras import backend as K

from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV

##BDT

from xgboost import XGBClassifier



branches_NN = ["VBFjj_deltaEta", "VBFjj_mass", "VBFjet1_pt", "VBFjet2_pt", "VBFjet1_eta", "VBFjet2_eta", "VBFjet1_btag", "VBFjet2_btag", "VBFjj_HT", "dau1_pt", "dau2_pt", "jet5_VBF_eta", "jet5_VBF_pt","jet5_VBF_btag", "dau1_MVAisoNew", "dau2_MVAisoNew","bjet1_bID", "bjet2_bID", "njets20","dau1_z","dau2_z","bjet1_z", "bjet2_z", "tauH_z", "bH_z", "HH_z","VBFjj_dEtaSign","bH_VBF1_deltaEta","dib_dEtaSign", "jet3_pt", "jet3_eta","jet4_pt", "jet4_eta", "jj_mass", "jj_deltaEta"]





def create_model(neurons=[400, 400, 300],dropout_rate=None, learn_rate=None, momentum=None, branches_NN = branches_NN):
    # create model
    model = Sequential()

    model.add(Dense(neurons[0],init='uniform', input_dim=len(branches_NN), activation='relu' ))
    if len(neurons) > 1:
        model.add(Dense(neurons[1], init='uniform', activation='relu'))
    if len(neurons) > 2:
        model.add(Dense(neurons[2], init='uniform', activation='relu'))
    if len(neurons) > 3:
        model.add(Dense(neurons[3], init='uniform', activation='relu'))
    if len(neurons) > 4:
        model.add(Dense(neurons[4], init='uniform', activation='relu'))

    model.add(Dense(1, init='uniform', activation='sigmoid'))
    if dropout_rate: model.add(Dropout(dropout_rate))
    if learn_rate and momentum: optimizer = SGD(lr=learn_rate, momentum=momentum)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model









sigDir = '/data_CMS/cms/amendola/HH2017Skim_Jan2019/SKIMS_6June2019_Run2017_0jets_VBFtrigger/SKIM_VBFSM'
bkgDir = '/data_CMS/cms/amendola/HH2017Skim_Jan2019/SKIMS_6June2019_Run2017_0jets_VBFtrigger/SKIM_GGHSM'



parser = argparse.ArgumentParser(description='Command line parser of plotting options')

#string opts
parser.add_argument('--sep', dest='sep', help='separation plot', action='store_true', default=False)
parser.add_argument('--ROC', dest='ROC', help='roc curve', action='store_true',default=False)
parser.add_argument('--name', dest='name', help='selection name for plot', action='store_true',default=None)
parser.add_argument('--dir', dest='dir', help='analysis output folder name', action='store_true',default="./")
parser.add_argument('--cv', dest = 'cv', help='use cross validation to estimate the performance', action='store_true', default=False)
parser.add_argument('--optimize', dest = 'optimize', help='optimize neurons', action='store_true', default=False)
parser.add_argument('--makeBDT', dest = 'makeBDT', help='BDT', action='store_true', default=False)
#parser.add_argument('--callbacks', dest = 'callbacks', help='list of callbacks', default=list(mc))

args = parser.parse_args()




#fill the array

sig = uproot.open( sigDir+"/total_tt_baseline_VBFloose.root")["HTauTauTree"] 
sigTree= np.array(list(sig.arrays(branches = branches_NN).values())).T

        
bkg = uproot.open( bkgDir+"/total_tt_baseline_VBFloose.root")["HTauTauTree"] 
bkgTree = np.array(list(bkg.arrays(branches = branches_NN).values())).T
print("VBF entries = %d" % sigTree.shape[0])
print("ggF entries = %d" % bkgTree.shape[0])

sigT = sigTree[:min(sigTree.shape[0], bkgTree.shape[0])]
bkgT = bkgTree[:min(sigTree.shape[0], bkgTree.shape[0])]

#0s for ggF, 1s for VBFs
output = np.append( np.ones(sigT.shape[0]), np.zeros(bkgT.shape[0]))
input_raw = np.append(sigT, bkgT, axis = 0) 

scaler = StandardScaler()
input = scaler.fit_transform(input_raw)

print("VBF entries = %d" % sigT.shape[0])
print("ggF entries = %d" % bkgT.shape[0])



## create model

if args.optimize:
# Split the dataset in two equal parts
    
    model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=10, verbose=0)
    # define the grid search parameters
    #neurons = [[400, 200, 10], [100,50, 10], [400, 400, 100], [400, 100, 100], [100, 100, 10], [400, 400, 300] , [300, 200, 100, 100], [400, 300, 200, 100]]
    neurons = [[400, 200, 10], [100,50, 10], [100, 100, 10], [400, 100] , [100,100], [100, 10], [50,10], [20, 10]]
    #neurons = [[10], [20]]
    param_grid = dict(neurons = neurons)
    print(param_grid)
    kfold = StratifiedKFold(5, True, 4567)
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv = kfold, scoring = scoring, refit='AUC')
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.8, random_state=1)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    
    
    results = grid_result.cv_results_
    #print (results)

    accuracy_score_mean = results['mean_test_Accuracy']
    accuracy_score_std = results['std_test_Accuracy']
    #
    AUC_score_mean = results['mean_test_AUC']
    AUC_score_std = results['std_test_AUC']
    #
    params = results['params']
    print(params)
    for i in range(0,len(params)):
        print ('Accuracy = %.2f%% +/- %.2f%%, AUC = %.2f +/- %.2f, with %r'%  (accuracy_score_mean[i]*100, accuracy_score_std[i]*100 ,AUC_score_mean[i], AUC_score_std[i], params[i]))           
    
    print("Best AUC: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #test_model = create_model()
    #prediction= model.predict(X_test)
    #auc = roc_auc_score(y_test, prediction) 
    #print("%s: %.2f%%; AUC = %.2f%%" % (model.metrics_names[1], scores[1]*100, auc))



if args.cv:
    model = create_model(neurons = [20, 10])
    cvscores = []
    AUC = []
    kfold = StratifiedKFold(5, True, 3456)
    for train, test in kfold.split(input, output):
        X_train, X_test, y_train, y_test = input[train], input[test], output[train], output[test]
        model.fit(X_train, y_train, epochs=7, batch_size=10, verbose = 0)
        scores = model.evaluate(X_test, y_test)
        prediction= model.predict(X_test)
        auc  = roc_auc_score(y_test, prediction) 
        print("%s: %.2f%%; AUC = %.2f%%" % (model.metrics_names[1], scores[1]*100, auc))
        cvscores.append(scores[1] * 100)
        AUC.append(auc)

    print("Accuracy = %.2f%% (+/- %.2f%%); AUC = %.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores), np.mean(AUC), np.std(AUC)))






if args.makeBDT:

    model = XGBClassifier()
    cvscores = []
    AUC = []
    kfold = StratifiedKFold(5, True, 3456)
    for train, test in kfold.split(input, output):
        X_train, X_test, y_train, y_test = input[train], input[test], output[train], output[test]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prediction = [round(value) for value in y_pred]
        auc  = roc_auc_score(y_test, prediction) 
        accuracy = accuracy_score(y_test, prediction)
        print("Accuracy: %.2f%%; AUC = %.2f%%" %  (accuracy*100, auc))
        cvscores.append(accuracy* 100)
        AUC.append(auc)

    print("Accuracy = %.2f%% (+/- %.2f%%); AUC = %.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores), np.mean(AUC), np.std(AUC)))





if not args.cv and not args.optimize and not args.makeBDT:

    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.30, random_state=1)
    model = create_model(neurons=[100, 10])

    #   min_delta = 0.0001
    #   p_epochs = 5
    early_stop = EarlyStopping(monitor='val_loss', patience = 5)#, min_delta=min_delta, 
    #                               patience=p_epochs, verbose=1)
    
    
    mc = ModelCheckpoint('best_model_tt.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    

    history = model.fit(X_train, y_train,
                        validation_split = 0.2,
                        epochs=10, initial_epoch=0,
                        batch_size=10, shuffle=True, callbacks=[mc, early_stop])
                        #callbacks=[mc, early_stop])
                        

    mp.rc('figure', figsize=(5,5))

    plt.plot(history.epoch, history.history["loss"], label="loss")
    plt.plot(history.epoch, history.history["val_loss"], label = "val_loss")

    plt.legend()
    plt.savefig('history.pdf')
    plt.show()

    

    saved_model = load_model('best_model_tt.h5')
    # evaluate the model
    _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))



 
    results_signal = saved_model.predict(X_test[y_test==1])
    results_bkg = saved_model.predict(X_test[y_test==0])
    
    mp.rc('figure', figsize=(5,5))
    
    plt.hist(results_signal, bins = 20, range =(0,1), label="signal")
    plt.hist(results_bkg, bins = 20, range = (0,1), label="bkg", alpha = .5)
    plt.legend()
    plt.savefig('separation.png')
    plt.show()


    prediction = saved_model.predict(X_test)
    mp.rc('figure', figsize=(5,5),  dpi=140)
    fp , tp, th = roc_curve(y_test, prediction)
    plt.plot(fp, tp, 'r')
    plt.xlabel('false positive')
    plt.ylabel('true positive')
    plt.savefig('roc.png')
    plt.show()

