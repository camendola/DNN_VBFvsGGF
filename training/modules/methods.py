from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import *


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, make_scorer


from sklearn import metrics
from keras import backend as K

from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier, plot_importance
import xgboost as xgb

import matplotlib as mp
import matplotlib.pyplot as plt

import pandas as pd


def create_model(neurons=[100,50, 10],branches= 10,dropout_rate=None, learn_rate=None, momentum=None):
    # create model
    model = Sequential()
    model.add(Dense(neurons[0],init='uniform', input_dim=branches, activation='relu' ))
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


def opt_neurons(input, output, neurons, branches_size):
    # Split the dataset in two equal parts
    #branches = branches
    model = KerasClassifier(build_fn=create_model, branches= branches_size, epochs=15, batch_size=10, verbose=0)
    # define the grid search parameters
    neurons = neurons
    print(neurons)
    param_grid = dict(neurons = neurons)
    kfold = StratifiedKFold(5, True, 4567)
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv = kfold, scoring = scoring, refit='AUC')
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.8, random_state=1)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    results = grid_result.cv_results_

    accuracy_score_mean = results['mean_test_Accuracy']
    accuracy_score_std = results['std_test_Accuracy']
    #
    AUC_score_mean = results['mean_test_AUC']
    AUC_score_std = results['std_test_AUC']
    #
    params = results['params']
    print(params)
    for i in range(0,len(params)):
        print ('Accuracy = %.2f%% +/- %.2f%%, AUC = %.4f +/- %.4f, with %r'%  (accuracy_score_mean[i]*100, accuracy_score_std[i]*100 ,AUC_score_mean[i], AUC_score_std[i], params[i]))           
    
    print("Best AUC: %f using %s" % (grid_result.best_score_, grid_result.best_params_))




def cv_NN(input, output, neurons, branches_size):
    model = create_model(neurons = neurons, branches = branches_size)
    cvscores = []
    AUC = []
    kfold = StratifiedKFold(5, True, 3456)
    for train, test in kfold.split(input, output):
        X_train, X_test, y_train, y_test = input[train], input[test], output[train], output[test]
        model.fit(X_train, y_train, epochs=7, batch_size=10, verbose = 0)
        scores = model.evaluate(X_test, y_test)
        prediction= model.predict(X_test)
        auc  = roc_auc_score(y_test, prediction) 
        print("%s: %.2f%%; AUC = %.4f%%" % (model.metrics_names[1], scores[1]*100, auc))
        cvscores.append(scores[1] * 100)
        AUC.append(auc)

    print("Accuracy = %.2f%% (+/- %.2f%%); AUC = %.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores), np.mean(AUC), np.std(AUC)))


def opt_BDT(input, output, params, show, names):
    model = XGBClassifier(**params)
    xgb_param = model.get_xgb_params()
    cvscores = []
    AUC = []
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)
    matrix_train = xgb.DMatrix(X_train, label=y_train)
    cvresult = xgb.cv(xgb_param, matrix_train, num_boost_round=model.get_params()['n_estimators'], nfold=5, metrics='auc', early_stopping_rounds=20)

    model.fit(X_train, y_train,eval_metric='auc')
    model.set_params(n_estimators=cvresult.shape[0])
    y_pred = model.predict(X_test)
    prediction = [round(value) for value in y_pred]
    auc  = roc_auc_score(y_test, prediction) 
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy: %.2f%%; AUC = %.4f%%" %  (accuracy*100, auc))
    if show:
        model.get_booster().feature_names = names 
        mp.rc('figure', figsize=(5,5))
        plot_importance(model.get_booster())
        plt.subplots_adjust(left=0.3)       
        plt.show()







def cv_BDT(input, output, params, show, names):
    model = XGBClassifier(**params)
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
        print("Accuracy: %.2f%%; AUC = %.4f%%" %  (accuracy*100, auc))
        cvscores.append(accuracy * 100)
        AUC.append(auc)

    print("Accuracy = %.2f%% (+/- %.2f%%); AUC = %.4f (+/- %.4f)" % (np.mean(cvscores), np.std(cvscores), np.mean(AUC), np.std(AUC)))
    if show:
        model.get_booster().feature_names = names 
        mp.rc('figure', figsize=(5,5))

        plot_importance(model.get_booster(), max_num_features=15, importance_type = 'gain')
        plt.subplots_adjust(left=0.3)         
        plt.show()    


def best_early_stopping(input, output, args, neurons, branches_size, show):

    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.30, random_state=1)
    model = create_model(neurons=neurons, branches = branches_size)

    
    early_stop = EarlyStopping(monitor='val_loss', patience = 100)
    
    if args.name: 
        name = args.name
    else:
        name='channel_' + str(args.channel) + '_neurons'
        for i in neurons:
            name='%s_%d' % (name, i)
        name='%s_%s' % (name, selection)
    modelname = 'models/best_earlystop_%s.h5' % name
    print ("Save to %s" % modelname)

    mc = ModelCheckpoint(modelname, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    history = model.fit(X_train, y_train,
                        validation_split = 0.3,
                        epochs=100, initial_epoch=0,
                        batch_size=10, shuffle=True, callbacks=[mc, early_stop])

    if (show):
        mp.rc('figure', figsize=(5,5))

        plt.plot(history.epoch, history.history["loss"], label="loss")
        plt.plot(history.epoch, history.history["val_loss"], label = "val_loss")

        plt.legend()
        plt.savefig('plots/history_%s.pdf' % name)
        plt.show()

    saved_model = load_model(modelname)
    # evaluate the model
    _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
