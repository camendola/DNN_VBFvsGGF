from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import *
from sklearn.externals import joblib

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
import modules.plot as plotter


import pandas as pd


def create_model(
    neurons=[100, 50, 10],
    branches=10,
    dropout_rate=None,
    learn_rate=None,
    momentum=None,
):
    # create model
    model = Sequential()
    model.add(Dense(neurons[0], init="uniform", input_dim=branches, activation="relu"))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    if len(neurons) > 1:
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        model.add(Dense(neurons[1], init="uniform", activation="relu"))
    if len(neurons) > 2:
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        model.add(Dense(neurons[2], init="uniform", activation="relu"))
    if len(neurons) > 3:
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        model.add(Dense(neurons[3], init="uniform", activation="relu"))
    if len(neurons) > 4:
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        model.add(Dense(neurons[4], init="uniform", activation="relu"))
    model.add(Dense(1, init="uniform", activation="sigmoid"))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    if learn_rate:
        optimizer = SGD(lr=learn_rate)
    elif momentum:
        optimizer = SGD(momentum=momentum)
    if learn_rate and momentum:
        optimizer = SGD(lr=learn_rate, momentum=momentum)
    # Compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def opt_neurons(input, output, neurons, branches_size, channel, selection):
    # Split the dataset in two equal parts
    # branches = branches
    model = KerasClassifier(
        build_fn=create_model,
        dropout_rate=0.2,
        branches=branches_size,
        learn_rate=0.01,
        epochs=50,
        batch_size=10,
        verbose=0,
    )
    # define the grid search parameters
    neurons = neurons
    print(neurons)
    param_grid = dict(neurons=neurons)
    kfold = StratifiedKFold(5, True, 4567)
    scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=1,
        cv=kfold,
        scoring=scoring,
        refit="AUC",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        input, output, test_size=0.8, random_state=1
    )
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    results = grid_result.cv_results_

    accuracy_score_mean = results["mean_test_Accuracy"]
    accuracy_score_std = results["std_test_Accuracy"]
    #
    AUC_score_mean = results["mean_test_AUC"]
    AUC_score_std = results["std_test_AUC"]
    #
    txt = open(
        "models/DNNnodes_channel_%s_selection_%s.txt" % (channel, selection), "w"
    )

    params = results["params"]
    print(params)

    for i in range(0, len(params)):
        print(
            "Accuracy = %.2f%% +/- %.2f%%, AUC = %.4f +/- %.4f, with %r"
            % (
                accuracy_score_mean[i] * 100,
                accuracy_score_std[i] * 100,
                AUC_score_mean[i],
                AUC_score_std[i],
                params[i],
            )
        )
        txt.write(
            "Accuracy = %.2f%% +/- %.2f%%, AUC = %.4f +/- %.4f, with %r\n"
            % (
                accuracy_score_mean[i] * 100,
                accuracy_score_std[i] * 100,
                AUC_score_mean[i],
                AUC_score_std[i],
                params[i],
            )
        )

    print("Best AUC: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    txt.close()


def cv_NN(input, output, neurons, branches_size, channel, selection, show):

    cvscores = []
    AUC = []
    train_cvscores = []
    train_AUC = []
    kfold = StratifiedKFold(5, True, 6789)
    for train, test in kfold.split(input, output):
        model = create_model(
            neurons=neurons, branches=branches_size, dropout_rate=0.2, learn_rate=0.01
        )
        X_train, X_test, y_train, y_test = (
            input[train],
            input[test],
            output[train],
            output[test],
        )
        model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
        scores = model.evaluate(X_test, y_test)
        prediction = model.predict(X_test)
        auc = roc_auc_score(y_test, prediction)
        print(
            "%s: %.2f%%; AUC = %.4f%%" % (model.metrics_names[1], scores[1] * 100, auc)
        )

        cvscores.append(scores[1] * 100)
        AUC.append(auc)
        train_scores = model.evaluate(X_train, y_train)
        train_prediction = model.predict(X_train)
        train_auc = roc_auc_score(y_train, train_prediction)
        train_cvscores.append(train_scores[1] * 100)
        train_AUC.append(train_auc)

    print(
        "Accuracy test= %.2f%% (+/- %.2f%%); AUC = %.2f (+/- %.2f)"
        % (np.mean(cvscores), np.std(cvscores), np.mean(AUC), np.std(AUC))
    )
    print(
        "Accuracy train= %.2f%% (+/- %.2f%%); AUC = %.2f (+/- %.2f)"
        % (
            np.mean(train_cvscores),
            np.std(train_cvscores),
            np.mean(train_AUC),
            np.std(train_AUC),
        )
    )
    name = "channel_" + str(channel) + "_neurons"

    for i in neurons:
        name = "%s_%d" % (name, i)
    name = "%s_%s" % (name, selection)

    if show:
        plotter.plot_separation(model, X_test, y_test, name, True)
        plotter.plot_ROC(model, X_test, y_test, name, True)


def train_NN(input, output, neurons, branches_size, channel, selection, show, scaler):

    cvscores = []
    AUC = []

    model = create_model(
        neurons=neurons, branches=branches_size, dropout_rate=0.2, learn_rate=0.01
    )

    model.fit(input, output, epochs=50, batch_size=10, verbose=0)
    name = "root_channel_" + str(channel) + "_neurons"
    for i in neurons:
        name = "%s_%d" % (name, i)
    name = "%s_%s" % (name, selection)
    scaler_filename = "scaler_%s.pkl" % name
    joblib.dump(scaler, scaler_filename)
    modelname = "models/%s.h5" % name
    print("Save to %s" % modelname)
    model.save(modelname)


def opt_BDT(input, output, params, show, names):

    model = XGBClassifier(**params)
    xgb_param = model.get_xgb_params()
    cvscores = []
    AUC = []
    X_train, X_test, y_train, y_test = train_test_split(
        input, output, test_size=0.2, random_state=42
    )
    matrix_train = xgb.DMatrix(X_train, label=y_train)
    cvresult = xgb.cv(
        xgb_param,
        matrix_train,
        num_boost_round=model.get_params()["n_estimators"],
        nfold=5,
        metrics="auc",
        early_stopping_rounds=30,
        verbose_eval=True,
    )
    model.set_params(n_estimators=cvresult.shape[0])
    model.fit(X_train, y_train, eval_metric="auc")
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    prediction = [round(value) for value in y_pred]
    auc = roc_auc_score(y_test, y_prob[:, 1])
    accuracy = accuracy_score(y_test, prediction)

    print("Accuracy: %.2f%%; AUC = %.4f%" % (accuracy * 100, auc))
    if show:

        name = "channel_" + str(channel) + "_BDT"
        name = "%s_%s" % (name, selection)
        modelname = "models/%s.h5" % name
        print("Save to %s" % modelname)

        plotter.plot_separation(model, X_test, y_test, name, False)
        plotter.plot_ROC(model, X_test, y_test, name, False)
        model.get_booster().feature_names = names
        mp.rc("figure", figsize=(5, 5))
        plot_importance(model.get_booster())
        plt.subplots_adjust(left=0.3)
        plt.show()


def cv_BDT(input, output, params, show, channel, selection, names):

    # model = XGBClassifier()

    cvscores = []
    AUC = []

    cvscores_train = []
    AUC_train = []
    kfold = StratifiedKFold(5, True, 3456)
    for train, test in kfold.split(input, output):
        model = XGBClassifier(**params)
        X_train, X_test, y_train, y_test = (
            input[train],
            input[test],
            output[train],
            output[test],
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        prediction = [round(value) for value in y_pred]
        auc = roc_auc_score(y_test, y_prob[:, 1])
        accuracy = accuracy_score(y_test, prediction)
        print("Accuracy: %.2f%%; AUC = %.4f%%" % (accuracy * 100, auc))
        cvscores.append(accuracy * 100)
        AUC.append(auc)

        y_prob = model.predict_proba(X_train)
        y_pred = model.predict(X_train)
        prediction = [round(value) for value in y_pred]
        auc = roc_auc_score(y_train, y_prob[:, 1])
        accuracy = accuracy_score(y_train, prediction)
        print("Accuracy train: %.2f%%; AUC = %.4f%%" % (accuracy * 100, auc))
        cvscores_train.append(accuracy * 100)
        AUC_train.append(auc)

    print(
        "Accuracy test = %.2f%% (+/- %.2f%%); AUC = %.4f (+/- %.4f)"
        % (np.mean(cvscores), np.std(cvscores), np.mean(AUC), np.std(AUC))
    )
    print(
        "Accuracy train = %.2f%% (+/- %.2f%%); AUC = %.4f (+/- %.4f)"
        % (
            np.mean(cvscores_train),
            np.std(cvscores_train),
            np.mean(AUC_train),
            np.std(AUC_train),
        )
    )
    if show:

        name = "channel_" + str(channel) + "_BDT"
        name = "%s_%s" % (name, selection)
        modelname = "models/%s.h5" % name
        print("Save to %s" % modelname)
        plotter.plot_separation(model, X_test, y_test, name, False)
        plotter.plot_ROC(model, X_test, y_test, name, False)
        model.get_booster().feature_names = names
        mp.rc("figure", figsize=(5, 5))

        plot_importance(
            model.get_booster(), max_num_features=15, importance_type="gain"
        )
        plt.subplots_adjust(left=0.3)
        plt.show()


def best_early_stopping(input, output, args, neurons, branches_size, selection, show):

    X_train, X_test, y_train, y_test = train_test_split(
        input, output, test_size=0.30, random_state=1
    )
    model = create_model(neurons=neurons, branches=branches_size, dropout_rate=0.2)

    print(branches_size)

    early_stop = EarlyStopping(monitor="val_loss", patience=100)

    if args.name:
        name = args.name
    else:
        name = "channel_" + str(args.channel) + "_neurons"
        for i in neurons:
            name = "%s_%d" % (name, i)
        name = "%s_%s" % (name, selection)
    modelname = "models/best_earlystop_%s.h5" % name
    print("Save to %s" % modelname)

    mc = ModelCheckpoint(
        modelname, monitor="val_acc", mode="max", verbose=1, save_best_only=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.3,
        epochs=1000,
        initial_epoch=0,
        batch_size=100,
        shuffle=True,
        callbacks=[mc, early_stop],
    )

    if show:
        mp.rc("figure", figsize=(5, 5))

        plt.plot(history.epoch, history.history["loss"], label="loss")
        plt.plot(history.epoch, history.history["val_loss"], label="val_loss")

        plt.legend()
        plt.savefig("plots/history_%s.pdf" % name)
        plt.show()

    saved_model = load_model(modelname)
    # evaluate the model
    _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    print("Train: %.3f, Test: %.3f" % (train_acc, test_acc))
