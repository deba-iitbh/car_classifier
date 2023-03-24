import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier


def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


# Baseline
def svc_baseline(xtrain, ytrain, xtest, ytest):
    Baseline_model = SVC(kernel="rbf")
    Baseline_model.fit(xtrain, ytrain)

    ypred_train = Baseline_model.predict(xtrain)
    train_accuracy = np.mean(ypred_train == ytrain)
    ypred_test = Baseline_model.predict(xtest)
    test_accuracy = np.mean(ypred_test == ytest)
    return train_accuracy, test_accuracy


# Final Model
def ensemble_model(xtrain, ytrain, xtest, ytest):
    ens_model = BaggingClassifier(n_estimators=20, random_state=30)
    ens_model.fit(xtrain, ytrain)
    ypred_train = ens_model.predict(xtrain)
    train_accuracy = np.mean(ypred_train == ytrain)
    ypred_test = ens_model.predict(xtest)
    test_accuracy = np.mean(ypred_test == ytest)
    return train_accuracy, test_accuracy


if __name__ == "__main__":
    # Config
    DATA_DIR = "input"
    res = pd.DataFrame(
        {},
        columns=[
            "No. of Datapoints",
            "Bad Deal Percentage",
            "Train Accuracy",
            "Test Accuracy",
        ],
    )

    # Data Loading
    train_data = pd.read_csv(f"{DATA_DIR}/train_data.csv")
    test_data = pd.read_csv(f"{DATA_DIR}/test_data.csv")
    test_data = test_data.set_index("Deal_num")
    test_labels = pd.read_csv(f"{DATA_DIR}/test_labels.csv")
    test_labels = test_labels.set_index("Deal_num")

    # Categorical variable Encoder
    x_encoder = OrdinalEncoder(
        categories=[
            ["low", "med", "high", "vhigh"],
            ["low", "med", "high", "vhigh"],
            ["2", "3", "4", "5more"],
            ["2", "4", "more"],
            ["small", "med", "big"],
            ["low", "med", "high"],
        ]
    )
    y_encoder = OrdinalEncoder(categories=[["Bad_deal", "Nice_deal"]])

    # Data splitting into X and y
    xtrain, ytrain = (
        train_data.drop("How_is_the_deal", axis=1),
        train_data[["How_is_the_deal"]],
    )

    # Encode the categorical data
    xtrain = pd.DataFrame(
        x_encoder.fit_transform(xtrain), columns=train_data.columns[:-1]
    )
    xtest = pd.DataFrame(x_encoder.fit_transform(test_data), columns=test_data.columns)
    ytrain = pd.DataFrame(
        y_encoder.fit_transform(ytrain), columns=[train_data.columns[-1]]
    )
    ytest = pd.DataFrame(
        y_encoder.fit_transform(test_labels), columns=test_labels.columns
    )

    # Standard Normalization
    xtrain_sc, tr_scaler = scale_data(xtrain)
    xtest_sc, _ = scale_data(xtest, tr_scaler)

    svc_train_ac, svc_test_ac = svc_baseline(
        xtrain_sc, ytrain.values.reshape(-1), xtest_sc, ytest.values.reshape(-1)
    )
    bad_deal_perc = ytrain["How_is_the_deal"].value_counts(normalize=True)[1]
    res.loc["SVC"] = [
        xtrain.shape[0],
        bad_deal_perc,
        svc_train_ac,
        svc_test_ac,
    ]

    # SMOT
    smot = SMOTE(sampling_strategy="minority", k_neighbors=3)
    xsmot, ysmot = smot.fit_resample(xtrain.values, ytrain.values.reshape(-1))
    xsmot = pd.DataFrame(xsmot, columns=xtrain.columns)
    ysmot = pd.DataFrame(ysmot, columns=ytrain.columns)

    xsmot_sc, sc_scaler = scale_data(xsmot)
    xtest_sc, _ = scale_data(xtest, sc_scaler)
    svc_train_ac_smot, svc_test_ac_smot = svc_baseline(
        xsmot_sc, ysmot.values.reshape(-1), xtest_sc, ytest.values.reshape(-1)
    )
    print(type(ysmot))
    bad_deal_perc = ysmot["How_is_the_deal"].value_counts(normalize=True)[1]
    res.loc["SVC (SMOT)"] = [
        xsmot.shape[0],
        bad_deal_perc,
        svc_train_ac_smot,
        svc_test_ac_smot,
    ]

    ens_train_ac_smot, ens_test_ac_smot = ensemble_model(
        xsmot_sc, ysmot.values.reshape(-1), xtest_sc, ytest.values.reshape(-1)
    )
    res.loc["Ensemble (SMOT)"] = [
        xsmot.shape[0],
        bad_deal_perc,
        ens_train_ac_smot,
        ens_test_ac_smot,
    ]
    print(res)
