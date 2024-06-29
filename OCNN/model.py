import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

from ocnn import OneClassNeuralNetwork


def main():
    file_path = 'data/satellite.csv'
    df = pd.read_csv(file_path)
    data = df.to_numpy()
    y_actual = data[:, -1]
    n=len(y_actual)
    lst=[]
    for x in y_actual:
        lst.append([x])
    y_actual=lst
    X = data[:, :-1]
    num_features = X.shape[1]
    num_hidden = 25
    r = 1.0
    epochs = 100
    nu = 0.10

    oc_nn = OneClassNeuralNetwork(num_features, num_hidden, n, r)
    model, history, train_history= oc_nn.train_model(X,y_actual, epochs=epochs, nu=nu, init_lr=0.01)

    plt.style.use("ggplot")
    plt.figure()
    
    lst=[]
    for i in range(1,epochs+1):
        lst.append(i)
    plt.plot(lst, train_history["recall"], label="Recall")
    plt.plot(lst, train_history["precision@5"], label="Precision@5")
    plt.plot(lst, train_history["precision@10"], label="Precison@10")
    plt.plot(lst, train_history["auc"], label="AUC_ROC")  
    plt.savefig(f'figures/annthyroid1.png')
    #plt.plot(history.epoch, history.history["auc_roc"], label="AUC_ROC")

    plt.title("OCNN Performance vs Epoch curve")
    plt.xlabel("Epoch")
    plt.ylabel("Performance Metric")
    plt.legend(loc="upper right")
    plt.show()

    # y_pred = model.predict(X)

    # r = history.history['r'].pop()

    # s_n = [y_pred[i, 0] - r >= 0 for i in range(len(y_pred))]



if __name__ == "__main__":
    main()
    exit()
