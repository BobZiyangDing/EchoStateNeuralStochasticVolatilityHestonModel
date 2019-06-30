import numpy
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import os
from sklearn.linear_model import Ridge

def EchoStateDeepTest(x_dim, connectivity, spectral_radius,
                  u_num, u_dim, u_mag,
                  leak, cutout, forget, 
                  cv_start, cv_end, cv_step, val_cut, verbose, input_U, target_Y, resvoir_num):
    if verbose:
        print()
        print("-----------------------------------------------Making Transition and input matrix------------------------------------------------")
        print()

    # Making inner transition sparse matrix W
    W_list = []
    for i in range(resvoir_num):
        nans = np.random.randint(0, int(1/connectivity)-1 , size=(x_dim, x_dim))
        W = np.random.uniform(-0.4, 0.4, x_dim*x_dim).reshape([x_dim, x_dim])
        W = np.where(nans, np.nan, W)
        W = np.nan_to_num(W)
        E, _ = np.linalg.eig(W)
        e_max = np.max(np.abs(E))
        W /= np.abs(e_max)/spectral_radius   
        W_list.append(W)
        
    #
    W_trans_list = []
    for i in range(resvoir_num-1):
        W_trans = np.random.uniform(-1, 1, x_dim*x_dim).reshape([x_dim, x_dim])
        W_trans_list.append(W_trans) 
        
        
    # Making input matrix W_in
    W_in = np.random.uniform(-1, 1, x_dim*u_dim).reshape([x_dim, u_dim])
    W_in = W_in / (np.linalg.svd(W_in)[1].tolist()[0]*1.2)

    if verbose:
        print("Shape of W: ", str(W.shape))
        print("Shape of W_in:", str(W_in.shape))

        print()
        print("-----------------------------------------------Making input and inner states------------------------------------------------------")
        print()

        
        
        
    # Making Inner States
    X = u
       
    x_lists = []
    x_list = [np.zeros([x_dim]), np.zeros([x_dim])]

    # 1st reservoir
    for i in range(u_num):
        x_next = (1-leak)*x_list[-2] + leak * np.tanh( np.matmul(W_in, u[i]) + np.matmul(W_list[0], x_list[-1] ) +  np.random.rand(x_dim))
        x_list.append(x_next)
    x_lists.append(x_list)
    
    # 2-last reservoirs
    for res in range(resvoir_num-1):
        x_list = [np.zeros([x_dim]), np.zeros([x_dim])]
        for i in range(u_num):
            x_next = (1-leak)*x_list[-2] + leak * np.tanh( np.matmul(W_trans_list[res], x_lists[res][i+2]) + np.matmul(W_list[res], x_list[-1] ) +  np.random.rand(x_dim))
            x_list.append(x_next)
        x_lists.append(x_list)
        
    states = np.array(x_lists[-1][1:]).reshape(u_num+1, x_dim)

    if verbose:
        print("Inner States: # of samples x # of dimension:", str(states.shape))
        print("Input States: # of samples x # of dimension:", str(u.shape))

        print()
        print("------------------------------------------------Concatenate data and Y sequence data------------------------------------------------")
        print()

        # Making Concatenated data

    X = np.concatenate([states[:-1,:], u], axis=1)
    X = X[:-1, :]

    
    
    
    
    # Faking Target sequence
    Y = target_Y


    if verbose:
        print("Inner + Input States: # of samples x # of dimension:", str(X.shape))
        print("Targeted fitting sequence: # of samples x # of dimension:", str(Y.shape))

        print()
        print("--------------------------------------------------Splitting Data to 3-------------------------------------------------------------")
        print()

    # Split into 3 trunks, usless trunk, regressing trunk, predicting trunk
    useless_X = X[:forget, :]
    useless_Y = Y[:forget, :]

    regress_X = X[forget:-cutout, :]
    regress_Y = Y[forget:-cutout, :]
    
    train_size = int(regress_X.shape[0]*(1-val_cut))
    
    train_X = np.split(regress_X, [train_size, regress_X.shape[0]+1])[0]
    train_Y = np.split(regress_Y, [train_size, regress_X.shape[0]+1])[0]
       
    val_X = np.split(regress_X, [train_size, regress_X.shape[0]+1])[1]
    val_Y = np.split(regress_Y, [train_size, regress_X.shape[0]+1])[1]
        
    predict_X = X[-cutout:, :]
    predict_Y = Y[-cutout:, :]

    if verbose:
        print("useless_X: # of samples x # of dimension:", str(useless_X.shape))
        print("useless_Y: # of samples x # of dimension:", str(useless_Y.shape))
        print("regress_X: # of samples x # of dimension:", str(regress_X.shape))
        print("regress_Y: # of samples x # of dimension:", str(regress_Y.shape))
        print("train_X: # of samples x # of dimension:", str(train_X.shape))
        print("train_Y: # of samples x # of dimension:", str(train_Y.shape))
        print("val_X: # of samples x # of dimension:", str(val_X.shape))
        print("val_Y: # of samples x # of dimension:", str(val_Y.shape))
        print("predict_X: # of samples x # of dimension:", str(predict_X.shape))
        print("predict_Y: # of samples x # of dimension:", str(predict_Y.shape))

        print()
        print("---------------------------------------------------Conducting Regression----------------------------------------------------------")
        print()

    alpha = cv_start
    mse = {}
    while alpha <= cv_end:

        # Conducting linear regression
        reg = Ridge(alpha).fit(train_X, train_Y)

        # Making prediction
        valhat_Y = reg.predict(val_X)
        alpha += cv_step

        loss = np.mean(np.multiply(   (val_Y - valhat_Y), (val_Y - valhat_Y)))
        mse[alpha] = loss


    best_mse = min(list(mse.values()))
    best_alpha = list(mse.keys())[list(mse.values()).index(best_mse)]


    # using best regression again
    reg = Ridge(best_alpha).fit(regress_X, regress_Y)

    # Making prediction
    predhat_Y = reg.predict(predict_X)
    
    # showing training error, 
    regrpred_Y = reg.predict(regress_X)
    train_mse = np.mean(np.multiply(   (regrpred_Y - regress_Y), (regrpred_Y - regress_Y)))
    pred_mse = np.mean(np.multiply(   (predhat_Y - predict_Y), (predhat_Y - predict_Y)))
    
    print("model ridge coefficient:", best_alpha)
    print("Model training mse:", train_mse)
    print("Model validation mse:", best_mse)
    print("Model prediction mse:", pred_mse)
    print("Model prediction average error", math.sqrt(pred_mse))

    if verbose:    
        print("regress coefficient length:", len(reg.coef_[0].tolist()))
        print("first 5 coefficient of model:", reg.coef_[0, :5])
        print("Predicted length equal to target length:", predhat_Y.shape==predict_Y.shape)

        print()
        print("-----------------------------------------------Producing Graphic Visualization----------------------------------------------------")
        print()


    # Producing Graphic Visualization

    # 后面100个，真实值与预测的
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_title('Predict and Groud Truth'.format("seaborn"), color='C0')   

    ax.plot([j for j in range(cutout)], [predict_Y[j] for j in range(predict_Y.shape[0])])
    ax.plot([j for j in range(cutout)], [predhat_Y[j] for j in range(predhat_Y.shape[0])], "--")


    # 所有的，真实值与预测的
    hat_Y = reg.predict(X)

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_title('Predict and Groud Truth everything'.format("seaborn"), color='C1')   

    ax.plot([j for j in range(forget, u_num-1)], [Y[j] for j in range(forget, u_num-1)], ":", alpha = 0.7)
    ax.plot([j for j in range(forget, u_num-1)], [hat_Y[j] for j in range(forget, u_num-1)], "red", linewidth=1)
    ax.axvline(x=forget, ls = "--", c = "yellow")
    ax.axvline(x=u_num - cutout, ls = "--", c = "yellow")

    
    # Different in prediction
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_title('Predict and Groud Truth everything'.format("seaborn"), color='C1')   

    ax.plot([j for j in range(forget, u_num-1)], [Y[j]-hat_Y[j] for j in range(forget, u_num-1)], "black", linewidth=1)
    ax.axvline(x=forget, ls = "--", c = "yellow")
    ax.axvline(x=u_num - cutout, ls = "--", c = "yellow")


    # all predictor signals
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_title('all predictor signals'.format("seaborn"), color='C1')   
    for i in range(1):
        ax.plot([j for j in range(1, u_num-1)], [X[j, i] for j in range(1, u_num-1)], linewidth=0.5)
    for i in range(-1,-6, -1):
        ax.plot([j for j in range(1, u_num-1)], [X[j, i] for j in range(1, u_num-1)], linewidth=0.5)
    ax.axvline(x=forget, ls = "--", c = "yellow")
    ax.axvline(x=u_num - cutout, ls = "--", c = "yellow")