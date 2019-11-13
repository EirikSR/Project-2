from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import scipy.integrate

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.cm as cm

import scikitplot as skp
import pandas as pd
import numpy as np


def setup_Design_matrix():

    df = pd.read_excel("data.xls")    
    df = df.rename(columns={"default payment next month":"Y"})

    #Dataset is divided into numerical and categorical columns, these have to be treated seperately

    Num = ["LIMIT_BAL","AGE",       "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
    Cat =["SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]


    df[Num] = df[Num].astype(float)
    lb = LabelBinarizer()   

    #Create a column with same length as df
    X = np.ones(len(df))
    y = np.array(df['Y'])
    
    X = np.c_[X, np.array(df[Num])]

    for i in range(len(Cat)):
        X = np.c_[X, lb.fit_transform(np.array(df[Cat[i]]))]
    
    ss = StandardScaler()
    
    X = ss.fit_transform(X)

    return X, y

def MSE(comp, exac):
        mse = 0
       
        exac = exac.ravel()
        comp = comp.ravel()
       
        for x , y in zip(comp, exac):
            mse += ((x-y)**2)
       
        return mse/len(comp)

def R2(x, x_t):

    x_t = x_t.reshape(-1, 1)
    x = x.reshape(-1, 1)
    x_sum = np.sum(x)    
    
    n = len(x)
    mean = (1./n)*x_sum       
    
    return 1- (sum((x-x_t)**2))/(sum((x-mean)**2))

def CumGain(predict, y_test):
    p0 = 1-predict
    p1 = predict

    a = np.array(p0)
    a = a.reshape(-1, 1)

    b = np.array(p1)
    b = b.reshape(-1, 1)

    c = np.hstack([a,b])
    p_split = c

    
    def bestCurve(y):
        default = 0
        for i in y:
            if (i == 1):
                default += 1
        
        a = np.linspace(0, 1, default, endpoint=True)
        b = np.ones(len(y) - default)
        c = np.hstack([a,b])

        return c
    y_test = np.squeeze(y_test)
    y_best_fit = bestCurve(y_test)
    x_values = np.linspace(0, 1, len(y_test), endpoint=True)


    fig, ax = plt.subplots()
    
    skp.metrics.plot_cumulative_gain(y_test, p_split, ax=ax, title=None)

    ax.plot(x_values, y_best_fit)

    ax.legend(["Default", "Not default", "Best curve", "Baseline"], bbox_to_anchor=(1, 0.3))

    plt.title("Cumulative Gain for Credit card, Logistic Regression")
    plt.xlabel("Percentage of sample data")
    plt.ylabel("Cumulative number of target data")
    plt.show()

    R1, R2 = AreaIntegrator(x_values, y_test, y_best_fit, p0, p1)
    return R1, R2

def plotSurface(x_d, y_d, z_d):
    fig = plt.figure()
    ax = fig.gca(projection='3d')   
   
    surf = ax.plot_surface(x_d, y_d, z_d, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z - Franke')
    
    # Add a color bar which maps values to colors.
    clb = fig.colorbar(surf, shrink=0.5, aspect=5)
    clb.ax.set_title('Level')

    plt.show()

def Franke(x,y, noise = True):

    a = 0.75*np.exp(-(0.25*(9*x-2)**2)   - 0.25*((9*y-2)**2))
    b = 0.75*np.exp(-((9*x+1)**2)/49.0   - 0.10* (9*y+1))
    c = 0.50*np.exp(-(9*x-7)**2/4.0      - 0.25*((9*y-3)**2))
    d = -0.2*np.exp(-(9*x-4)**2          -       (9*y-7)**2)
    
    ret = a+b+c+d
    if noise:

        ret = ret + np.random.normal(0, 1)
        
    return ret

def AreaIntegrator(x, y, best_fit, p0, p1):
    baseline = 0.5 # x = x

    a_best_fit = scipy.integrate.simps(best_fit, x) - baseline

    x, p0_curve = skp.helpers.cumulative_gain_curve(y, p0, 0)
    a_not_default = scipy.integrate.simps(p0_curve, x) - baseline

    x, p1_curve = skp.helpers.cumulative_gain_curve(y, p1, 1)
    a_default = scipy.integrate.simps(p1_curve, x) - baseline


    r_not_default = a_not_default / a_best_fit
    r_default = a_default / a_best_fit

    return r_default, r_not_default 

def print_confusion(y_test, predict):
    acc = accuracy_score(y_test, predict)
    print 'Final model accuracy: ', acc
    print 'Sum of predicted defaults:', np.sum(predict)
    print 'Confusion matrix:'
    print confusion_matrix(y_test, predict)