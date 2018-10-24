import pandas as pd
from sklearn.utils import shuffle
import statistics as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
data = pd.read_csv(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignment 2\data.csv',header=None)
data.columns = ['Wife', 'Husband', 'Rvalue']
data['Wife'] = pd.to_numeric(data['Wife'])
data['Husband'] = pd.to_numeric(data['Husband'])
sensitivity = []
specificity = []
Hit_rate = []
PPV = []
NPV = []
mean_hit_rate=[]
st_hit_rate = []
mean_npv=[]
st_npv=[]
mean_ppv=[]
st_ppv=[]
mean_sens=[]
st_sen=[]
mean_spec=[]
st_spec=[]

# Function to classify a data point based on the distances of the neighbours.
# The b values is the deciding parameter here.
def neighbor_find(swife, shusband, b):
    for i, row in df_ref.iterrows():
        w1 = row['Wife']
        h1 = row['Husband']
        d = pow(((w1 - swife) ** 2 + (h1 - shusband) ** 2), 0.5)
        if d != 0:
            distance[i] = d
    n = sorted(distance.values())
    nei = {}
    q = {v: k for k, v in distance.items()}
    for i in range(b):
        ele = n[i]
        nei[ele] = q[ele]
    val = []
    for i in nei.keys():
        val.append(nei[i])
    s = 0
    for ind, d2 in df_ref.iterrows():
        if ind in val:
            s += d2['Rvalue']
    return s


for b in range(2,7):
    for h in range(10):
        TPositive = 0
        TNegative = 0
        FPositive = 0
        FNegative = 0
        # Shuffle and Split the data according to the requirement
        data = shuffle(data)
        df_ref = pd.DataFrame(data.iloc[:320, :])
        df_test = pd.DataFrame(data.iloc[320:, :])
        for i, r in df_test.iterrows():
            df_test.loc[i, 'Pvalue'] = 0
        # Iterating through every row in the Dataframe
        for index, rows in df_test.iterrows():
            distance = {}
            swife = rows['Wife']
            shusband = rows['Husband']
            s = neighbor_find(swife, shusband, b)
            if s > float(b > 2):
                rows['Pvalue'] = 1
            else:
                rows['Pvalue'] = 0

            # Calculation of TPositive, TNegative, FPositive, FNegative
            if rows['Pvalue'] == rows['Rvalue'] and rows['Pvalue'] == 1:
                TPositive += 1
            elif rows['Pvalue'] == rows['Rvalue'] and rows['Pvalue'] == 0:
                TNegative += 1
            elif rows['Pvalue'] != rows['Rvalue'] and rows['Pvalue'] == 1:
                FPositive += 1
            elif rows['Pvalue'] != rows['Rvalue'] and rows['Pvalue'] == 0:
                FNegative += 1
        # Calculation of the Parameter
        Hit_rate.append((TNegative + TPositive) / 79)
        sensitivity.append(TPositive / (TPositive + FNegative))
        specificity.append(TNegative / (FPositive + TNegative))
        PPV.append(TPositive / (TPositive + FPositive))
        NPV.append(TNegative / (TNegative + FNegative))
    mean_hit_rate.append(st.mean(Hit_rate))
    st_hit_rate.append(st.stdev(Hit_rate))
    mean_sens.append(st.mean(sensitivity))
    st_sen.append(st.stdev(sensitivity))
    mean_spec.append(st.mean(specificity))
    st_spec.append(st.stdev(specificity))
    mean_ppv.append(st.mean(PPV))
    st_ppv.append(st.stdev(PPV))
    mean_npv.append(st.mean(NPV))
    st_npv.append(st.stdev(NPV))
# best neighbopor value is found
Ac_hit_rate = max(Hit_rate)
x = Hit_rate.index(Ac_hit_rate)
x=x+2

# Corresponding error bars of the best neighbor is plotted
plt.figure()
plt.errorbar(range(2, 7, 1), mean_hit_rate,st_hit_rate, linestlye = None)
plt.xlabel("Number of neighbours")
plt.ylabel("Mean Hit Rate")
plt.title("Mean Hit Rate vs Number of neighbours")
plt.show()

plt.figure()
plt.errorbar(range(2, 7, 1), mean_sens,st_sen, linestyle = None)
plt.xlabel("Number of neighbours")
plt.ylabel("Mean Sensitivity")
plt.title("Mean Sensitivity vs Number of neighbours")
plt.show()

plt.figure()
plt.errorbar(range(2, 7, 1), mean_spec, st_spec, linestyle = None)
plt.xlabel("Number of neighbours")
plt.ylabel("Mean Specificity")
plt.title("Mean Specificity vs Number of neighbours")
plt.show()

plt.figure()
plt.errorbar(range(2, 7, 1), mean_ppv, st_ppv, linestyle = None)
plt.xlabel("Number of neighbours")
plt.ylabel("Mean PPV")
plt.title("Mean PPV vs Number of neighbours")
plt.show()

plt.figure()
plt.errorbar(range(2, 7, 1), mean_npv, st_npv, linestyle = None)
plt.xlabel("Number of neighbours")
plt.ylabel("Mean NPV")
plt.title("Mean NPV vs Number of neighbours")
plt.show()

# Initialization of Parameters
h=0.1
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


X = pd.DataFrame(data.iloc[:, :2])
Y = pd.DataFrame(data.iloc[:, 2])

# Calculation of mesh grid parameters and mesh grid creation
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z= np.zeros_like(xx)

# Classification of every point in the mesh grid
x_mat = np.arange(x_min, x_max, h)
y_mat = np.arange(y_min, y_max, h)
for i in range(len(x_mat)):
    for j in range(len(y_mat)):
        s= neighbor_find(xx[j,i], yy[j,i], x)
        if s>=x/2:
            Z[j,i]=1
        else:
            Z[j,i]=0
# Put the result into a color plot
# Decision Boundary
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y.iloc[:, 0], cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Decision Boundary for KNN Classifier")
plt.show()

