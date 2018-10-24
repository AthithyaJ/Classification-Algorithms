import pandas as pd
from sklearn.utils import shuffle
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.interactive(False)

# Importing the .csv file from the local location as Dataframe.
#  This line varies fom machine to machine. Correct path must be given#

data = pd.read_csv(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignment 2\data.csv', header = None)
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
# The radius values is the deciding parameter here.
def rad_neighbors(swife, shusband, dis):
    # Iterating through every row in the Dataframe
    for i, row in df_ref.iterrows():
        w1 = row['Wife']
        h1 = row['Husband']
        d = pow(((w1 - swife) ** 2 + (h1 - shusband) ** 2), 0.5)
        if d != 0 and d <= dis:
            distance[i] = d
    s1 = []
    s2 = []
    for i2, d2 in df_ref.iterrows():
        if i2 in distance.keys():
            if df_ref.loc[i2, 'Rvalue'] == 1:
                s1.append(1)
            elif df_ref.loc[i2, 'Rvalue'] == 0:
                s2.append(0)
    return s1,s2


for dis in np.arange(0.5, 2.0, 0.1):
    for h in range(10):
        # Shuffle and Split the data according to the requirement
        data = shuffle(data)
        df_ref = pd.DataFrame(data.iloc[:320, :])
        df_test = pd.DataFrame(data.iloc[320:, :])

        for i, r1 in df_test.iterrows():
            df_test.loc[i, 'Pvalue'] = 0
        TPositive = 0
        TNegative = 0
        FPositive = 0
        FNegative = 0
        # Iterating through every row in the Dataframe
        for index, rows in df_test.iterrows():
            distance = {}
            swife = rows['Wife']
            shusband = rows['Husband']
            s1, s2 = rad_neighbors(swife, shusband, dis)
            if len(s1) >= len(s2):
                    df_test.loc[index, 'Pvalue'] = 1
            elif len(s1) < len(s2):
                    df_test.loc[index, 'Pvalue'] = 0

            # Calculation of TPositive, TNegative, FPositive, FNegative
            if df_test.loc[index, 'Pvalue'] == df_test.loc[index, 'Rvalue'] and df_test.loc[index, 'Pvalue'] == 1:
                TPositive += 1
            elif df_test.loc[index, 'Pvalue'] == df_test.loc[index, 'Rvalue'] and df_test.loc[index, 'Pvalue'] == 0:
                TNegative += 1
            elif df_test.loc[index, 'Pvalue'] != df_test.loc[index, 'Rvalue'] and df_test.loc[index, 'Pvalue'] == 1:
                FPositive += 1
            elif df_test.loc[index, 'Pvalue'] != df_test.loc[index, 'Rvalue'] and df_test.loc[index, 'Pvalue'] == 0:
                FNegative += 1
        # Calculation of the Parameters
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
# radius corresponding to the best hit rate is found
m = max(mean_hit_rate)
ju= 0.5
for item in mean_hit_rate:
    if item == m:
        break
    else:
        ju +=0.1
r=ju

# Corresponding error bar graphs of the best radius value is plotted
plt.figure()
plt.errorbar(np.arange(0.5, 2.0, .1), mean_hit_rate, st_hit_rate, linestyle = None)
plt.xlabel("Radius Values")
plt.ylabel("Mean Hit Rate")
plt.title("Mean Hit Rate vs Radius Values")
plt.show()

plt.figure()
plt.errorbar(np.arange(0.5, 2.0, .1), mean_sens, st_sen, linestyle = None)
plt.xlabel("Radius Values")
plt.ylabel("Mean Sensitivity")
plt.title("Mean Sensitivity vs Radius Values")
plt.show()

plt.figure()
plt.errorbar(np.arange(0.5, 2.0, .1), mean_spec, st_spec, linestyle = None)
plt.xlabel("Radius Values")
plt.ylabel("Mean Specificity")
plt.title("Mean Specificity vs Radius Values")
plt.show()

plt.figure()
plt.errorbar(np.arange(0.5, 2.0, .1), mean_ppv, st_ppv, linestyle = None)
plt.xlabel("Radius Values")
plt.ylabel("Mean PPV")
plt.title("Mean PPV vs Radius Values")
plt.show()

plt.figure()
plt.errorbar(np.arange(0.5, 2.0, .1), mean_npv, st_npv, linestyle = None)
plt.xlabel("Radius Values")
plt.ylabel("Mean NPV")
plt.title("Mean NPV vs Radius Values")
plt.show()

# Initialization of Parameters
h=0.1
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Calculation of mesh grid parameters and mesh grid creation
X = pd.DataFrame(data.iloc[:, :2])
Y = pd.DataFrame(data.iloc[:, 2])
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z= np.zeros_like(xx)

# Function to classify every point in  the mesh grid
x_mat = np.arange(x_min, x_max, h)
y_mat = np.arange(y_min, y_max, h)
for i in range(len(x_mat)):
    for j in range(len(y_mat)):
        s1, s2 = rad_neighbors(xx[j,i], yy[j,i], r)
        if len(s1)>= len(s2):
            Z[j,i]=1
        else:
            Z[j,i]=0
# Put the result into a color plot
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y.iloc[:, 0], cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Decision Boundary for Radius-based Classifier for Radius %i" %r)
plt.show()