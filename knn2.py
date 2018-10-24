import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Importing the .csv file from the local location as Dataframe.
# This line varies fom machine to machine. Correct path must be given
data = pd.read_csv(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignment 2\data.csv',header=None)
data.columns = ['Wife', 'Husband', 'Rvalue']
data['Wife'] = pd.to_numeric(data['Wife'])
data['Husband'] = pd.to_numeric(data['Husband'])
# Shuffle and Split the data according to the requirement
data = shuffle(data)
df_ref= pd.DataFrame(data.iloc[:300, :])
df_test = pd.DataFrame(data.iloc[300:, :])
df_test['Pvalue'] = 0
sensitivity = []
specificity=[]
ppv=[]
npv=[]
acc=[]
Hit_rate = []
TPositive = 0
TNegative = 0
FPositive = 0
FNegative = 0

# Function to find the B no.of nearest neighbour to swife, shusband data point.
# Returns the added Rvalue of all b neighbours
def neighbor_find(swife, shusband,b):
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


# iterating through varying values of k
for b in range(2, 10):
        TPositive = 0
        TNegative = 0
        FPositive = 0
        FNegative = 0
        # Iteration through every row in the dataframe using index and row object
        for index, rows in df_test.iterrows():
              distance = {}
              swife = rows['Wife']
              shusband = rows['Husband']
              s=neighbor_find(swife,shusband, b)
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
        # Calculation of Parameters
        Hit_rate.append((TNegative + TPositive) / 99)
        sensitivity.append(TPositive / (TPositive + FNegative))
        specificity.append(TNegative / (FPositive + TNegative))
        ppv.append(TPositive / (TPositive + FPositive))
        npv.append(TNegative / (TNegative + FNegative))

names = ["Hit Rate", "Sensitivity", "Specificity", "PPV", "NPV"]
# The K value with maximum Hit rate is found
Ac_hit_rate = max(Hit_rate)
x = Hit_rate.index(Ac_hit_rate)
Ac_sensitivity = sensitivity[x]
Ac_specificity = specificity[x]
Ac_ppv = ppv[x]
Ac_npv = npv[x]
measures = [Ac_hit_rate, Ac_sensitivity, Ac_specificity, Ac_ppv, Ac_npv]
x +=2

# Bar graph for found K value against all metrics
ind = range(len(measures))
plt.bar(ind, measures, 0.3)
plt.xlabel('Metrics')
plt.ylabel('Metric Value')
plt.xticks(ind, names, fontsize=6)
plt.title('Bar Graph for Test Data, with k=%i' %x )
plt.show()


X = pd.DataFrame(df_test.iloc[:, :2])
Y = pd.DataFrame(df_test.iloc[:, 2])
# Initialization of values
n_neighbors = x
h=0.1
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# Calculation of mesh grid parameters and mesh grid creation
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z= np.zeros_like(xx)

# Segment for Classification of every point in the mesh grid as 1 or 0
x_mat = np.arange(x_min, x_max, h)
y_mat = np.arange(y_min, y_max, h)
for i in range(len(x_mat)):
    for j in range(len(y_mat)):
        pval = neighbor_find(xx[j,i], yy[j,i], x)
        if pval>=x/2:
            Z[j,i]=1
        else:
            Z[j,i]=0

# Put the result into a color plot
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y.iloc[:, 0], cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Wife Salary')
plt.ylabel('Husband Salary')
plt.title(" KNN Classification")
plt.show()