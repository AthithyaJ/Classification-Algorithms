import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
plt.interactive(False)

# Importing the .csv file from the local location as Dataframe.
#  This line varies fom machine to machine. Correct path must be given
data = pd.read_csv(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignment 2\data.csv', header = None)
data.columns = ['Wife', 'Husband', 'Rvalue']
data['Pvalue'] = 0
data['Wife'] = pd.to_numeric(data['Wife'])
data['Husband'] = pd.to_numeric(data['Husband'])
data['Wife'] = 0.1 * data['Wife']
data['Husband'] = 0.1 * data['Husband']
# Shuffle and Split the data according to the requirement
data = shuffle(data)
df = pd.DataFrame(data.iloc[:300, :])
df_test = pd.DataFrame(data.iloc[300:, :])
sensitivity = 0
specificity=0
ppv=0
npv=0
acc=[]
Hit_rate = 0
Hit_rate2 = 0
TPositive = 0
TNegative = 0
FPositive = 0
FNegative = 0
# Initialize weight values
w=[1,1,1]
eta = 0.1
e = 0
epoch = 50

# Function to calculate the weight change based on the error value
def weight_val(w,r,e):
    w[0] = w[0] + eta * e
    w[1] = w[1] + eta * e *(df.loc[r, 'Wife'])
    w[2] = w[2] + eta * e *(df.loc[r, 'Husband'])

# Function to calculate the Value of the perceptron for  given data input
def perce_line(x1, x2, w):
    rvalue = x1 * w[1] + x2 * w[2] + w[0]
    return rvalue

for l in range(10):
 # Weight Training done on the refernce dataframe
 for a in range(epoch):
     # Iteration through every row in the dataframe using index and row object
     for index, rows in df.iterrows():
         x1=df.loc[index, 'Wife']
         x2=df.loc[index, 'Husband']
         val = perce_line(x1, x2, w)
         if val > 0:
             v = 1
         else:
            v = 0
         err = df.loc[index, 'Rvalue'] - v
         weight_val(w, index, err)
 TPositive = 0
 TNegative = 0
 FPositive = 0
 FNegative = 0
 # Weights tested on the Test Dataframe
 for i, row in df_test.iterrows():
       x1 = df_test.loc[i, 'Wife']
       x2 = df_test.loc[i, 'Husband']
       rvalue = perce_line(x1, x2, w)
       if rvalue <= 0 :
           df_test.loc[i, 'Pvalue'] = 0
       else:
           df_test.loc[i, 'Pvalue'] = 1
       # Calculation of TPositive, TNegative, FPositive, FNegative on the train data
       if df_test.loc[i, 'Pvalue'] == df_test.loc[i, 'Rvalue'] and df_test.loc[i, 'Pvalue'] == 1:
           TPositive += 1
       elif df_test.loc[i, 'Pvalue'] == df_test.loc[i, 'Rvalue'] and df_test.loc[i, 'Pvalue'] == 0:
           TNegative += 1
       elif df_test.loc[i, 'Pvalue'] != df_test.loc[i, 'Rvalue'] and df_test.loc[i, 'Pvalue'] == 1:
           FPositive += 1
       elif df_test.loc[i, 'Pvalue'] != df_test.loc[i, 'Rvalue'] and df_test.loc[i, 'Pvalue'] == 0:
           FNegative += 1
 # Calculation of Parameters on the Train Data
 sensitivity += TPositive / (TPositive + FNegative)
 specificity += TNegative / (FPositive + TNegative)
 ppv += TPositive / (TPositive + FPositive)
 npv += TNegative / (TNegative + FNegative)
 Hit_rate = (TNegative + TPositive) / 99
 Hit_rate2 += (TNegative + TPositive) / 99
 acc.append(1-Hit_rate)


# Calculation of TPositive, TNegative, FPositive, FNegative on the test data
for ind1, r in df.iterrows():
    if df.loc[ind1, 'Pvalue'] == df.loc[ind1, 'Rvalue'] and df.loc[ind1, 'Pvalue'] == 1:
        TPositive += 1
    elif df.loc[ind1, 'Pvalue'] == df.loc[ind1, 'Rvalue'] and df.loc[ind1, 'Pvalue'] == 0:
        TNegative += 1
    elif df.loc[ind1, 'Pvalue'] != df.loc[ind1, 'Rvalue'] and df.loc[ind1, 'Pvalue'] == 1:
        FPositive += 1
    elif df.loc[ind1, 'Pvalue'] != df.loc[ind1, 'Rvalue'] and df.loc[ind1, 'Pvalue'] == 0:
        FNegative += 1

# Calculation of Parameters on the Test Data
sensitivity2 = TPositive / (TPositive + FNegative)
specificity2 = TNegative / (FPositive + TNegative)
ppv2 = TPositive / (TPositive + FPositive)
npv2 = TNegative / (TNegative + FNegative)
Hit_train = (TNegative + TPositive) / 300
names = ["Hit Rate", "Sensitivity", "Specificity", "PPV", "NPV"]
measures2 = [Hit_train, sensitivity2, specificity2, ppv2, npv2]
measures = [Hit_rate2, sensitivity, specificity, ppv, npv]
# PLot of error vs epoch
plt.plot(range(0,500,50), acc)
plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.title('Epoch vs Error Rate')
plt.show()

# Bar graph for different Metrics

ind = range(len(measures))
plt.bar(ind, measures, 0.3)
plt.xlabel("Metrics")
plt.ylabel("Metric Value")
plt.xticks(ind, names, fontsize=6)
plt.title("Bar Graph for Test Data")
plt.show()

ind2 = range(len(measures2))
plt.bar(ind, measures2, 0.3)
plt.xlabel("Metrics")
plt.ylabel("Metrics Value")
plt.xticks(ind, names, fontsize=6)
plt.title("Bar Graph for Train Data")
plt.show()

# Decision Boundary calculation
X = pd.DataFrame(df_test.iloc[:, :2])
Y = pd.DataFrame(df_test.iloc[:, 2])

# Initialization of values
h=0.001
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Calculation of mesh grid parameters and mesh grid creation
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z= np.zeros_like(xx)

# Segment for Classification of every point in the mesh grid as 1 or 0
x_mat = np.arange(x_min, x_max, h)
y_mat = np.arange(y_min, y_max, h)
for i in range(len(x_mat)):
    for j in range(len(y_mat)):
        pval = perce_line(xx[j,i], yy[j,i], w)
        if pval <= 0 :
            Z[j,i]=1
        else:
            Z[j,i]=0
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].


# Put the result into a color plot as Decision Boudary

plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y.iloc[:, 0], cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Wife Salary')
plt.ylabel('Husband Salary')
plt.title('Perceptron')
plt.show()