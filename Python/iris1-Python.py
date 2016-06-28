# Iris Dataset

# 1. Working with Two Features and Two Targets
from sklearn import datasets
import pandas as pd
import numpy as np

# load the iris dataset
iris = datasets.load_iris()

# start with the first two features: sepal length (cm) and sepal width (cm)
X = iris.data[:100,:2]

# what type of data is X? numpy array  
type(X)

# what is the shape of X? 100 rows by 2 columns
X.shape

# save the target values as y
y = iris.target[:100]

# plot the X data
import matplotlib.pyplot as plt

# Define bounds on the X and Y axes
X_min, X_max = X[:,0].min()-.5, X[:,0].max()+.5
y_min, y_max = X[:,1].min()-.5, X[:,1].max()+.5

for target in set(y):
    x = [X[i,0] for i in range(len(y)) if y[i]==target]
    z = [X[i,1] for i in range(len(y)) if y[i]==target]
    plt.scatter(x,z,color=['red','blue'][target], label=iris.target_names[:2][target])

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.xlim(X_min,X_max)
plt.ylim(y_min,y_max)
plt.title('Scatter Plot of Sepal Length vs. Sepal Width')
plt.legend(iris.target_names[:2], loc='lower right')
plt.show()





# combine the dataset for visualization
combined = np.column_stack((X,y))

import seaborn as sns
df = pd.DataFrame(data = combined,
                  index = range(X.shape[0]),
                  columns = ['Sepal Length','Sepal Width', 'Iris Variety'])

               

sns.lmplot('Sepal Length',
           'Sepal Width',
           data = df,
           fit_reg = False,
           hue='Iris Variety')

plt.title('Scatter Plot of Sepal Length vs. Sepal Width (seaborn)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
