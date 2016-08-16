# Iris Dataset

# 1. Working with Two Features and Two Targets
from sklearn import datasets

# load the iris dataset
iris = datasets.load_iris()

# start with the first two features: sepal length (cm) and sepal width (cm)
X = iris.data[:100,:2]

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

# scale X data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)


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
plt.legend(iris.target_names[:2], loc='upper right')
plt.show()

# split data to evaluate model performance
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
lr.fit(X_train, y_train)


# look at coefficients of the model
# create features list
features = ['Intercept']
for item in iris.feature_names[:2]:
    features.append(item)
    
# create coefficients array
coefficients = lr.intercept_
coefficients = np.append(coefficients, lr.coef_)

# combine features and coefficients
model = pd.DataFrame(zip(features, coefficients))
print model

X_train[0,:] # first training observation
lr.predict_proba(X_train[0,:]) # what is the predicted probability?
lr.predict(X_train[0,:]) # what is the predicted class?

X_train[13,:] # 13th training observation
lr.predict_proba(X_train[13,:]) # what is the predicted probability?
lr.predict(X_train[13,:]) # what is the predicted class?

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


y_pred = lr.predict(X_test) # apply the model to the test set
accuracy_score(y_test, y_pred) # calculate the accuracy score


y_pred2 = lr.predict(X_train)
accuracy_score(y_train, y_pred2)

# plot
import sys
sys.path.append("C:/Users/Craig/Documents/GitHub/Python-Machine-Learning")
from functions_module import plot_decision_regions

plot_decision_regions(X_train, y_train, lr)
plt.figure(figsize=(20,15))

