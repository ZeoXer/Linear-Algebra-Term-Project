#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[2]:


def sigmoid(x):
    return(1 / (1 + np.exp(-x)))

def plot_sigmoid():
    x = np.linspace(-6, 6, 100)
    y = sigmoid(x)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, y)
    ax.axvline(0, color = 'black')
    ax.axhline(y = 0, ls = ':', color = 'k', alpha = 0.5)
    ax.axhline(y = 0.5, ls = ':', color = 'k', alpha = 0.5)
    ax.axhline(y = 1, ls = ':', color = 'k', alpha = 0.5)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Sigmoid function")
    plt.show()

plot_sigmoid()


# In[3]:


def plot_cross_entropy():
    epsilon = 1e-5
    h = np.linspace(epsilon, 1-epsilon) # 利用微小值 epsilon 避免 log(0) 的錯誤
    y1 = -np.log(h)
    y2 = -np.log(1 - h)
    fig, ax = plt.subplots(1, 2, figsize = (8, 4))
    ax[0].plot(h, y1)
    ax[0].set_title("$y=1$\n$-\log(\sigma(Xw))$")
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels([0, 1])
    ax[0].set_xlabel("$\sigma(Xw)$")
    ax[1].plot(h, y2)
    ax[1].set_title("$y=0$\n$-\log(1-\sigma(Xw))$")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels([0, 1])
    ax[1].set_xlabel("$\sigma(Xw)$")
    plt.tight_layout()
    plt.show()
    
plot_cross_entropy()


# In[2]:


players = pd.read_csv("https://raw.githubusercontent.com/yaojenkuo/ml-newbies/master/player_stats.csv")
cols = ["Name", "heightMeters", "weightKilograms"]
players["Name"] = players["firstName"] + " " + players["lastName"]
players[cols]


# In[3]:


X = players['heightMeters'].values.astype(float).reshape(-1, 1)
y = players['weightKilograms'].values.astype(float)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LinearRegression()
h.fit(X_train, y_train)

# Get the regression line's intercept and slope
print("Intercept: {}".format(h.intercept_))
print("Slope: {}".format(h.coef_[0]))
print()

# The test result
y_pred = h.predict(X_valid)
print("Test result:")
print(y_pred[:10])


# In[4]:


X1 = np.linspace(X.min() - 0.1, X.max() + 0.1).reshape(-1, 1)
y_hat = h.predict(X1)
fig, ax = plt.subplots()
ax.scatter(X_train.ravel(), y_train, label="train data")
ax.scatter(X_valid.ravel(), y_valid, label="test data")
ax.plot(X1.ravel(), y_hat, c="black", label="regression line")
ax.legend()
plt.show()


# In[5]:


# Simplify to 2 position
pos_dict = {0: "G", 1: "F"} # 0 stands for guard and 1 stands for forward
pos = players['pos'].values
pos_binary = np.array([0 if p[0] == "G" else 1 for p in pos])
np.unique(pos_binary)

X = players[['apg', 'rpg']].values.astype(float)
y = pos_binary
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LogisticRegression(C=1e06, solver='liblinear')
h.fit(X_train, y_train)
print("Intercept: {}".format(h.intercept_[0]))
print("Slope: {}".format(h.coef_))


# In[6]:


p_hat = h.predict_proba(X_valid)
print("     prob(0)        prob(1)")
print(p_hat[:10, :])
y_pred = np.argmax(p_hat, axis=1) # choose 0/1 that habe the biggest probibility
print("Prediction result:")
print(y_pred[:10])
y_pred_label = [pos_dict[i] for i in y_pred]
print(y_pred_label[:10])


# In[7]:


resolution = 50
apg = players['apg'].values.astype(float)
rpg = players['rpg'].values.astype(float)
X1 = np.linspace(apg.min() - 0.5, apg.max() + 0.5, num=resolution).reshape(-1, 1)
X2 = np.linspace(rpg.min() - 0.5, rpg.max() + 0.5, num=resolution).reshape(-1, 1)
APG, RPG = np.meshgrid(X1, X2)

def plot_contour_filled(XX, YY, resolution=50):
    PROBA = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            xx_ij = XX[i, j]
            yy_ij = YY[i, j]
            X_plot = np.array([xx_ij, yy_ij]).reshape(1, -1)
            z = h.predict_proba(X_plot)[0, 1]
            PROBA[i, j] = z
    fig, ax = plt.subplots()
    CS = ax.contourf(XX, YY, PROBA, cmap="RdBu")
    ax.set_title("Probability of being predicted as a forward")
    ax.set_xlabel("Assists per game")
    ax.set_ylabel("Rebounds per game")
    fig.colorbar(CS, ax=ax)
    plt.show()
    
plot_contour_filled(APG, RPG)


# In[8]:


def plot_decision_boundary(XX, YY, x, y, target_vector, pos_dict, h, resolution=50):
    Y_hat = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            xx_ij = XX[i, j]
            yy_ij = YY[i, j]
            X_plot = np.array([xx_ij, yy_ij]).reshape(1, -1)
            z = h.predict(X_plot)
            Y_hat[i, j] = z
    fig, ax = plt.subplots()
    CS = ax.contourf(XX, YY, Y_hat, alpha=0.2, cmap="RdBu")
    colors = ['red', 'blue']
    unique_categories = np.unique(target_vector)
    for color, i in zip(colors, unique_categories):
        xi = x[target_vector == i]
        yi = y[target_vector == i]
        ax.scatter(xi, yi, c=color, edgecolor='k', label="{}".format(pos_dict[i]), alpha=0.6)
    ax.set_title("Decision boundary of Forwards vs. Guard")
    ax.set_xlabel("Assists per game")
    ax.set_ylabel("Rebounds per game")
    ax.legend()
    plt.show()
    
plot_decision_boundary(APG, RPG, apg, rpg, y, pos_dict, h)


# In[21]:


# building the dictionary of 7 positions
unique_pos = players['pos'].unique()
pos_dict = {i: p for i, p in enumerate(unique_pos)}
pos_dict_reversed = {k: v for v, k in pos_dict.items()}
pos_multiple = players['pos'].map(pos_dict_reversed)
print("Dictionary of 7 positions:")
print(pos_dict)
print()

# one versus rest
X = players[['apg', 'rpg']].values.astype(float)
y = pos_multiple
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LogisticRegression(C=1e05, solver='liblinear', multi_class='ovr') # setting the parameter multi_class
h.fit(X_train, y_train)
p_hat = h.predict_proba(X_valid)
print("Prediction result:")
print(p_hat[:5])
y_pred = np.argmax(p_hat, axis=1)
y_pred_label = [pos_dict[i] for i in y_pred]
print(y_pred_label[:5])


# In[23]:


# softmax
h = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial') 
# setting the parameter multi_class
h.fit(X_train, y_train)
p_hat = h.predict_proba(X_valid)
print("Prediction result:")
print(p_hat[:5])
y_pred = np.argmax(p_hat, axis=1)
y_pred_label = [pos_dict[i] for i in y_pred]
print(y_pred_label[:5])

