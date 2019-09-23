# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ann'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython


#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    validation_curve,
    learning_curve,
    train_test_split
)

figure_export_path = './figures/'

np.random.seed(0)

#%% [markdown]
# # Read in Car Data

#%%
df = pd.read_csv("../datasets/cars_data.csv")

#%% [markdown]
# # Define Methods

#%%
def plot_learning_curve(
    train_sizes,
    train_scores,
    test_scores,
    title="Learning Curve",
    y_title="Score",
    x_title="Training Examples",
    download_file_name="plot_learning_curve",
):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean + test_scores_std,
            mode="lines",
            line=dict(color="green", width=1),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean - test_scores_std,
            mode="lines",
            line=dict(color="green", width=1),
            showlegend=False,
            fill="tonexty",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean + train_scores_std,
            mode="lines",
            line=dict(color="red", width=1),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean - train_scores_std,
            mode="lines",
            line=dict(color="red", width=1),
            showlegend=False,
            fill="tonexty",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean,
            marker=dict(color="green"),
            name="Cross-Validation Score",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            marker=dict(color="red"),
            name="Training Score",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title, showgrid=True, gridwidth=1, gridcolor="LightPink"),
        yaxis=dict(title=y_title, showgrid=True, gridwidth=1, gridcolor="LightPink"),
    )
    fig.write_image(f"{figure_export_path}/{download_file_name}.png")
    fig.show()

def plot_validation_curve(
    param_range,
    train_scores,
    test_scores,
    title="Validation Curve",
    y_title="Score",
    x_title="Param Range",
    download_file_name="plot_validation_curve",

):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=test_scores_mean + test_scores_std,
            mode="lines",
            line=dict(color="green", width=1),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=test_scores_mean - test_scores_std,
            mode="lines",
            line=dict(color="green", width=1),
            showlegend=False,
            fill="tonexty",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=train_scores_mean + train_scores_std,
            mode="lines",
            line=dict(color="red", width=1),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=train_scores_mean - train_scores_std,
            mode="lines",
            line=dict(color="red", width=1),
            showlegend=False,
            fill="tonexty",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=test_scores_mean,
            marker=dict(color="green"),
            name="Cross-Validation Score",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=train_scores_mean,
            marker=dict(color="red"),
            name="Training Score",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title, showgrid=True, gridwidth=1, gridcolor="LightPink"),
        yaxis=dict(title=y_title, showgrid=True, gridwidth=1, gridcolor="LightPink"),
    )
    
    fig.write_image(f"{figure_export_path}/{download_file_name}.png")
    fig.show()

#%% [markdown]
# ## Train Test Split

#%%
car_details_df = df[
    [
        "Make",
        "Year",
        "Engine HP",
        "Engine Cylinders",
        "Driven_Wheels",
        "Number of Doors",
        "Vehicle Size",
        "Vehicle Style",
        "highway MPG",
        "city mpg",
        "MSRP",
    ]
]


#%%
print(f"Shape BEFORE dropping NA: {car_details_df.shape}")
no_na_car_details_df = car_details_df.dropna()
print(f"Shape AFTER dropping NA: {no_na_car_details_df.shape}")
print(f"Dropped {car_details_df.shape[0] - no_na_car_details_df.shape[0]} rows")

lowest_val_count = 10
print(f"Dropping rows with value counts on Make < {lowest_val_count}")
print(f"Shape BEFORE dropping lowest val counts: {car_details_df.shape}")
no_na_car_details_df = no_na_car_details_df.groupby('Make').filter(lambda x : len(x) > lowest_val_count)
print(f"Lowest value_count: {no_na_car_details_df['Make'].value_counts()[-1]}")
print(f"Shape AFTER dropping lowest val counts: {no_na_car_details_df.shape}")


dummies = pd.get_dummies(no_na_car_details_df[["Vehicle Size", "Vehicle Style", "Driven_Wheels"]])
X = no_na_car_details_df.drop(["Make"], axis=1)
y = no_na_car_details_df["Make"]

# Drop Dummies
X = X.drop(["Vehicle Size", "Vehicle Style", "Driven_Wheels"], axis=1)

# concat dumiies
X = pd.concat([X, dummies], axis=1)


#%%
X_trainset, X_testset, y_trainset, y_testset = train_test_split(
    X, y, test_size=0.30, random_state=2
)

#%% [markdown]
# # NN

#%%
from sklearn.neural_network import MLPClassifier


#%%
get_ipython().run_cell_magic('time', '', 'solver = "sgd"\nhidden_layer_sizes = (2, 10)\nactivation = "logistic"\nnn_clf = MLPClassifier(\n    solver=solver,\n    hidden_layer_sizes=hidden_layer_sizes,\n    activation=activation,\n    max_iter=1000,\n)\nnn_clf.fit(X_trainset, y_trainset)')


#%%
cross_val_score(nn_clf, X_trainset, y_trainset, cv=5)


#%%
nn_pred = nn_clf.predict(X_testset)
print(nn_clf.score(X_testset, y_testset))
print(classification_report(y_testset, nn_pred))

#%% [markdown]
# # Plot Learning Curve

#%%
train_sizes = np.linspace(round(X_trainset.shape[0]* .1), X_trainset.shape[0] * .7, num=10).astype(int)
train_sizes


#%%
train_sizes, train_scores, test_scores = learning_curve(
    nn_clf,
    X_trainset,
    y_trainset,
    cv=5,
    n_jobs=-1,
    train_sizes=train_sizes
)


#%%
plot_learning_curve(
    train_sizes,
    train_scores,
    test_scores,
    title="ANN Learning Curve (Cars)",
    x_title="Training Examples",
    y_title="Score (Accuracy %)",
    download_file_name="ann_learning_curve",
3#)

#%% [markdown]
# ## Validation Curve

#%%
param_range = np.linspace(0.0000001, .01, num=10)
ne_train_scores, ne_test_scores = validation_curve(
    nn_clf,
    X_trainset,
    y_trainset,
    "alpha",
    param_range,
    scoring="accuracy",
    cv=5,
)


#%%
plot_validation_curve(
    param_range,
    ne_train_scores,
    ne_test_scores,
    x_title='learning_rate',
    y_title='score (accuracy %)'
    "'learning_rate' validation curve",
    download_file_name="boost_alpha_vcurve",
)

#%% [markdown]
# # Final Score

#%%
nn_clf_final = MLPClassifier(
    solver=solver,
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    max_iter=1000,
    alpha=0.00001
)
nn_clf_final.fit(X_trainset, y_trainset)


#%%
nn_pred = nn_clf_final.predict(X_testset)
print(nn_clf_final.score(X_testset, y_testset))
print(classification_report(y_testset, nn_pred))


#%%



