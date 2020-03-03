import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
df = pd.read_csv(
    'Datasets/concrete_strength_raw.csv',
    header=0)
df.columns = [
    'Cement','Blast','FlyAsh','Water','Super','Coarse','Fine','Age','Concrete'
]
# df = df.sample(frac=0.1, random_state=1)

train_cols = df.columns[0:-1]
label = df.columns[-1]
X = df[train_cols]
y = df[label]
#X,y = datasets.load_boston(return_X_y=True) 
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
from interpret import show
from interpret.data import ClassHistogram

hist = ClassHistogram().explain_data(X_train, y_train, name = 'Train Data')
show(hist)
print(type(hist))
from interpret.glassbox import ExplainableBoostingRegressor, LogisticRegression, ClassificationTree, DecisionListClassifier

ebm = ExplainableBoostingRegressor(random_state=seed)
ebm.fit(X_train, y_train)   #Works on dataframes and numpy arrays
ebm_global = ebm.explain_global(name='EBM')
for i in range(7):
	ebm_global.visualize(i).write_html('Concrete_Strength/CS_'+df.columns[i]+'.html')	

preds = ebm.predict(X_test)
#for i in range(len(preds)):
	#print(preds[i],y_test[i])
print(preds)
print(y_test)

#ebm_global.visualize(0).write_html('zero.html')
#ebm_local = ebm.explain_local(X_test, y_test)
#ebm_local.visualize(0).write_html("local_zero.html")
