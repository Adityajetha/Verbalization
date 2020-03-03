import pandas as pd
import scipy.stats as st

#sort in order of pcc
def sort_pcc(objs):
  return sorted(objs, key = lambda i: i['pcc'])

#sort in order of impact
def sort_impact(objs):
  return sorted(objs, key = lambda i: i['impact'])

#sort alphabetically 
def sort_features(objs):
  return sorted(objs, key = lambda i: i['feature'])

#location of raw data
location="../Datasets/boston_housing_raw.csv"
#loaction of output of chunk.py or aschunk.py
fl="../Boston_Housing/out_59.0_"
#column number of the first feature
start=1
#column number of the outcome feature
label=-1

df = pd.read_csv(location,header=0)
features=df.columns[start:label]

objs=[]
for feature in features:
  #chas not a feature in ashraf's 59.0 data
  if(feature=="chas"):
    continue

  file = pd.read_csv(fl+feature+".csv", header=0)
  no_chunks=int(sum(file["is_critical"])+1)
  start=0
  end=0
  chunks_inds=[]
  #finding all the datapoints in a chunk, getting the start and the end indices of a chunk
  for row in range(len(file[feature])):
    if(int(file["is_critical"][row])==1):
      chunks_inds.append((start,end))
      start=end+1
      end+=1
    else:
      end+=1
  chunks_inds.append((start,end-1))

  #storing all the information of a chunk in a list of dictionaries
  for chunk in chunks_inds:
    x_min=file[feature][chunk[0]]
    x_max=file[feature][chunk[-1]]
    y_min=file[file.columns[1]][chunk[0]]
    y_max=file[file.columns[1]][chunk[-1]]
    impact=abs(y_max-y_min)
    sign=y_max-y_min
    X=[]
    Y=[]
    for ind in range(chunk[0],chunk[1]+1):
      X.append(file[feature][ind])
      Y.append(file[file.columns[1]][ind])
    score=st.pearsonr(X,Y)
    mean=sum(Y)/len(Y)
    obj={}
    obj["y_min"]=y_min
    obj["y_max"]=y_max
    obj["x_max"]=x_max
    obj["x_min"]=x_min
    obj["mean"]=mean
    obj["feature"]=feature
    obj["pcc"]=score[0]
    obj["impact"]=impact
    obj["sign"]=sign
    objs.append(obj)

pd.DataFrame(sort_features(objs),columns=["feature","x_min","x_max","y_min","y_max","mean","impact","sign","pcc"]).to_csv("../Boston_Housing/sorted_f_59.0.csv",index=False)


