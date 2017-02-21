#https://www.kaggle.com/c/house-prices-advanced-regression-techniques
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import *
from enum import Enum


trainfile = pd.read_csv("train.csv",index_col=0)
testfile = pd.read_csv("test.csv",index_col=0)
#print trainfile["MSSubClass"][1]
#print trainfile.head()
# print "---------------------------------------------"
# print testfile.head()

# print trainfile.info()
# print "---------------------------------------------"
# print testfile.info()


train_data = trainfile.as_matrix()

#print train_data[0]
#print train_data

train_features = train_data[:,:-1]
train_labels = train_data[:,-1:]

mat_price = np.array(trainfile["SalePrice"])

# print train_features[:,1]
#print train_labels
'''
reg = LinearRegression()
reg.fit(train_features,train_labels)
coefs = reg.coef_
print coefs
'''

#print trainfile.info()

# print trainfile["MSSubClass"].head()

# MSSubClass_dummies = pd.get_dummies(trainfile["MSSubClass"])
# print MSSubClass_dummies

'''
		20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES
'''

'''
def MSSC(x):
	arr = [20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190]
	j=0
	for i in arr:
		if x==i:
			return j
		j+=1


mssc = np.zeros(16)
msscsum = np.zeros(16)
i=0
for x in train_features[:,0:1]:
	msscsum[MSSC(x[0])]+=train_labels[i,0]
	mssc[MSSC(x[0])]+=1
	i+=1
average =  msscsum/mssc
# print average.shape
# print train_features[:,0:1].shape
plt.scatter(['20','30','40','45','50','60','70','75','80','85','90','120','150','160','180','190'],average)
#plt.show()
plt.savefig("MSSubClass.png")
'''

'''
#MSZoning
#Can be removed
mszoningmat = train_features[:,1]
ct = np.zeros(5)

sumval = np.zeros(5)
i=0
for x in mszoningmat:
	ct[getval(x)]+=1
	sumval[getval(x)]+=train_labels[i,0]
	i+=1
avg = sumval/ct
# for i in ct:
# 	print i
# for i in avg:
# 	print i
plt.plot(np.array([1,2,3,4,5]),avg)
plt.savefig("MSZoning_average")
MSZoning_train_dummies = pd.get_dummies(trainfile['MSZoning'])
MSZoning_test_dummies = pd.get_dummies(testfile['MSZoning'])
#print MSZoning_train_dummies
for i,s in trainfile.iterrows():
		trainfile.loc[i,"MSZoning"] = getval(s["MSZoning"])
for i,s in testfile.iterrows():
		testfile.loc[i,"MSZoning"] = getval(s["MSZoning"])
# print trainfile.head()
# print testfile.head()

#Not able t join it.
# trainfile.join(MSZoning_train_dummies)	
# testfile.join(MSZoning_test_dummies)
# trainfile.drop(['MSZoning'],inplace=True,axis=1)
# testfile.drop(['MSZoning'],inplace=True,axis=1)

# print trainfile.head()
'''
def getval_MSZoning(x):
	arr = ['C (all)','FV','RH','RL','RM']
	j=0
	for i in arr:
		if x==i:
			return j
		j+=1
for i,s in trainfile.iterrows():
		trainfile.loc[i,"MSZoning"] = getval_MSZoning(s["MSZoning"])
for i,s in testfile.iterrows():
		testfile.loc[i,"MSZoning"] = getval_MSZoning(s["MSZoning"])
testfile["MSZoning"].fillna(3,inplace=True)
testfile["MSZoning"]=testfile["MSZoning"].astype(int)
trainfile["MSZoning"]=trainfile["MSZoning"].astype(int)
# print trainfile.head()
# print testfile.head()

#LotFrontage
#contains around 200 NaN values
#print testfile["LotFrontage"].head()
train_lot_mean = trainfile["LotFrontage"].mean()
train_lot_std = trainfile["LotFrontage"].std()
train_lot_nanct = trainfile["LotFrontage"].isnull().sum()

# print train_lot_mean,train_lot_std,train_lot_nanct

test_lot_mean = testfile["LotFrontage"].mean()
test_lot_std = testfile["LotFrontage"].std()
test_lot_nanct = testfile["LotFrontage"].isnull().sum()

rand_train = np.random.randint(train_lot_mean - train_lot_std,train_lot_mean + train_lot_std, size = train_lot_nanct)
rand_test = np.random.randint(test_lot_mean - test_lot_std,test_lot_mean + test_lot_std, size = test_lot_nanct)

trainfile["LotFrontage"][np.isnan(trainfile["LotFrontage"])]=rand_train
testfile["LotFrontage"][np.isnan(testfile["LotFrontage"])]=rand_test

# print trainfile.info()
# print testfile.info()

'''
#Plot for LotFrontage
# plt.scatter(trainfile["LotFrontage"],trainfile["SalePrice"])
# plt.show()
# MIN_SALES_PRICE = np.amin(np.array(trainfile["SalePrice"]))
# MAX_SALES_PRICE = np.amax(np.array(trainfile["SalePrice"]))

mat_lot = np.array(trainfile["LotFrontage"])
mat_price = np.array(trainfile["SalePrice"])
MAXVAL = np.amax(mat_lot)
MINVAL = np.amin(mat_lot)

ct = np.zeros(10)
sm = np.zeros(10)

factor = math.ceil((MAXVAL-MINVAL)/10.)
i=0
print ct[0]
for x in mat_lot:
	ct[(x-MINVAL)/factor]+=1
	sm[(x-MINVAL)/factor]+=mat_price[i]
	i+=1

# plt.hist(np.array(trainfile["LotFrontage"]),5,(np.array(trainfile["SalePrice"])-MIN_SALES_PRICE)/((MAX_SALES_PRICE- MIN_SALES_PRICE)*1000.))
plt.plot(np.array([0,1,2,3,4,5,6,7,8,9]),sm/ct)
plt.savefig("LotFrontage.png")
'''
'''
#street
ct_street = np.zeros(2)
sum_street = np.zeros(2)
i=0
mat_price = np.array(trainfile["SalePrice"])
for x in trainfile["Street"]:
	if x=="Pave":
		ct_street[0]+=1
		sum_street[0]+=mat_price[i]
	else:
		ct_street[1]+=1
		sum_street[1]+=mat_price[i]
	i+=1
print ct_street
print sum_street/ct_street
'''
#Dropping Street as it has no significance and Alley as it has not sufficient data
trainfile.drop(["Street","Alley"],axis=1,inplace=True)
testfile.drop(["Street","Alley"],axis=1,inplace=True)

# print trainfile.info()
# print testfile.info()
'''
#LotShape
ct_lotshape = np.zeros(4)
sum_lotshape = np.zeros(4)
i=0
for x in trainfile["LotShape"]:
	if x=="Reg":
		ct_lotshape[0]+=1
		sum_lotshape[0]+=mat_price[i]
	elif x=="IR1":
		ct_lotshape[1]+=1
		sum_lotshape[1]+=mat_price[i]
	elif x=="IR2":
		ct_lotshape[2]+=1
		sum_lotshape[2]+=mat_price[i]
	else:
		ct_lotshape[3]+=1
		sum_lotshape[3]+=mat_price[i]
	i+=1

print ct_lotshape
print sum_lotshape/ct_lotshape
'''
def getval_LotShape(x):
	if x=="Reg":
		return 0
	elif x=="IR1":
		return 1
	elif x=="IR2":
		return 2
	else:
		return 3

for i,s in trainfile.iterrows():
		trainfile.loc[i,"LotShape"] = getval_LotShape(s["LotShape"])
for i,s in testfile.iterrows():
		testfile.loc[i,"LotShape"] = getval_LotShape(s["LotShape"])
testfile["LotShape"]=testfile["LotShape"].astype(int)
trainfile["LotShape"]=trainfile["LotShape"].astype(int)
# print trainfile.info()
# print testfile.info()

#General function used again and again
def average_counts(ntype,lists,varname):
	ct = np.zeros(ntype)
	sums = np.zeros(ntype)
	i=0
	for x in trainfile[varname][pd.isnull(trainfile[varname])==False]:
		#print x
		j=0
		for y in lists:
			if y==x:
				break
			j+=1
		ct[j]+=1
		sums[j]+=mat_price[i]
		i+=1
	print ct
	print sums/ct
	return ct,sums

def give_dummy_vars(lists,varname):
	for i,s in trainfile.iterrows():
		j=0
		for y in lists:
			if y==s[varname]:
				break
			j+=1
		trainfile.loc[i,varname] = j
	for i,s in testfile.iterrows():
		j=0
		for y in lists:
			if y==s[varname]:
				break
			j+=1
		testfile.loc[i,varname] = j
	testfile[varname]=testfile[varname].astype(int)
	trainfile[varname]=trainfile[varname].astype(int)

def printinfo():
	print trainfile.info()
	print testfile.info()
'''
#LandContour
ct_LandContour, sum_LandContour = average_counts(4,["Lvl","Bnk","HLS","Low"],"LandContour")
print ct_LandContour
print sum_LandContour/ct_LandContour
'''
give_dummy_vars(["Lvl","Bnk","HLS","Low"],"LandContour")

# print trainfile.info()
# print testfile.info()
'''
#Utilities Fill NaN values in testfile
ct_util,sum_util = average_counts(4,["AllPub","NoSewr","NoSeWa","ELO"],"Utilities")
print ct_util
print sum_util/ct_util
'''

#Drop Utilities
trainfile.drop(["Utilities"],axis=1,inplace=True)
testfile.drop(["Utilities"],axis=1,inplace=True)

#LotConfig
'''
ct_lotconfig, sum_lotconfig = average_counts(5,["Inside","Corner","CulDSac","FR2","FR3"],"LotConfig")
print ct_lotconfig
print sum_lotconfig/ct_lotconfig
'''
give_dummy_vars(["Inside","Corner","CulDSac","FR2","FR3"],"LotConfig")

#LandSlope
# ct_ls,sum_ls = average_counts(3,["Gtl","Mod","Sev"],"LandSlope")
give_dummy_vars(["Gtl","Mod","Sev"],"LandSlope")
'''
#Neighbourhood
neighbors=["Blmngtn","Blueste","BrDale","BrkSide","ClearCr","CollgCr","Crawfor","Edwards","Gilbert","IDOTRR","MeadowV","Mitchel","NAmes","NoRidge","NPkVill","NridgHt","NWAmes","OldTown","SWISU","Sawyer","SawyerW","Somerst","StoneBr","Timber","Veenker"]

ct_ng,sum_ng = average_counts(25,neighbors,"Neighborhood")
plt.plot(range(0,25),sum_ng/ct_ng)
plt.savefig("neighbors.png")
'''
neighbors=["Blmngtn","Blueste","BrDale","BrkSide","ClearCr","CollgCr","Crawfor","Edwards","Gilbert","IDOTRR","MeadowV","Mitchel","NAmes","NoRidge","NPkVill","NridgHt","NWAmes","OldTown","SWISU","Sawyer","SawyerW","Somerst","StoneBr","Timber","Veenker"]
give_dummy_vars(neighbors,"Neighborhood")
'''
#Condition1&2
ct_cond1,sum_cond1 = average_counts(9,conditions,"Condition1")
ct_cond2,sum_cond2 = average_counts(9,conditions,"Condition2")
ct_cond = np.array(ct_cond1)+np.array(ct_cond2)
sum_cond = np.array(sum_cond1)+np.array(sum_cond2)
print ct_cond
print sum_cond/ct_cond
# plt.plot(range(0,9),sum_cond/ct_cond)
# plt.savefig("Conditon1_and_2.png")
'''
conditions = ["Artery","Feedr","Norm","RRNn","RRAn","PosN","PosA","RRNe","RRAe"]
give_dummy_vars(conditions,"Condition1")
give_dummy_vars(conditions,"Condition2")

#BldgType
#ct_btype,sum_btype = average_counts(5,["1Fam","2fmCon","Duplex","TwnhsE","Twnhs"],"BldgType")
give_dummy_vars(["1Fam","2fmCon","Duplex","TwnhsE","Twnhs"],"BldgType")

#HouseStyle
#ct_htype,sum_htype = average_counts(8,["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl"],"HouseStyle")
give_dummy_vars(["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl"],"HouseStyle")

#OverallQuality
#ct_overall,sum_overall = average_counts(10,range(1,11),"OverallQual")

#OverallCondition
#ct_overall,sum_overall = average_counts(10,range(1,11),"OverallCond")
#Dropping Overall Condition as it does not give expected result

trainfile.drop(["OverallCond"],axis=1,inplace=True)
testfile.drop(["OverallCond"],axis=1,inplace=True)

#YearBuilt
# MIN_Year = np.amin(np.array(trainfile["YearBuilt"]))
# MAX_Year = np.amax(np.array(trainfile["YearBuilt"]))
'''
mat_yearbuilt = (np.array(trainfile["YearBuilt"])-1870)/10
ct_yearbuilt = np.zeros(15)
sum_yearbuilt = np.zeros(15)
i=0
for x in mat_yearbuilt:
	ct_yearbuilt[x]+=1
	sum_yearbuilt[x]+=mat_price[i]
	i+=1
print ct_yearbuilt
print sum_yearbuilt/ct_yearbuilt
plt.plot(range(0,15),sum_yearbuilt/ct_yearbuilt)
plt.savefig("YearBuilt.png")
'''
# mat_yearbuilt = (np.array(trainfile["YearBuilt"])-1870)/10
# j=0
# for i,s in trainfile.iterrows():
# 	trainfile.loc[i,"YearBuilt"] = mat_yearbuilt[j]
# 	j+=1
# mat_yearbuilt = (np.array(testfile["YearBuilt"])-1870)/10
# j=0
# for i,s in testfile.iterrows():
# 	testfile.loc[i,"YearBuilt"] = mat_yearbuilt[j]
# 	j+=1
# trainfile["YearBuilt"]=trainfile["YearBuilt"].astype(int)
# testfile["YearBuilt"]=testfile["YearBuilt"].astype(int)
#Drop YearBuilt

trainfile.drop(["YearBuilt"],axis=1,inplace=True)
testfile.drop(["YearBuilt"],axis=1,inplace=True)

# print trainfile["YearBuilt"].head()
# print testfile["YearBuilt"].head()

#YearRemodAdd
'''
mat_yearadd = (np.array(trainfile["YearRemodAdd"])-1870)/10
ct_yearadd = np.zeros(15)
sum_yearadd = np.zeros(15)
i=0
for x in mat_yearadd:
	ct_yearadd[x]+=1
	sum_yearadd[x]+=mat_price[i]
	i+=1
print ct_yearadd
print sum_yearadd/ct_yearadd
plt.plot(range(0,15),sum_yearadd/ct_yearadd)
plt.savefig("YearRemodAdd.png")
'''
# print np.amin(np.array(trainfile["YearRemodAdd"]))
# print np.amin(np.array(testfile["YearRemodAdd"]))
mat_yearadd = (np.array(trainfile["YearRemodAdd"])-1950)/10
j=0
for i,s in trainfile.iterrows():
	trainfile.loc[i,"YearRemodAdd"] = mat_yearadd[j]
	j+=1
mat_yearadd = (np.array(testfile["YearRemodAdd"])-1950)/10
j=0
for i,s in testfile.iterrows():
	testfile.loc[i,"YearRemodAdd"] = mat_yearadd[j]
	j+=1
trainfile["YearRemodAdd"]=trainfile["YearRemodAdd"].astype(int)
testfile["YearRemodAdd"]=testfile["YearRemodAdd"].astype(int)

# print trainfile["YearRemodAdd"].head()
# print testfile["YearRemodAdd"].head()

#RoofStyle
#ct_rd,sum_rs=average_counts(6,["Flat","Gable","Gambrel","Hip","Mansard","Shed"],"RoofStyle")
give_dummy_vars(["Flat","Gable","Gambrel","Hip","Mansard","Shed"],"RoofStyle")

#RoofMatl
#ct_rm,sum_rm = average_counts(8,["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"],"RoofMatl")

#General Function
def dropfunc(varname):
	trainfile.drop([varname],axis=1,inplace=True)
	testfile.drop([varname],axis=1,inplace=True)

#Drop RoofMatl
dropfunc("RoofMatl")

#Exterior1&2
# conditions = ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc","MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd","WdShing"]
# ct_cond1,sum_cond1 = average_counts(17,conditions,"Exterior1st")
# ct_cond2,sum_cond2 = average_counts(17,conditions,"Exterior2nd")
# ct_cond = np.array(ct_cond1)+np.array(ct_cond2)
# sum_cond = np.array(sum_cond1)+np.array(sum_cond2)
# print ct_cond
# print sum_cond/ct_cond


# give_dummy_vars(conditions,"Condition1")
# give_dummy_vars(conditions,"Condition2")

#Drop exterior1and2
dropfunc("Exterior1st")
dropfunc("Exterior2nd")


#MasVnrType   COntains NaN objects
#ct_mvt,sum_mvt= average_counts(5,["BrkCmn","BrkFace","CBlock","None","Stone"],"MasVnrType")
trainfile["MasVnrType"].fillna("None",inplace=True)
testfile["MasVnrType"].fillna("None",inplace=True)

give_dummy_vars(["BrkCmn","BrkFace","None","Stone"],"MasVnrType")

#MSVnrArea
#print (trainfile["MasVnrArea"]==0).sum()
#Drop as almost half of the values are zero
dropfunc("MasVnrArea")

#ExterQual
#cteq,sumeq = average_counts(5,["Ex","Gd","TA","Fa","Po"],"ExterQual")
#print (testfile["ExterQual"]=="Po").sum()
give_dummy_vars(["Ex","Gd","TA","Fa"],"ExterQual")

#ExterCond
# ctec,sumec = average_counts(5,["Ex","Gd","TA","Fa","Po"],"ExterCond")
# give_dummy_vars(["Ex","Gd","TA","Fa"],"ExterCond")
#Drop it
dropfunc("ExterCond")

#Foundation
#cf,sumf = average_counts(6,["BrkTil","CBlock","PConc","Slab","Stone","Wood"],"Foundation")
give_dummy_vars(["BrkTil","CBlock","PConc","Slab","Stone","Wood"],"Foundation")

#BsmtQual
# ctbsm,sumbsm = average_counts(5,["Ex","Gd","TA","Fa","Po"],"BsmtQual")
trainfile["BsmtQual"].fillna("TA",inplace=True)
testfile["BsmtQual"].fillna("TA",inplace=True)
give_dummy_vars(["Ex","Gd","TA","Fa"],"BsmtQual")
# print (testfile["BsmtQual"]=="Po").sum()

#BsmtCond
# ctbsm,sumbsm = average_counts(5,["Ex","Gd","TA","Fa","Po"],"BsmtCond")
# give_dummy_vars(["Ex","Gd","TA","Fa","Po"],"BsmtCond")
#Drop it
dropfunc("BsmtCond")

#BsmtExposure
# ctbm,sumbm = average_counts(5,["Gd","Av",'Mn','No','NA'],"BsmtExposure")
trainfile["BsmtExposure"].fillna("No",inplace=True)
testfile["BsmtExposure"].fillna("No",inplace=True)
give_dummy_vars(["Gd","Av",'Mn','No'],"BsmtExposure")

#Basement finishing
basement=['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA']
# ct1,sum1 = average_counts(7,basement,"BsmtFinType1")
trainfile["BsmtFinType1"].fillna("Unf",inplace=True)
testfile["BsmtFinType1"].fillna("Unf",inplace=True)
give_dummy_vars(basement,"BsmtFinType1")
#Drop finishing type 2
dropfunc("BsmtFinType2")
dropfunc("BsmtFinSF2")
# print (testfile["BsmtFinSF1"]==0).sum()
# testfile["BsmtFinSF1"].fillna()
# plt.scatter(trainfile["BsmtFinSF1"],trainfile["SalePrice"])
# plt.savefig("BsmtFinSF1.png")
testfile["BsmtFinSF1"].fillna(0,inplace=True)

#BsmtUnfSF
# print (trainfile["BsmtUnfSF"]==0).sum()
# plt.scatter(trainfile["BsmtUnfSF"],trainfile["SalePrice"])
# plt.savefig("BsmtUnfSF.png")
#Drop it
dropfunc("BsmtUnfSF")

#TotalBsmtSF
# plt.scatter(trainfile["TotalBsmtSF"],trainfile["SalePrice"])
# plt.savefig("TotalBsmtSF.png")

testfile["TotalBsmtSF"].fillna(0,inplace=True)

#Heating
#ctht,sumht = average_counts(6,['Floor','GasA','GasW','Grav','OthW','Wall'],'Heating')
#Drop it
dropfunc("Heating")

#HeatingQC
#ctbsm,sumbsm = average_counts(5,["Ex","Gd","TA","Fa","Po"],"HeatingQC")
give_dummy_vars(["Ex","Gd","TA","Fa","Po"],"HeatingQC")

# CentralAir
#ctca,sumca = average_counts(2,["N","Y"],"CentralAir")
give_dummy_vars(["N","Y"],"CentralAir")

#Electrical 1 null in train
#ctet,sumet = average_counts(5,['SBrkr','FuseA','FuseF','FuseP','Mix'],"Electrical")
#Drop it
dropfunc("Electrical")

#General function for plotting
def plotfunc(varname):
	plt.scatter(trainfile[varname],trainfile["SalePrice"])
	plt.savefig(varname+".png")

#1stFlrSF
#plotfunc("1stFlrSF")

#2ndFlrSF
# plotfunc("2ndFlrSF")
#print (trainfile["2ndFlrSF"]==0).sum()
#drop it
dropfunc("2ndFlrSF")

#LowQualFinSF
#plotfunc("LowQualFinSF")
#print (trainfile["LowQualFinSF"]==0).sum()
dropfunc("LowQualFinSF")

#GrLivArea
#plotfunc("GrLivArea")
# print (trainfile["GrLivArea"]==0).sum()

#BsmtFullBath 2 na in testfile
# plotfunc("BsmtFullBath")
# print (trainfile["BsmtFullBath"]==0).sum()
# ctbath,sumbath = average_counts(4,range(0,4),"BsmtFullBath")

testfile["BsmtFullBath"].fillna(0,inplace=True)

#BsmtHalfBath
#ctbath,sumbath = average_counts(4,range(0,4),"BsmtFullBath")
testfile["BsmtHalfBath"].fillna(0,inplace=True)

#Full&HalfBath
# ctbath,sumbath = average_counts(4,range(0,4),"FullBath")

# ctbath,sumbath = average_counts(4,range(0,4),"HalfBath")

#BedroomAbvGr
# ctbed,sumbed = average_counts(10,range(0,10),"BedroomAbvGr")

#KitchenAbvGr
#ctbed,sumbed = average_counts(10,range(0,10),"KitchenAbvGr")
#drop it
dropfunc("KitchenAbvGr")

#KitchenQual
#ctkq,sumkq = ctbsm,sumbsm = average_counts(5,["Ex","Gd","TA","Fa","Po"],"KitchenQual")
testfile["KitchenQual"].fillna("TA",inplace=True)
give_dummy_vars(["Ex","Gd","TA","Fa"],"KitchenQual")

#TotRmsAbvGrd
# plotfunc("TotRmsAbvGrd")

#Functional nan in test
# ctf,sumf = average_counts(8,['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal'],'Functional')
#Drop
dropfunc("Functional")

#Fireplaces
#plotfunc("Fireplaces")
#ctfp,sumfp = average_counts(4,range(0,4),"Fireplaces")

#FireplaceQu drop as only 729 datapts available
dropfunc("FireplaceQu")

#GarageType many na
#ctga,sumga = average_counts(7,['2Types','Attchd','Basment','BuiltIn','CarPort','Detchd','NA'],"GarageType")
trainfile["GarageType"].fillna("Attchd",inplace=True)
testfile["GarageType"].fillna("Attchd",inplace=True)
give_dummy_vars(['2Types','Attchd','Basment','BuiltIn','CarPort','Detchd','NA'],"GarageType")

#GarageYrBlt
# MINBUILT = np.amin(np.array(trainfile["GarageYrBlt"][pd.isnull(trainfile["GarageYrBlt"])==False]))
# MAXBUILT = np.amax(np.array(trainfile["GarageYrBlt"][pd.isnull(trainfile["GarageYrBlt"])==False]))
garagemat = (np.array(trainfile["GarageYrBlt"][pd.isnull(trainfile["GarageYrBlt"])==False]).astype(int)-1900)/10
# ct_yearadd = np.zeros(12)
# sum_yearadd = np.zeros(12)
# i=0
# for x in garagemat:
# 	ct_yearadd[x]+=1
# 	sum_yearadd[x]+=mat_price[i]
# 	i+=1
# print ct_yearadd
# print sum_yearadd/ct_yearadd
#ctnull = trainfile["GarageYrBlt"][pd.isnull(trainfile["GarageYrBlt"])].sum()
ctnull = pd.isnull(trainfile["GarageYrBlt"]).sum()
meanv = int((trainfile["GarageYrBlt"][pd.isnull(trainfile["GarageYrBlt"])==False]).mean())
stdv = int((trainfile["GarageYrBlt"][pd.isnull(trainfile["GarageYrBlt"])==False]).std())
ctnull2 = pd.isnull(testfile["GarageYrBlt"]).sum()
meanv2 = int((testfile["GarageYrBlt"][pd.isnull(testfile["GarageYrBlt"])==False]).mean())
stdv2 = int((testfile["GarageYrBlt"][pd.isnull(testfile["GarageYrBlt"])==False]).std())
#print ctnull,meanv,stdv

rand_train = np.random.randint(meanv-stdv,meanv+stdv, size = ctnull)
rand_test = np.random.randint(meanv2-stdv2,meanv2+stdv2, size = ctnull2)

trainfile["GarageYrBlt"][np.isnan(trainfile["GarageYrBlt"])]=rand_train
testfile["GarageYrBlt"][np.isnan(testfile["GarageYrBlt"])]=rand_test

for i,s in trainfile.iterrows():
		trainfile.loc[i,"GarageYrBlt"] = (trainfile.loc[i,"GarageYrBlt"]-1900)/10
for i,s in testfile.iterrows():
		testfile.loc[i,"GarageYrBlt"] = (testfile.loc[i,"GarageYrBlt"]-1900)/10

trainfile["GarageYrBlt"]=trainfile["GarageYrBlt"].astype(int)
testfile["GarageYrBlt"]=testfile["GarageYrBlt"].astype(int)
# print trainfile["GarageYrBlt"].head()
# print testfile["GarageYrBlt"].head()

#GarageFinish
#ctg,sumg = average_counts(3,['Fin','RFn','Unf'],'GarageFinish')
# print (testfile["GarageFinish"]=="NA").sum()
give_dummy_vars(['Fin','RFn','Unf'],'GarageFinish')
ctn = pd.isnull(trainfile["GarageFinish"]).sum()
ctn2 = pd.isnull(testfile["GarageFinish"]).sum()
trainfile["GarageFinish"][pd.isnull(trainfile["GarageFinish"])] = np.random.randint(0,2,size=ctn)
testfile["GarageFinish"][pd.isnull(testfile["GarageFinish"])] = np.random.randint(0,2,size=ctn)

# print trainfile["GarageFinish"].head()
# print testfile["GarageFinish"].head()

#GarageCars
# plotfunc("GarageCars")
testfile["GarageCars"].fillna(3,inplace=True)

#GarageArea
#plotfunc("GarageArea")
meanvalue = int((testfile["GarageArea"][pd.isnull(testfile["GarageArea"])==False]).mean())
# print meanvalue
testfile["GarageArea"].fillna(meanvalue,inplace=True)
'''
#GarageQual
#ctkq,sumkq = ctbsm,sumbsm = average_counts(6,["Ex","Gd","TA","Fa","Po","NA"],"GarageQual")
dropfunc("GarageQual")

#GarageCond
#ctkq,sumkq = ctbsm,sumbsm = average_counts(6,["Ex","Gd","TA","Fa","Po","NA"],"GarageCond")
dropfunc("GarageCond")

#PavedDrive
#ctp,sump = average_counts(3,['Y','P','N'],"PavedDrive")
dropfunc("PavedDrive")

#WoodDeckSF
#plotfunc("WoodDeckSF")

#OpenPorchSF
#plotfunc("OpenPorchSF")

#EnclosedPorch
#plotfunc("EnclosedPorch")
dropfunc("EnclosedPorch")

#3SsnPorch
# plotfunc("3SsnPorch")
dropfunc("3SsnPorch")

#ScreenPorch
# plotfunc("ScreenPorch")
dropfunc("ScreenPorch")

#PoolArea
# plotfunc("PoolArea")
dropfunc("PoolArea")
dropfunc("PoolQC")
'''
#MiscVal
#plotfunc("MiscVal")
# print (trainfile["MiscVal"]==0).sum()
trainfile.drop(["GarageQual","GarageCond","PavedDrive","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature","MiscVal"],axis=1,inplace=True)
testfile.drop(["GarageQual","GarageCond","PavedDrive","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature","MiscVal"],axis=1,inplace=True)

# plotfunc("MoSold")

#YrSold
# plotfunc("YrSold")
dropfunc("YrSold")

#SaleType
#ctst,sumst =average_counts(10,['WD','CWD','VWD','New','COD','Con','ConLw','ConLI','ConLD','Oth'],"SaleType")
dropfunc("SaleType")

#SaleCondition
#ctsc,sumsc = average_counts(6,['Normal','Abnorml','AdjLand','Alloca','Family','Partial'],"SaleCondition")
dropfunc("SaleCondition")

#Regression
train_data = trainfile.as_matrix()
train_features = train_data[:,:-1]
train_labels = train_data[:,-1:]

# print train_features[0]
# print "------------------------------"
# print train_labels[0]

