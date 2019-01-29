from func import *
from time import clock

start_time= clock()

test_size=85000
df= pd.read_csv('u.data',delimiter='\t',header=None,engine='python')
df=df.drop(columns=[3])

data= np.array(df)
data[:,:2]-=1

#Splitting the data for testing and training
train= data[:test_size]
test= data[test_size:]


#Extracting list of all users and movies to build utility matrix
users=list(set(data[:,0]))
movies= list(set(data[:,1]))

util= np.zeros(shape=(len(movies),len(users)))

#Inserting values into utility matrix
for row in train:
    u= row[0]
    m= row[1]
    r= row[2]
    util[m][u]=r

#Precalculating Similarities for the vectors
print("Calculating Similarity Matrix...")
sim=[[cosine(util[i],util[j]) for j in range(len(movies))] for i in range(len(movies))]
sim= np.array(sim)

pred=[]
i=0
for t in test:
    i+=1
    print("Predicting test case {0}".format(i))
    ans= predict_collab(t[1],t[0],util,sim)
    #Restrictng the valid space for the answer.
    if ans<0:
        ans=0
    if ans>5:
        ans=5
    pred.append(ans)

#Evaluation metrics
pred= np.array(pred)
y=test[:,-1]
pairs= list(zip(y,pred))
print("The RMSE value is {0}".format(rmse(pred,y)))
print("Time taken by the program was {0} seconds".format(clock()-start_time))
print("The spearman correlation is {0}".format(spearman(pred,y)))
