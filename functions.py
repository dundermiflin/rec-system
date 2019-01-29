import pandas as pd
import numpy as np
import math
np.random.seed(7)

def cosine(a,b):
    '''
        Calculates the cosine distance between two numpy vectors.

        Small values are added to the denominator to prevent Divide by Zero exceptions.
    '''
    mod_a= (1e-5+np.sum(a**2))**0.5
    mod_b= (1e-5+np.sum(b**2))**0.5
    prod= a.dot(b.T)
    return prod/(mod_a*mod_b)

def top_collab(movie,user,mat,sim,k):
    '''
        Finds k most similar movies to a particular watched by a given user. Returns a tuple containing an array of movies and their similarity to the input movie.

        As baselining is not used, we can't include movies that he hasn't rated. The distances are precalculated as distance between same pair can be used multiple times.
    '''
    scores=[]
    movie_vec= mat[movie]
    for i in range(mat.shape[0]):
        if mat[i][user]>0:
            scores.append((i,sim[movie][i]))
    closest= sorted(scores,key= lambda x:x[1],reverse=True)[0:k]
    best= [c[0] for c in closest]
    distances= [c[1] for c in closest]
    return (best,distances)

def predict_collab(movie,user,mat,sim):
    '''
        Function to predict rating for a particular movie by a given user for Collaborative filtering.

        First, the most similar movies are extracted. Then the rating is calculated by calculating weighted mean of the ratings with the weight being the similarities.
    '''
    best,distances= top_collab(movie,user,mat,sim,50)
    pred=0
    for i in range(len(best)):
        pred+=distances[i]*mat[best[i]][user]
    pred/=(1e-5+np.sum(np.array(distances)))
    return pred


def rmse(a,b):
    '''
        Function to calculate the root mean square error between the predicted and the target values.
    '''
    return (np.mean((a-b)**2)**0.5)

def spearman(a,b):
    '''
        Calculates the Spearman correlation between target and predicted ratings.

        The formula used is sp= 1-(6*sum(difference^2)/(n*(n^2-1)))
    '''
    d2= np.sum((a-b)**2)
    d2*=6
    n= a.shape[0]
    d2/= n*(n**2-1)

    return (1-d2)

def precision(pairs,k):
    '''
        Calculates the accuracy for the top-k values on a zipped array with target and predicted ratings 
    '''
    pairs= sorted(pairs,key= lambda x: x[1],reverse=True)
    pairs= pairs[:k]
    correct=0
    for p in pairs:
        if p[0]==p[1]:
            correct+=1
    n= len(pairs)
    return correct/k


