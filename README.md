# rec-system
A program to predict users' ratings for movies using Collaborative Filtering.


#Dataset#
The Movielens dataset was used to predict movie ratings for users.
Numer of ratings= 100,000

#Technique#
Item-item collaborative filtering was used wherever applicable, thus size of utility matrix = movies X users

#Hyperparameters#
k=200
Nearest neighbours= [5,10,50] --> Results calculated by taking average over the range
Train-test split= 80%:20%
