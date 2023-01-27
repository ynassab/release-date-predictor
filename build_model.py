from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import pickle


if __name__ == "__main__":
	df = read_csv('data.csv', engine='python') # import data
	
	# select columns of interest
	df = df[['artists', 'popularity', 'year']]
	
	# generate unique ID for each artist
	artist_dict = {}
	unique_id = 0
	for value in df['artists'].unique():
		artist_dict[value] = unique_id
		unique_id += 1
	
	df['artists'] = df['artists'].apply( lambda x : artist_dict[x] )
	
	df.sample(frac=1)  # shuffle dataframe rows
	
	train_df, test_df = train_test_split(df, test_size=0.2)
	
	train_target = train_df.pop('year')
	test_target = test_df.pop('year')
	
	# find the optimal n_neighbours
	x_range = range(1,20)
	scores = []
	for i, n_neighbors in enumerate(x_range):
		knn = KNeighborsClassifier(n_neighbors=n_neighbors)
		knn.fit(train_df.values, train_target.values)
		score = knn.score(test_df, test_target)
		scores.append(score)
		print(i, score)
	
	
	# display fit evolution
	plt.scatter(x_range, scores)
	plt.title('Evolution of Fit Accuracy')
	plt.xlabel('Accuracy')
	plt.ylabel('Number of Nearest Neighbours')
	plt.savefig('./fit_evolution.png')
	plt.show()
	
	
	# return the KNN with the best fit
	max_i = scores.index(max(scores))
	
	n_neighbours = max_i + 1
	
	knn = KNeighborsClassifier(n_neighbors=n_neighbors)
	knn.fit(train_df.values, train_target.values)
	savepath = './model.obj'
	with open(savepath, 'wb') as f:
		pickle.dump(knn, f)
	
