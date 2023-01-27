from pandas import read_csv
import pickle


if __name__ == "__main__":
	
	print('Loading data...')
	
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
	
	df.sample(frac=0.1)  # random sample
	target = df.pop('year')
	
	print("Loading model...")
	
	with open('./model.obj', 'rb') as f:
		knn = pickle.load(f)
	
	print(f'Model accuracy: {knn.score(df, target)}')
	
	
