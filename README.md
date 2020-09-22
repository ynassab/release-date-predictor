# release-date-predictor
Predict the release year of a song based on the qualities of its sound (e.g. tempo, loudness, "acousticness", "speechiness"). Song release dates range from 1928 to 2020. See the file 'data.csv' for all metrics used. The dataset, gathered using the Spotify API, is provided by the Kaggle user Yamac Eren Ay under a Community Data License Agreement.

# Implementation
Two hidden layers, each with 14 neurons corresponding to the input metrics, minimize the mean squared error (MSE) of the predicted year. Thus, the cost function rewards the model for guessing close to the correct year, even if not exact. Using accuracy as the loss and metric target was not expected to be effective since there are 93 unique years present in the dataset, and the input parameters are rather ambitious, making accurate predictions challenging.

In the limit of a perfect model, the MSE would be 0, although the actual global minimum of this parameter space is probably somewhere between 0 and 10. The target MSE for this model is 25 (i.e. predictions are, on average, about five years off target). For reference, a model typically has an MSE on the order of 10^6 after one training epoch.

# Current Limitations
In minimizing MSE, the model ultimately converges on a local minimum of approximately 600 by only guessing the average year in the dataset. This should be expected given a random starting point in the parameter space: the gradient of MSE will almost always point towards the average for a seemingly random dataset. Only when the parameters are within very specific ranges of values will there be gradient descent to somewhere in the vicinity of the global minimum.

# Possible solutions

1. Decrease the number of parameters, with or without including qualities that do not correspond the sound characteristics (e.g. UTF-8-encoded song and artist names). However, this is unlikely to fix the issue of the vary wide basin in the parameter space leading to the local minimum. Also, providing artist names gives a hint to the release years - it could be useful to test, but goes against the current vision for the project.
2. Implement 'accuracy' as the target metric, or explore other loss and metric targets, and engineer the number of model layers and neurons as well as the runtime with those in mind. The model may need to be trained over dozens of epochs before overfitting is observed; thus far, it has shown (what is assumed to be) significant underfitting with accuracy-targetting. If a clear overfitting regime is found, it should be possible to find a Goldilocks regime through simple trial and error. A custom activation function on the last layer for restricting the output to be within the possible year range (1928-2020) will be especially useful here.
3. Enforce an upper limit on the MSE - one that is impossible to achieve by guessing the average year alone, such as 100 - and otherwise keep the model as is. Continuously retrain and/or recycle the models until it escapes the local minimum. This method will likely take an exceedingly long time, and it is probably not an optimal solution without access to a powerful GPU.
