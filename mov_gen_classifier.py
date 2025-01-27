import numpy as np
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

movies = pd.read_csv('movies.csv')
movies[movies["Title"] == "wild wild west"].iloc[:,[0, 1, 2, 3, 4, 14, 49, 1042, 4004]]
movies = movies.set_index('Title', drop = False)
movies["Title"] = movies.index

vocab_mapping = pd.read_csv('stem.csv')
stemmed = np.take(movies.columns, np.arange(3, len(movies.columns)))

vocab_table = pd.DataFrame({'Stem': stemmed})
vocab_table = pd.merge(vocab_table, vocab_mapping, on = "Stem", how = "left")
vocab_table[1101:1111]

stemmed_message = 'veget'
stemmed_message

stemList = list(stemmed)
maxval = 0
stem = ''
for i in stemList:
    val = vocab_table.loc[vocab_table['Stem'] == i].count()['Stem']
    if val >= maxval:
        maxval = val
        stem = i
maxstem = vocab_table.groupby(["Stem"]).count().reset_index()
maxstem.loc[maxstem["Word"] == maxstem["Word"].max(), "Stem"]
most_stem = maxstem.loc[maxstem["Word"] == maxstem["Word"].max(), "Stem"].item()
most_stem
vocab_table["Stem length"] = vocab_table["Stem"].str.len()
vocab_table["Word length"] = vocab_table["Word"].str.len()

vocab_table['difference'] = vocab_table['Word length'] - vocab_table['Stem length']

diff = vocab_table.loc[vocab_table['difference'] == 0]

longest_uncut = diff.sort_values('Word length', ascending = False).iloc[1,0]
longest_uncut

outer_space = movies[["outer", "space"]]
sns.scatterplot(data = outer_space, x = "outer", y = "space")
plt.xlim(-.0005, .001);


outer = movies["outer"]
space = movies["space"]

def su(a):
    mean = (a-np.mean(a))
    std = np.std(mean)
    return mean/std

outer_su = su(movies['outer'])
space_su = su(movies['space'])

outer_space_r = np.mean(outer_su*space_su)
outer_space_r
word_x = 'light'
word_y = 'dark'
arr_x = movies[word_x]
arr_y = movies[word_y]

x_su = su(arr_x)
y_su = su(arr_y)

r = np.mean(x_su * y_su)

slope = r * (np.std(arr_y) / np.std(arr_x))
intercept = np.mean(arr_y) - slope * np.mean(arr_x)
sns.scatterplot(x = arr_x, y = arr_y)
max_x = np.max(movies[word_x])
plt.title(f"Correlation: {np.round(r,3)}, magnitude greater than .2: {abs(r) >= 0.2}")
sns.lineplot(x = [0, max_x * 1.3], y = [intercept, intercept + slope * (max_x*1.3)], color='gold');
training_proportion = 17/20

num_movies = len(movies)
num_train = int(num_movies * training_proportion)
num_test = num_movies - num_train

train_movies = movies.iloc[np.arange(num_train),:]
test_movies = movies.iloc[np.arange(num_train, num_movies)]

print("Training: ",   len(train_movies), ";",
      "Test: ",       len(test_movies))

def comedy_proportion(table):
    len_c = len(table[table['Genre'] == 'comedy'])
    len_t = len(table)
    return len_c/len_t

test_prop = comedy_proportion(test_movies)
train_prop = comedy_proportion(train_movies)
proportion_df = pd.DataFrame({'Tables': ['Test Set', 'Train Set'], 'Proportion': [test_prop, train_prop]})
proportion_df.plot.barh(x = 'Tables', y = 'Proportion')


def plot_with_two_features(test_movie, training_movies, x_feature, y_feature):
    """Plot a test movie and training movies using two features."""
    test_row = test_movies.loc[test_movie]
    distances = pd.DataFrame({
            "x_feature": test_row[x_feature],
            "y_feature": test_row[y_feature],
            'Genre': 'unknown',
            'Title': test_movie
                }, index = [0])
    
    for movie in training_movies:
        row = train_movies.loc[movie]
        distances.loc[len(distances)] = [row[x_feature], row[y_feature], row['Genre'], movie]

    sns.set_palette("dark")
    sns.scatterplot(data = distances, x = "x_feature", y = "y_feature", hue = "Genre", s = 70)
    plt.legend(labels = distances["Title"], fontsize = "small", labelcolor = sns.color_palette()[0:3], markerscale=0)

  training = ["clerks.", "the avengers"]
plot_with_two_features("monty python and the holy grail", training, "water", "feel")
movies.loc[movies["Title"] =="monty python and the holy grail",['water','feel']]

python = movies.loc[movies["Title"] =="monty python and the holy grail",['water','feel']]
avengers = movies.loc[movies["Title"] =="the avengers",['water','feel']]

one_distance = math.sqrt((math.pow(python['water'].item() - avengers['water'].item(),2)) +  ((math.pow(python['feel'].item() - avengers['feel'].item(),2))))
one_distance
training = ["clerks.", "the avengers", "the silence of the lambs"] 
plot_with_two_features("monty python and the holy grail", training, "water", "feel");
def distance_two_features(title0, title1, x_feature, y_feature):
    x1 = movies.loc[movies["Title"] == title0, x_feature].item()
    x2 = movies.loc[movies["Title"] == title1, x_feature].item()
    
    y1 = movies.loc[movies["Title"] == title0, y_feature].item()
    y2 = movies.loc[movies["Title"] == title1, y_feature].item()
    
    x_sub = (x1 - x2)**2
    y_sub = (y1 - y2)**2
    diff = math.sqrt(x_sub+y_sub)
    return diff

for movie in np.array(["clerks.", "the silence of the lambs"]):
    movie_distance = distance_two_features(movie, "monty python and the holy grail", "water", "feel")
    print(movie, 'distance:\t', movie_distance)
def distance_from_python(title):
    """The distance between the given movie and "monty python and the holy grail", 
    based on the features "water" and "feel".
    
    This function takes a single argument:
      title: A string, the name of a movie.
    """
    

    return distance_two_features(title, title1='monty python and the holy grail', x_feature="water", y_feature="feel")
  distances = movies.apply(lambda row: distance_from_python(row['Title']), axis=1)

close_movies = pd.DataFrame(columns=['Title', 'Genre', 'water', 'feel', 'distance from python'])
close_movies['Title'] = movies['Title']
close_movies['Genre'] = movies['Genre']
close_movies['water'] = movies['water']
close_movies['feel'] = movies['feel']
close_movies['distance from python'] = distances

close_movies.sort_values('distance from python', inplace=True)
close_movies = close_movies.head(5)
close_movies = close_movies[['Title', 'Genre', 'water', 'feel', 'distance from python']]
def most_common(label, table):
    """The most common element in a column of a table.
    
    This function takes two arguments:
      label: The label of a column, a string.
      table: A table.
     
    It returns the most common value in that column of that table.
    In case of a tie, it returns any one of the most common values
    """
    grouptable = table.groupby(by=label).count().reset_index()
    max_count = grouptable.iloc[:,1].max()
    dfvals = grouptable.loc[grouptable.iloc[:,1] == max_count, label]
    
    return dfvals.iloc[0]
most_common('Genre', close_movies)
def distance(features_array1, features_array2):
    """The Euclidean distance between two arrays of feature values."""
    sol = 0
    for i in range(len(features_array1)):
        sub_diff = features_array1[i] - features_array2[i]
        sum_squ = sub_diff ** 2
        sol += sum_squ
    return np.sqrt(sol)


first = movies.iloc[0,5:-1].to_list()
second = movies.iloc[1,5:-1].to_list()

distance_first_to_second = distance(first, second)
distance_first_to_second
my_features = np.array(['i', 'the', 'to', 'a', 'it', 'and', 'that', 'of', 'your', 'what'])

train_my_features = train_movies[list(my_features)]
test_my_features = test_movies[list(my_features)]
print(f"Movie:", test_movies.iloc[0,:][['Title', 'Genre']])

print("Features:")
test_my_features.iloc[0,:]
genre_and_distances = np.empty(0)

for i in range(len(train_my_features)):
    genre_and_distances = np.append(genre_and_distances, distance(train_my_features.iloc[i], test_my_features.iloc[0]))


genre_and_distances = train_movies[['Genre']]
genre_and_distances['Distance'] = genre_and_distances
genre_and_distances
my_assigned_genre = genre_and_distances.nsmallest(7, 'Distance')['Genre'].value_counts().idxmax()
my_assigned_genre_was_correct = True

print("The assigned genre, {}, was{}correct.".format(my_assigned_genre, " " if my_assigned_genre_was_correct else " not "))
def classify(test_row, train_rows, train_labels, k):
    """Return the most common class among k nearest neighbors to test_row."""
    distances = np.sqrt(np.sum((train_rows - test_row) ** 2, axis=1))
    k_nearest = train_labels[np.argsort(distances)[:k]]
    unique_classes, class_counts = np.unique(k_nearest, return_counts=True)
    max_count = np.argmax(class_counts)
    predicted_class = unique_classes[max_count]
    return predicted_class
tron_genre = classify(test_my_features.loc[test_movies['Title'] == 'tron'].values[0], train_my_features, np.array(train_movies["Genre"]), 13)
tron_genre
def classify_feature_row(row):
    return classify(row, train_my_features, train_movies["Genre"], 13)
    
classify_feature_row(test_my_features.iloc[0, :])

test_guesses = test_my_features.apply(classify_feature_row, axis=1)
proportion_correct = sum(test_guesses == test_movies["Genre"]) / len(test_guesses)
proportion_correct
test_movie_correctness = test_movies[["Title", "Genre"]].copy()
test_movie_correctness["Was correct"] = test_guesses == test_movies["Genre"]

new_features = np.array(["laugh", "marri", "dead", "heart", "cop"])

train_new = train_movies[new_features]
test_new = test_movies[new_features]

def another_classifier(row):
    return classify(row, train_new, np.array(train_movies["Genre"]), 13)

test_guesses = test_new.apply(another_classifier, axis=1)
new_correct = sum(test_guesses == test_movies["Genre"]) / len(test_guesses)

