import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time

# Use Pandas to upload the csv and put it into a dataframe and datatable
csv = pd.read_csv("bookdataset.csv")
data = pd.DataFrame(data=csv)

# Replace no/yes to 0/1 to make things easy and numerical
df = data.replace(to_replace="Yes", value=1)
df.replace(to_replace="No", value=0, inplace=True)
training_df = df.drop(columns="Book ID", inplace=False, axis=1)

# the chosen one
selection = training_df.iloc[0]


# ------------ DISTANCE ALGORITHMS -----------
# Knn analyzes the dataset to find the top most similar rows given a row
# it measures the distance between the rows using euclidean method
# it votes on the number of similar features to the given target
# k is the number of neighbours we want to find
def knn(dataframe, selected_book, k):
    # t0 = time.process_time()
    # drop any missing values (alternatively i could find an average of the values to make an educated guess to fill)
    dataframe = dataframe.dropna()
    # reshape the row for formatting
    target = selected_book.values.reshape(1, -1)
    # added one to compensate for index 0
    neighbour = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    neighbour.fit(dataframe)

    # finds the index indices for the neighbours
    distance, index = neighbour.kneighbors(target)

    top5 = []
    knn_index = index[0][1:]
    # pull each row from the original dataframe and add it to the top 5 list
    for i in knn_index:
        item = dataframe.loc[i]
        top5.append(item)
    top5_df = pd.DataFrame(top5)
    # t1 = time.process_time()
    # print("KNN Time Elapsed:", t1-t0)
    return top5_df


# Euclidian method to find distance between two rows
def euclidian(dataframe, selected_book):
    # t0 = time.process_time()
    # drop any missing values (alternatively i could find an average of the values to make an educated guess to fill)
    dataframe = dataframe.dropna()
    dist_tuples = []
    for index, row in dataframe.iterrows():
        series1 = pd.Series(row.values)
        series2 = pd.Series(selected_book.values)
        # Manual euclidian formula
        distance = sum((series1 - series2) ** 2) ** 0.5
        dist_tuples.append((index, distance))
    # sort based on the second tuple value (the distance)
    dist_tuples.sort(key=lambda x: x[1])
    # skips the first one as it is the selected book and distance is 0.0
    top5_tuples = dist_tuples[1:6]
    # pull from original dataframe
    top5 = []
    for i in top5_tuples:
        item = dataframe.iloc[i[0]]
        top5.append(item)
    top5_df = pd.DataFrame(top5)
    # t1 = time.process_time()
    # print("Euclidean Time Elapsed:", t1-t0)
    return top5_df


# ------------- PERCENTAGE MATHS -----------
# compare two rows to check similarity percentage rate using the jaccard method
# this will find the "crossroads" of the data indices
def book_similarity(selected, top):
    # create sets of indices for each row where a 1 is found
    selected_set = set(np.where(selected == 1)[0])
    top_set = set(np.where(top == 1)[0])
    # uses intersection to find the "crossroads" in which the sets meet
    intersection = selected_set.intersection(top_set)
    union = selected_set.union(top_set)
    # calculates the similarity and creates a percentile at the 2nd decimal
    similarity = len(intersection) / len(union)

    return np.round(similarity * 100, 2)


# ---------------- FILE SAVE -----------------
def save_top(top_books, percentages):
    new_rows = []
    # get the selected rows from the original dataset
    for index, row in top_books.iterrows():
        new_rows.append(data.iloc[index])
    df = pd.DataFrame(new_rows)
    df['Similarity %'] = np.nan
    for item in percentages:
        df.loc[item[0], 'Similarity %'] = item[1]
    df.to_csv(path_or_buf="top5books.csv", sep=',')



# -----------------GRAPHING--------------------
def percent_scatter(top, full):
    # all books in light blue
    plt.scatter([x[0] for x in full], [y[1] for y in full], color='lightpink')
    # top books in cyan
    plt.scatter([x[0] for x in top], [y[1] for y in top], color='deeppink')
    # to put labels on top book plot points for fun
    #for i in top:
    #    plt.text(i[0], i[1], "Book " + str(i[0]) + " " + str(i[1]) + "%")
    plt.ylabel("Book Similarity (Percentage)", color='deeppink')
    plt.xlabel("Top Books (Index)", color='deeppink')
    plt.title("Top Most Similar Books", color='deeppink')
    ticks = np.arange(0, len(full), 5)
    plt.xticks(ticks)
    plt.show()


def percent_bar(top, full):
    # all books in light blue
    plt.bar([x[0] for x in full], [y[1] for y in full], color='orchid')
    # top books in cyan
    plt.bar([x[0] for x in top], [y[1] for y in top], color='darkorchid')
    plt.ylabel("Book Similarity (Percentage)", color='darkorchid')
    plt.xlabel("Top Books (Index)", color='darkorchid')
    plt.title("Top Most Similar Books", color='darkorchid')
    ticks = np.arange(0, len(full), 5)
    plt.xticks(ticks)
    plt.show()


# top books found with k nearest neighbour method
knn_books = []
for index, row in knn(dataframe=training_df, selected_book=selection, k=5).iterrows():
    sim = book_similarity(selection, row)
    knn_books.append((index, sim))
# top books found with direct euclidian method
euclidian_books = []
for index, row in euclidian(dataframe=training_df, selected_book=selection).iterrows():
    sim = book_similarity(selection, row)
    euclidian_books.append((index, sim))
# all the books!
all_books = []
for index, row in df.iterrows():
    sim = book_similarity(selection, row)
    all_books.append((index, sim))

# KNN Graph plotting
#percent_scatter(top=knn_books, full=all_books)
#percent_bar(top=knn_books, full=all_books)

# Euclidean Graph plotting
#percent_scatter(top=euclidian_books, full=all_books)
#percent_bar(top=euclidian_books, full=all_books)

# The top books
#print(euclidian(dataframe=training_df, selected_book=selection))
#print(knn(dataframe=training_df, selected_book=selection, k=5))

# File write
#save_top(euclidian(dataframe=training_df, selected_book=selection), euclidian_books)
#save_top(knn(dataframe=training_df, selected_book=selection), knn_books)

