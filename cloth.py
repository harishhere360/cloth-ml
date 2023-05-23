import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clothing_similarity(text):
  """
  This function takes a text string and returns a list of the top-N most similar clothing items.

  Args:
    text: A text string describing a clothing item.

  Returns:
    A list of the top-N most similar clothing items.
  """

  
  vectorizer = CountVectorizer()
  features = vectorizer.fit_transform([text])


  similarities = cosine_similarity(features, features)


  ranked_items = np.argsort(similarities)[::-1]

  
  return ranked_items[:10]


def main():

  text = input("Enter a text string describing a clothing item: ")

  
  similar_items = clothing_similarity(text)

  
  for item in similar_items:
    print(item)


if __name__ == "__main__":
  main()
