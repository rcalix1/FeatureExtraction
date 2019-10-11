import os
import pandas as pd
import sklearn

#path ='Good_CSVs'
path = 'temp'

files = []
labels = []
example = []

for filename in os.listdir(path):
    if "Good" in filename:
        labels.append("1")
    else:
        labels.append("-1")
    filename = os.path.join(path, filename)
    print filename
    with open(filename) as f:
        content = f.read()
    content.replace(",", " ")
    content.replace('"', " ")
    example.append(content)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1, stop_words='english', max_features=1000)

dtm = vectorizer.fit_transform(example)

df = pd.DataFrame(dtm.toarray(), index=labels, columns=vectorizer.get_feature_names())
print(df)

df.to_csv(r'matrix.csv')

features_list= vectorizer.get_feature_names()
for feature in features_list:
    print str(feature)

