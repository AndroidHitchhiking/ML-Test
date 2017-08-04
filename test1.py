import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd
import ezodf


#df = pd.read_csv('svm_test_1.ods')
doc = ezodf.opendoc('svm_test_1.ods')
#print("Spreadsheet contains %d sheet(s)." % len(doc.sheets))
#print(df.head())

sheet = doc.sheets[0]
df_dict = {}
for i, row in enumerate(sheet.rows()):
    # row is a list of cells
    # assume the header is on the first row
    if i == 0:
        # columns as lists in a dictionary
        df_dict = {cell.value:[] for cell in row}
        # create index for the column headers
        col_index = {j:cell.value for j, cell in enumerate(row)}
        continue
    for j, cell in enumerate(row):
        # use header instead of column index
        df_dict[col_index[j]].append(cell.value)
# and convert to a DataFrame
df = pd.DataFrame(df_dict)
#Drop S.No. colomn
#df.replace('?',-99999, inplace=True)
df.drop(['S.No','Engg Group'], 1, inplace=True)
#df.drop(['None'], 1, inplace=True)
#df.dropna(inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
#print(df.tail())
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
print(df.head)
X = np.array(df.drop(['Applicatiom'], 1))
y = np.array(df['Applicatiom'])




#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
#features_train = vectorizer.fit_transform(X_train)
#features_test  = vectorizer.transform(X_test).toarray()
							 
clf = svm.SVC()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)



