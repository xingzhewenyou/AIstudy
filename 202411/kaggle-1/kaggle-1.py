import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the datasets
train_df = pd.read_csv('/Users/zhangwenyou/PycharmProjects/AIstudy/202411/kaggle-1/train.csv')
test_df = pd.read_csv('/Users/zhangwenyou/PycharmProjects/AIstudy/202411/kaggle-1/test.csv')


# Data preprocessing
def preprocess_data(df):
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Drop columns that won't be used
    df = df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

    return df


train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Separate features and target variable from training data
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred)}')

# Predict on the test data
test_predictions = model.predict(test_df)

# Prepare the submission file
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('/Users/zhangwenyou/PycharmProjects/AIstudy/202411/kaggle-1/test.csv')['PassengerId'],
    'Survived': test_predictions
})

submission.to_csv('/Users/zhangwenyou/PycharmProjects/AIstudy/202411/kaggle-1/submission.csv', index=False)
print('Submission file created successfully.')
