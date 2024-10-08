# %%
# Importando bibliotecas 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine import encoding
import matplotlib.pyplot as plt
import numpy as np

# %%
df = pd.read_csv("train_titanic.csv")
df_teste1 = pd.read_csv("test.csv")

# %%
df.head(5)

# %%
df.info()

# %%
df_teste1.shape

# %%
df.describe()

# %%
df.isnull().sum().sort_values(ascending=False)

# %%
df_teste1.isnull().sum().sort_values(ascending=False)

# %%
df["Sex"].value_counts()

# %%
df.groupby('Age')['Survived'].sum().sort_values(ascending=False).head(20)

# %%
df.groupby('Sex')[['Survived']].sum()

# %%
df_teste1

# %%
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])
df_teste = df_teste1.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

# %%
df_teste

# %%
sns.boxplot(df["Fare"])

# %%
sns.boxplot(df_teste["Fare"])

# %%
df["Age"] = df["Age"].fillna(df['Age'].median())
df_teste["Age"] = df_teste["Age"].fillna(df_teste['Age'].median())

# %%
df["Embarked"].value_counts()

# %%
df["Embarked"] = df["Embarked"].fillna("S")
df_teste["Embarked"] = df_teste["Embarked"].fillna("S")

# %%
df_teste["Fare"] = df_teste["Fare"].fillna(df["Fare"].median())

# %%
sobreviventes_por_sexo = df.groupby(['Sex', 'Survived']).size()

sns.barplot(x='Sex', y='Survived', data=df)

plt.title('Balanço de Homens e Mulheres que Sobreviveram')
plt.xlabel('Sexo')
plt.ylabel('Porcentagem de Sobreviventes')
plt.show()

# %%
bins = [0, 18, 30, 50, 100]  
labels = ['0-18', '19-30', '31-50', '51+']  

df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='AgeGroup', hue='Survived', palette={0: 'blue', 1: 'red'})

plt.title('Contagem de Sobreviventes e Não Sobreviventes por Faixa Etária')
plt.xlabel('Faixa Etária')
plt.ylabel('Contagem')
plt.legend(title='Sobreviveu', loc='upper right', labels=['Não', 'Sim'])
plt.show()

# %%
df = df.drop(columns=["AgeGroup"])

# %%
bins = [0, 50, 100, 150, 200, 300]
labels = ['0-50', '51-100', '101-150', '151-200', '201-280']
df['FareCategory'] = pd.cut(df['Fare'], bins=bins, labels=labels)

survival_rate = df.groupby('FareCategory')['Survived'].mean() * 100

plt.figure(figsize=(10, 6))
survival_rate.plot(kind='bar', color='blue')
plt.title('Taxa de Sobrevivência por Faixa de Tarifa Paga')
plt.xlabel('Faixa de Tarifa (em dólares)')
plt.ylabel('Taxa de Sobrevivência (%)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# %%
df = df.drop(columns=["FareCategory"])

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Pclass', hue='Survived', palette={0: 'red', 1: 'green'})

plt.title('Sobrevivência por Classe (Pclass) no Titanic')
plt.xlabel('Classe (Pclass)')
plt.ylabel('Contagem')
plt.legend(title='Sobreviveu', loc='upper right', labels=['Não', 'Sim'])
plt.xticks([0, 1, 2], ['1ª Classe', '2ª Classe', '3ª Classe'])
plt.grid(axis='y') 
plt.show()

# %%
lista_df = df.columns.to_list()

# %%
sns.boxplot(df)

# %%
df_num = df.select_dtypes(include=["int64", "float"])

# %%
correlation_matrix = df_num.corr()

# %%
plt.figure(figsize=(8, 6))  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            vmin=-1, vmax=1, linewidths=0.5)

# %%
y = df["Survived"]

# %%
features = df.columns.to_list()
cat_features = df.select_dtypes(include=["object"])
list_cat_features = cat_features.columns.to_list()
X = df[features]
X = X.drop(columns=["Survived"])
print(features)

# %%
features_teste = df_teste.columns.to_list()
cat_features_teste = df_teste.select_dtypes(include=["object"])
list_cat_features_teste = cat_features_teste.columns.to_list()
X_test_base = df_teste[features_teste]

print(features_teste)

# %%
onehot = encoding.OneHotEncoder(variables=list_cat_features)
onehot.fit(X)
X = onehot.transform(X)
X

# %%
X.isnull().sum()

# %%
onehot_teste = encoding.OneHotEncoder(variables=list_cat_features_teste)
onehot_teste.fit(X_test_base)
X_test_base = onehot_teste.transform(X_test_base)
X_test_base

# %%
X_test_base.isnull().sum()

# %%
X_test_base = X_test_base[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                          'Sex_male', 'Sex_female',
                          'Embarked_S', 'Embarked_C', 'Embarked_Q']]

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# %%
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# %%
y_pred = model_lr.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.3f}")

# %%
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()

# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree

# %%
model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(X_train, y_train)

# %%
y_pred = model_tree.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.3f}")

# %%
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# %%
y_pred = model_rf.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.3f}")

# %%
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()

# %%
test_predictions = model_lr.predict(X_test_base)

# %%
output = pd.DataFrame({
    "PassengerId": df_teste1["PassengerId"],
    "Survived": test_predictions
})
output.set_index('PassengerId', inplace=True)
output.to_csv('submission_kg.csv')
