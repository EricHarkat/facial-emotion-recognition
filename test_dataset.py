import pandas as pd

# Charger les données FER-2013 (CSV format)
# Télécharger le dataset directement sur Kaggle (kaggle FER-2013)
data = pd.read_csv('./fer2013.csv')

# Afficher les premieres lignes
print(data.head()) 