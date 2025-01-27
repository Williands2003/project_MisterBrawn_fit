import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#biblioteca de exercicios
class MisterBrawn:
    def __init__(self):
        print("Eu sou o Mister Brawn! E vou moldar você em uma máquina!")
        self.treinos_library = {
            1: "Flexão de braço",
            2: "Agachamento livre",
            3: "Prancha abdominal",
            4: "Burpee",
            5: "Polichinelos",
            6: "Abdominais básicos",
            7: "Mountain climbers",
            8: "Ponte de glúteos",
            9: "Corrida no lugar",
            10: "Salto com agachamento",
            11: "Elevação de pernas",
            12: "Cadeira na parede",
            13: "Afundo alternado",
            14: "Saltos laterais",
            15: "Superman",
            16: "L-sit",
            17: "Escaladores",
            18: "Dips em cadeira",
            19: "Pistol squat",
            20: "Push-up com aplauso",
            21: "Abdominal em V",
            22: "Plank jacks",
            23: "Passada lateral",
            24: "High knees",
            25: "Russian twist",
            26: "Bear crawl",
            27: "Saltos verticais",
            28: "Jumping lunges",
            29: "Ponte com uma perna",
            30: "Push-up diamante",
            31: "Abdominal bicicleta",
            32: "Side plank",
            33: "Sprint no lugar",
            34: "Agachamento sumô",
            35: "Levantamento lateral de perna",
            36: "Side lunges",
            37: "Skaters",
            38: "Toe taps",
            39: "Burpee com salto alto",
            40: "Push-up com apoio em um braço",
            41: "Abdominal reverso",
            42: "Plank to push-up",
            43: "Levantamento de panturrilha",
            44: "Step-ups em cadeira",
            45: "Climbers cruzados",
            46: "Hollow body hold",
            47: "Side plank com elevação de perna",
            48: "Agachamento com explosão",
            49: "Incline push-up",
            50: "Plank com rotação"
        }

    def treinos(self, categoria):
        treinos = [self.treinos_library[i+1] for i in range(50)]
        planos = {
            'A': treinos[:10],
            'A;B': treinos[:20],
            'A;B;C': treinos[:30],
            'A;B;C;D': treinos[:40]
        }
        return planos.get(categoria, "Plano inválido! Escolha entre A, A;B, A;B;C ou A;B;C;D!")

# Dados simulados
np.random.seed(42)
n = 1000
dados = {
    'idade': np.random.randint(18, 60, size=n),
    'peso': np.random.randint(50, 120, size=n),
    'altura': np.random.randint(150, 200, size=n),
    'atividade': np.random.randint(1, 6, size=n),
    'objetivo': np.random.choice([0, 1, 2], size=n),
    'dieta': np.random.choice(['low carb', 'high protein', 'vegetarian'], size=n, p=[0.4, 0.4, 0.2])
}

# DataFrame
df = pd.DataFrame(dados)
X = df[['idade', 'peso', 'altura', 'atividade', 'objetivo']]
y = df['dieta']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelos
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)

# Hiperparâmetros para Grid Search
parametros = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=random_forest, param_grid=parametros, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

melhor_modelo = grid_search.best_estimator_
melhor_modelo.fit(X_train, y_train)

# Avaliação
y_pred = melhor_modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
relatorio = classification_report(y_test, y_pred)
matriz_confusao = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {acuracia * 100:.2f}%")
print("\nRelatório de Classificação:")
print(relatorio)
print("\nMatriz de Confusão:")
print(matriz_confusao)

# Teste com novos dados
novos_dados = pd.DataFrame({
    'idade': [25, 45, 30],
    'peso': [70, 85, 60],
    'altura': [175, 160, 168],
    'atividade': [3, 2, 4],
    'objetivo': [0, 2, 1]
})

previsoes = melhor_modelo.predict(novos_dados)
for i, previsao in enumerate(previsoes):
    print(f"Usuário {i + 1}: Dieta sugerida - {previsao}")

# Treinos
mister_brawn = MisterBrawn()
planos_treino = ['A', 'A;B', 'A;B;C', 'A;B;C;D']
for plano in planos_treino:
    print(f"Plano de treino {plano}: {mister_brawn.treinos(plano)}")
