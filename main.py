# --- 1. Importações ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- 2. Classe do Treinador Virtual ---
class MisterBrawn:
    def __init__(self):
        print("Eu sou o Mister Brawn! Vamos treinar pesado!")
        # 60 exercícios variados para montar planos de treino de acordo com o nível da pessoa
        self.treinos_library = {
            1: "Flexão de braço", 2: "Agachamento livre", 3: "Prancha abdominal", 4: "Burpee",
            5: "Polichinelos", 6: "Abdominais básicos", 7: "Mountain climbers", 8: "Ponte de glúteos",
            9: "Corrida no lugar", 10: "Salto com agachamento", 11: "Elevação de pernas", 12: "Cadeira na parede",
            13: "Afundo alternado", 14: "Saltos laterais", 15: "Superman", 16: "L-sit",
            17: "Escaladores", 18: "Dips em cadeira", 19: "Pistol squat", 20: "Push-up com aplauso",
            21: "Abdominal em V", 22: "Plank jacks", 23: "Passada lateral", 24: "High knees",
            25: "Russian twist", 26: "Bear crawl", 27: "Saltos verticais", 28: "Jumping lunges",
            29: "Ponte com uma perna", 30: "Push-up diamante", 31: "Abdominal bicicleta", 32: "Side plank",
            33: "Sprint no lugar", 34: "Agachamento sumô", 35: "Levantamento lateral de perna", 36: "Side lunges",
            37: "Skaters", 38: "Toe taps", 39: "Burpee com salto alto", 40: "Push-up com apoio em um braço",
            41: "Abdominal reverso", 42: "Plank to push-up", 43: "Levantamento de panturrilha", 44: "Step-ups em cadeira",
            45: "Climbers cruzados", 46: "Hollow body hold", 47: "Side plank com elevação de perna", 48: "Agachamento com explosão",
            49: "Incline push-up", 50: "Plank com rotação", 51: "Remada curvada (com elástico)", 52: "Tríceps francês (com elástico)",
            53: "Good morning (sem peso)", 54: "Box jumps", 55: "Handstand push-up (na parede)", 56: "Dragon flag",
            57: "Glute kickback", 58: "Calf raises", 59: "Bird-dog", 60: "Dead bug"
        }

    def montar_plano(self, categoria):
        # Retorna o plano de treino baseado no nível informado
        treinos = list(self.treinos_library.values())
        planos = {
            'A': treinos[:15],          # Iniciante
            'A;B': treinos[:30],        # Intermediário
            'A;B;C': treinos[:45],      # Avançado
            'A;B;C;D': treinos[:60]     # Expert
        }
        return planos.get(categoria, "Plano inválido. Use A, A;B, A;B;C ou A;B;C;D.")

# --- 3. Gerando os Dados Fictícios ---
np.random.seed(42)  # Garante que os dados gerados sejam sempre iguais (reprodutível)
n_usuarios = 1000

# Cria dados simulando informações de pessoas diferentes
dados = {
    'idade': np.random.randint(18, 65, n_usuarios),
    'peso_kg': np.random.uniform(50, 120, n_usuarios).round(1),
    'altura_m': np.random.uniform(1.50, 2.00, n_usuarios).round(2),
    'nivel_atividade': np.random.randint(1, 6, n_usuarios),
    'objetivo': np.random.choice(['perder_peso', 'manter_peso', 'ganhar_massa'], n_usuarios),
    'dieta_alvo': np.random.choice(['low_carb', 'high_protein', 'vegetariana'], n_usuarios, p=[0.4, 0.4, 0.2])
}
df = pd.DataFrame(dados)

# --- 4. Ajustando os Dados ---
# Calcula o IMC (peso dividido pela altura ao quadrado)
df['imc'] = df['peso_kg'] / (df['altura_m'] ** 2)

# Converte o objetivo em número para o modelo poder entender
df['objetivo'] = pd.Categorical(df['objetivo']).codes

# Define as variáveis de entrada (X) e a saída (y)
X = df[['idade', 'peso_kg', 'altura_m', 'nivel_atividade', 'objetivo', 'imc']]
y = df['dieta_alvo']

# Divide os dados em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 5. Criando e Otimizando o Modelo ---
modelo = RandomForestClassifier(random_state=42)

# Testa várias combinações de parâmetros para achar a melhor
parametros_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(modelo, parametros_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

melhor_modelo = grid_search.best_estimator_

# --- 6. Avaliando o Modelo ---
y_pred = melhor_modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acuracia * 100:.2f}%\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# --- 7. Gráficos dos Resultados ---
# Matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=melhor_modelo.classes_, yticklabels=melhor_modelo.classes_)
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# Importância de cada variável usada pelo modelo
importancias = melhor_modelo.feature_importances_
indices = np.argsort(importancias)

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importancias[indices], color='blue')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Importância')
plt.title('Quais variáveis mais influenciam na dieta')
plt.show()

# --- 8. Salvando o Modelo ---
joblib.dump(melhor_modelo, 'modelo_dieta_fitness.pkl')
print("Modelo salvo como 'modelo_dieta_fitness.pkl'")

# Carregando novamente
modelo_carregado = joblib.load('modelo_dieta_fitness.pkl')
print("Modelo carregado com sucesso!")

# --- 9. Previsão com Novos Dados ---
novo_usuario = pd.DataFrame({
    'idade': [34],
    'peso_kg': [88],
    'altura_m': [1.78],
    'nivel_atividade': [4],
    'objetivo': [0],  # perder_peso
    'imc': [88 / (1.78 ** 2)]
})

previsao_dieta = modelo_carregado.predict(novo_usuario)
print("\n--- Previsão para Novo Usuário ---")
print(f"Dieta Sugerida: {previsao_dieta[0]}")

# --- 10. Sugestão de Treino ---
mister_brawn = MisterBrawn()
plano_sugerido = mister_brawn.montar_plano('A;B;C')
print("\n--- Plano de Treino Sugerido ---")
print(f"Plano com {len(plano_sugerido)} exercícios.")
print("Exercícios sugeridos:", plano_sugerido[:10])
