# bibliotecas essenciais.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib

# Classe que encapsula a lógica do personal trainer virtual, Mister Brawn.
class MisterBrawn:
    def __init__(self):
        print("Eu sou o Mister Brawn! Vamos treinar pesado e de forma inteligente!")
        # Base de conhecimento com 60 exercícios para a montagem dos planos.
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
        # Dicas rápidas de alimentação associadas a cada tipo de dieta.
        self.sugestoes_dieta = {
            'low_carb': "Foque em carnes magras, peixes, ovos, folhas verdes e gorduras saudáveis como abacate e nozes.",
            'high_protein': "Priorize frango, carne vermelha magra, peixes, ovos, laticínios e leguminosas como feijão e lentilha.",
            'vegetariana': "Consuma uma variedade de tofu, tempeh, lentilhas, grão de bico, sementes de chia, e quinoa para garantir a proteína."
        }



    

    # Monta um plano de treino semanal, distribuindo os exercícios de forma aleatória.
    def montar_plano(self, categoria, dias=3):
        treinos = list(self.treinos_library.values())
        # Mapeia a categoria de condicionamento (A, A;B, etc.) para um conjunto de exercícios.
        planos_base = {
            'A': treinos[:15],
            'A;B': treinos[:30],
            'A;B;C': treinos[:45],
            'A;B;C;D': treinos[:60]
        }


        
        exercicios_selecionados = planos_base.get(categoria, [])
        if not exercicios_selecionados:
            return "Plano inválido."

        plano_semanal = {}
        exercicios_por_dia = len(exercicios_selecionados) // dias
        for i in range(dias):
            inicio = i * exercicios_por_dia
            fim = inicio + exercicios_por_dia
            # Seleciona 5 exercícios aleatórios para cada dia de treino, sem repetição no mesmo dia.
            plano_semanal[f'Dia {i+1}'] = np.random.choice(exercicios_selecionados[inicio:fim], size=5, replace=False).tolist()
        return plano_semanal

    # Retorna uma sugestão de cardápio com base na dieta prevista.
    def sugerir_cardapio(self, tipo_dieta):
        return self.sugestoes_dieta.get(tipo_dieta, "Tipo de dieta não encontrado.")

# Fixa a semente para garantir a reprodutibilidade dos resultados na geração de dados.
np.random.seed(42)
n_usuarios = 1000

# Geração de um dataset sintético para simular uma população de usuários com diferentes perfis.
dados = {
    'idade': np.random.randint(18, 65, n_usuarios),
    'peso_kg': np.random.uniform(50, 120, n_usuarios).round(1),
    'altura_m': np.random.uniform(1.50, 2.00, n_usuarios).round(2),
    'nivel_atividade': np.random.randint(1, 6, n_usuarios),
    'objetivo': np.random.choice(['perder_peso', 'manter_peso', 'ganhar_massa'], n_usuarios),
    'dieta_alvo': np.random.choice(['low_carb', 'high_protein', 'vegetariana'], n_usuarios, p=[0.4, 0.4, 0.2]),
    'genero': np.random.choice(['M', 'F'], n_usuarios, p=[0.5, 0.5])
}
df = pd.DataFrame(dados)
# cálculo do Índice de Massa Corporal (IMC).
df['imc'] = df['peso_kg'] / (df['altura_m'] ** 2)

# Transformando variáveis textuais em numéricas para o modelo.
df['objetivo_cod'] = pd.Categorical(df['objetivo']).codes
df['genero_cod'] = pd.Categorical(df['genero']).codes
df['faixa_etaria'] = pd.cut(df['idade'], bins=[17, 30, 50, 65], labels=['jovem', 'adulto', 'senior'])
df['faixa_etaria_cod'] = pd.Categorical(df['faixa_etaria']).codes

# Seleção das features (variáveis independentes) e do alvo (variável dependente).
X = df[['idade', 'peso_kg', 'altura_m', 'nivel_atividade', 'objetivo_cod', 'imc', 'genero_cod', 'faixa_etaria_cod']]
y = df['dieta_alvo']

# Padronização das features para que tenham média 0 e desvio padrão 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Divisão dos dados em conjuntos de treino e teste, com estratificação para manter a proporção das classes.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Instanciação do modelo com balanceamento de classes para lidar com dados desiguais.
modelo = RandomForestClassifier(random_state=42, class_weight='balanced')

# Definição do grid de hiperparâmetros para a busca exaustiva (GridSearch).
parametros_grid = {
    'n_estimators': [150, 250],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'criterion': ['gini', 'entropy']
}

# Configuração e execução do GridSearchCV para encontrar a melhor combinação de parâmetros.
grid_search = GridSearchCV(modelo, parametros_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# O melhor modelo encontrado pelo GridSearch.
melhor_modelo = grid_search.best_estimator_
print(f"\nMelhores Parâmetros: {grid_search.best_params_}")

# Avaliação final do modelo otimizado com os dados de teste.
y_pred = melhor_modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do Modelo Otimizado: {acuracia * 100:.2f}%\n")
print("Relatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred, target_names=melhor_modelo.classes_))
# Validação cruzada para verificar a estabilidade do modelo.
scores = cross_val_score(melhor_modelo, X_scaled, y, cv=5)
print(f"Acurácia da Validação Cruzada: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*2*100:.2f}%)")

# Visualização da performance do modelo através da matriz de confusão.
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='viridis',
            xticklabels=melhor_modelo.classes_, yticklabels=melhor_modelo.classes_, annot_kws={"size": 14})
plt.title('Matriz de Confusão do Modelo Final', fontsize=16)
plt.xlabel('Rótulo Previsto', fontsize=12)
plt.ylabel('Rótulo Verdadeiro', fontsize=12)
plt.show()

# Análise da importância de cada feature para as previsões do modelo.
importancias = melhor_modelo.feature_importances_
indices = np.argsort(importancias)[::-1]
plt.figure(figsize=(12, 8))
sns.barplot(x=importancias[indices], y=[X.columns[i] for i in indices], palette='rocket')
plt.xlabel('Importância Relativa', fontsize=12)
plt.title('Importância das Features na Previsão da Dieta', fontsize=16)
plt.show()

# Persistência do modelo e do scaler treinados para uso futuro em produção.
joblib.dump(melhor_modelo, 'modelo_dieta_fitness_v2.pkl')
joblib.dump(scaler, 'scaler_dieta_v2.pkl')
print("Modelo salvo como 'modelo_dieta_fitness_v2.pkl'")
print("Scaler salvo como 'scaler_dieta_v2.pkl'")

# Exemplo de como carregar os artefatos salvos para fazer novas previsões.
modelo_carregado = joblib.load('modelo_dieta_fitness_v2.pkl')
scaler_carregado = joblib.load('scaler_dieta_v2.pkl')
print("Modelo e Scaler carregados com sucesso!")

# Função que processa os dados de um novo usuário e retorna a recomendação.
def prever_e_recomendar(dados_usuario):
    df_usuario = pd.DataFrame([dados_usuario])
    # Aplica as mesmas transformações de features que foram usadas no treino.
    df_usuario['imc'] = df_usuario['peso_kg'] / (df_usuario['altura_m'] ** 2)
    df_usuario['objetivo_cod'] = pd.Categorical(df_usuario['objetivo'], categories=['perder_peso', 'manter_peso', 'ganhar_massa']).codes
    df_usuario['genero_cod'] = pd.Categorical(df_usuario['genero'], categories=['M', 'F']).codes
    df_usuario['faixa_etaria'] = pd.cut(df_usuario['idade'], bins=[17, 30, 50, 65], labels=['jovem', 'adulto', 'senior'])
    df_usuario['faixa_etaria_cod'] = pd.Categorical(df_usuario['faixa_etaria'], categories=['jovem', 'adulto', 'senior']).codes
    
    # Padroniza os dados do usuário com o scaler carregado.
    usuario_scaled = scaler_carregado.transform(df_usuario[['idade', 'peso_kg', 'altura_m', 'nivel_atividade', 'objetivo_cod', 'imc', 'genero_cod', 'faixa_etaria_cod']])
    previsao_dieta = modelo_carregado.predict(usuario_scaled)
    return previsao_dieta[0]

# Simulação de um novo usuário para teste do sistema completo.
novo_usuario_dados = {
    'idade': 34, 'peso_kg': 88, 'altura_m': 1.78, 'nivel_atividade': 4,
    'objetivo': 'perder_peso', # Objetivo é perder peso.
    'genero': 'M'
}



dieta_sugerida = prever_e_recomendar(novo_usuario_dados)
print("\n--- Previsão Integrada para Novo Usuário ---")
print(f"Dieta Sugerida: {dieta_sugerida.upper()}")

mister_brawn = MisterBrawn()
plano_sugerido = mister_brawn.montar_plano('A;B;C', dias=4)
cardapio_sugerido = mister_brawn.sugerir_cardapio(dieta_sugerida)
print("\n--- Plano de Treino Personalizado (4 dias) ---")
for dia, exercicios in plano_sugerido.items():
    print(f"{dia}: {', '.join(exercicios)}")

print("\n--- Recomendação Nutricional ---")
print(cardapio_sugerido)
