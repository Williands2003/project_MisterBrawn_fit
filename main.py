import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Classe que encapsula a lógica 
class MisterBrawn:
    def __init__(self):
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
        if dias == 0:
            return {}
            
        exercicios_por_dia = len(exercicios_selecionados) // dias
        
        # Garante que temos pelo menos 5 exercícios para escolher, se possível
        if exercicios_por_dia < 5:
            exercicios_por_dia = min(5, len(exercicios_selecionados))

        exercicios_disponiveis = exercicios_selecionados.copy()
        np.random.shuffle(exercicios_disponiveis) # Embaralha para variedade

        for i in range(dias):
            if len(exercicios_disponiveis) >= 5:
                 plano_semanal[f'Dia {i+1}'] = np.random.choice(exercicios_disponiveis, size=5, replace=False).tolist()
                 # Remove os exercícios já escolhidos para evitar repetição nos próximos dias
                 exercicios_disponiveis = [ex for ex in exercicios_disponiveis if ex not in plano_semanal[f'Dia {i+1}']]
            else:
                break
                
        return plano_semanal

    # Retorna uma sugestão de cardápio com base na dieta prevista.
    def sugerir_cardapio(self, tipo_dieta):
        return self.sugestoes_dieta.get(tipo_dieta, "Tipo de dieta não encontrado.")

# --- FUNÇÕES DE PREVISÃO  ---

# Cache para carregar o modelo e o scaler apenas uma vez
@st.cache_resource
def carregar_modelos(modelo_path, scaler_path):
    """Carrega o modelo e o scaler salvos, tratando exceções."""
    if not os.path.exists(modelo_path) or not os.path.exists(scaler_path):
        st.error(f"Erro: Arquivos de modelo ('{modelo_path}') ou scaler ('{scaler_path}') não encontrados. "
                 "Certifique-se de que eles estão no mesmo diretório que o seu script `app.py`.")
        return None, None
    try:
        modelo_carregado = joblib.load(modelo_path)
        scaler_carregado = joblib.load(scaler_path)
        return modelo_carregado, scaler_carregado
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os modelos: {e}")
        return None, None

# Função que processa os dados de um novo usuário e retorna a recomendação.
def prever_e_recomendar(dados_usuario, modelo, scaler):
    df_usuario = pd.DataFrame([dados_usuario])
    # Aplica as mesmas transformações de features que foram usadas no treino.
    df_usuario['imc'] = df_usuario['peso_kg'] / (df_usuario['altura_m'] ** 2)
    df_usuario['objetivo_cod'] = pd.Categorical(df_usuario['objetivo'], categories=['perder_peso', 'manter_peso', 'ganhar_massa']).codes
    df_usuario['genero_cod'] = pd.Categorical(df_usuario['genero'], categories=['M', 'F']).codes
    df_usuario['faixa_etaria'] = pd.cut(df_usuario['idade'], bins=[17, 30, 50, 65], labels=['jovem', 'adulto', 'senior'])
    df_usuario['faixa_etaria_cod'] = pd.Categorical(df_usuario['faixa_etaria'], categories=['jovem', 'adulto', 'senior']).codes
    
    # Padroniza os dados do usuário 
    features_ordem = ['idade', 'peso_kg', 'altura_m', 'nivel_atividade', 'objetivo_cod', 'imc', 'genero_cod', 'faixa_etaria_cod']
    usuario_scaled = scaler.transform(df_usuario[features_ordem])
    
    previsao_dieta = modelo.predict(usuario_scaled)
    return previsao_dieta[0]

# --- INTERFACE DO STREAMLIT ---


modelo_carregado, scaler_carregado = carregar_modelos('modelo_dieta_fitness_v2.pkl', 'scaler_dieta_v2.pkl')

st.set_page_config(page_title="Mister Brawn", page_icon="💪", layout="wide")

st.title("Mister Brawn: Seu Personal Trainer Virtual 💪")
st.write("Insira seus dados na barra lateral à esquerda e receba uma recomendação de dieta e um plano de treino personalizados!")

# Barra lateral para entrada de dados do usuário
st.sidebar.header("Insira seus dados")

idade = st.sidebar.slider("Idade", 18, 65, 30)
peso = st.sidebar.slider("Peso (kg)", 40.0, 150.0, 70.0, 0.5)
altura = st.sidebar.slider("Altura (m)", 1.40, 2.20, 1.75, 0.01)
genero = st.sidebar.radio("Gênero", ('M', 'F'), format_func=lambda x: 'Masculino' if x == 'M' else 'Feminino')
objetivo = st.sidebar.selectbox("Qual é o seu principal objetivo?", 
                                ('perder_peso', 'manter_peso', 'ganhar_massa'),
                                format_func=lambda x: x.replace('_', ' ').capitalize())
nivel_atividade = st.sidebar.slider("Nível de Atividade Física", 1, 5, 3, 
                                    help="1: Sedentário (pouco ou nenhum exercício) | 2: Levemente ativo (1-2 dias/semana) | 3: Moderadamente ativo (3-4 dias/semana) | 4: Muito ativo (5-6 dias/semana) | 5: Extremamente ativo (exercício intenso diário)")

# Botão para gerar o plano
if st.sidebar.button("Gerar meu plano!"):
    if modelo_carregado is not None and scaler_carregado is not None:
        
        # Cria o dicionário 
        dados_usuario = {
            'idade': idade, 
            'peso_kg': peso, 
            'altura_m': altura, 
            'nivel_atividade': nivel_atividade,
            'objetivo': objetivo,
            'genero': genero
        }

        # Calcula o IMC para exibição
        imc = dados_usuario['peso_kg'] / (dados_usuario['altura_m'] ** 2)
        st.subheader(f"Seu IMC é: {imc:.2f}")

        #  previsão da dieta
        dieta_sugerida = prever_e_recomendar(dados_usuario, modelo_carregado, scaler_carregado)

        # Instancia o Mister Brawn
        mister_brawn = MisterBrawn()
        
        # Lógica para definir plano e dias de treino
        if nivel_atividade <= 2:
            categoria_treino = 'A;B'
            dias_treino = 3
        elif nivel_atividade <= 4:
            categoria_treino = 'A;B;C'
            dias_treino = 4
        else:
            categoria_treino = 'A;B;C;D'
            dias_treino = 5

        plano_sugerido = mister_brawn.montar_plano(categoria_treino, dias=dias_treino)
        cardapio_sugerido = mister_brawn.sugerir_cardapio(dieta_sugerida)
        
        st.divider()

        # --- Exibição dos Resultados ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🥗 Recomendação Nutricional")
            st.success(f"**Dieta Sugerida:** {dieta_sugerida.replace('_', ' ').capitalize()}")
            st.info(f"**Dica do Mister Brawn:** {cardapio_sugerido}")

        with col2:
            st.subheader(f"🏋️ Plano de Treino Personalizado ({dias_treino} dias)")
            if not plano_sugerido:
                st.warning("Não foi possível gerar um plano de treino com os parâmetros fornecidos.")
            else:
                 for dia, exercicios in plano_sugerido.items():
                    with st.expander(f"**{dia}**"):
                        for ex in exercicios:
                            st.write(f"- {ex}")
        
        st.divider()
        st.balloons()
        st.success("Plano gerado com sucesso! Lembre-se de consultar um profissional de saúde e um educador físico.")

#seção "Sobre o Modelo" para dar contexto
st.divider()
with st.expander("Sobre o Modelo de IA e o Projeto"):
    st.write("""
    Este projeto utiliza um modelo de Machine Learning (RandomForestClassifier) treinado em um conjunto de dados sintético 
    de 1.000 usuários para prever o tipo de dieta mais adequado com base no perfil do indivíduo.
    
    As principais características (features) usadas para a previsão são: `idade`, `peso`, `altura`, `IMC`, `nível de atividade`, 
    `objetivo` e `gênero`.
    
    O plano de treino é gerado de forma procedural pela classe `MisterBrawn`, que seleciona exercícios de uma biblioteca 
    predefinida de acordo com o nível de condicionamento do usuário.
    
    **Disclaimer:** Esta é uma ferramenta de demonstração. As recomendações geradas são baseadas em um modelo estatístico e 
    não substituem a orientação de um nutricionista, médico ou educador físico profissional.
    """)
