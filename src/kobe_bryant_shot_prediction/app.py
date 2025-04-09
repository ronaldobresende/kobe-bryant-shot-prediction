import streamlit as st
import requests

# URL do modelo MLflow
MLFLOW_URL = "http://localhost:5001/invocations"

# Campos esperados pelo modelo
FEATURE_NAMES = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"]

st.title("🏀 Previsão de Arremesso do Kobe Bryant")

st.markdown("Preencha os dados abaixo para prever se o arremesso foi convertido ou não.")

# Formulário de entrada
lat = st.number_input("Latitude", value=34.0)
lon = st.number_input("Longitude", value=-118.0)
minutes_remaining = st.number_input("Minutos Restantes", value=5)
period = st.selectbox("Período do Jogo", options=[1, 2, 3, 4], index=0)
playoffs = st.selectbox("É Playoff?", options=["Não", "Sim"])
shot_distance = st.number_input("Distância do Arremesso (ft)", value=15)

# Convertendo playoffs para 0 ou 1
playoffs_binary = 1 if playoffs == "Sim" else 0

# Botão para enviar
if st.button("Prever"):
    # Preparar o payload
    payload = {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": [[lat, lon, minutes_remaining, period, playoffs_binary, shot_distance]]
        }
    }

    #st.write("Payload enviado:", payload)

    try:
        response = requests.post(MLFLOW_URL, json=payload, headers={"Content-Type": "application/json"})
        #st.write("Resposta do servidor:", response.text)

        if response.status_code == 200:
            # Acessar a predição corretamenteResposta do servido
            prediction = response.json()["predictions"][0]
            if prediction == 1:
                st.success("✅ Acertou, mais um arremesso pra conta do Kobe!")
            else:
                st.warning("❌ Infelizmente essa o kobe errou.")
        else:
            st.error(f"Erro na requisição: {response.status_code}\nDetalhes: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Erro ao conectar ao servidor MLflow. Verifique se ele está em execução.")
    except Exception as e:
        st.error(f"Erro inesperado: {str(e)}")