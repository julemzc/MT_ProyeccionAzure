#INSTALAR el ambiente

python -m venv .venv

.venv\Scripts\activate

.venv\Scripts\python.exe -m pip install --upgrade pip 

pip install -r requeriments.txt



# EJECUTAR el proyecto

cd MT_ProyeccionAzure

.venv\Scripts\activate

streamlit run azure_forecast_app.py



# Para trabajar con notebook
.venv\Scripts\activate
pip install jupyter ipykernel
python -m ipykernel install --user --name=env_costos_azure --display-name "Costos Azure"