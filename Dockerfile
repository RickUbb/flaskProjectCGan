# Usar una imagen base con Python
FROM python:3.12-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos de tu aplicación al directorio de trabajo
COPY . .

# Instalar las dependencias especificadas en el archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que Flask estará corriendo (opcional, depende del puerto en el que corras Flask)
EXPOSE 5000

# Definir la variable de entorno FLASK_APP para que apunte a tu archivo principal
ENV FLASK_APP=app.py

# Comando para ejecutar la aplicación cuando el contenedor se inicie
CMD ["flask", "run", "--host=0.0.0.0"]
