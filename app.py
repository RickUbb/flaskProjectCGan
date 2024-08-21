# Importar el recolector de basura para liberar memoria cuando sea necesario
import pkg_resources
from keras.layers import TFSMLayer
from tensorflow.keras.layers import LeakyReLU
import gc
# Importar Flask y métodos para manejar solicitudes HTTP
from flask import Flask, request, jsonify, render_template
import pandas as pd  # Importar pandas para manipulación de datos
import numpy as np  # Importar numpy para manejo de arrays
import pickle  # Importar pickle para cargar objetos guardados
import tensorflow as tf  # Importar TensorFlow para manejar modelos de deep learning

app = Flask(__name__)  # Crear una instancia de la aplicación Flask


# Listar todas las librerías y sus versiones
installed_packages = pkg_resources.working_set
for package in sorted(installed_packages, key=lambda x: x.project_name.lower()):
    print(f"{package.project_name}=={package.version}")


# Cargar el modelo y el scaler
discriminator = tf.saved_model.load('gan_models/discriminator_fold10')
with open('gan_models/scaler_gan.pkl', 'rb') as f:
    scaler_gan = pickle.load(f)

# Cargar las estadísticas de los clientes desde un archivo CSV
client_stats = pd.read_csv('client_stats_actualizado.csv')

# Definir las características que el modelo utiliza
FEATURES = ['Valor', 'Dia_Semana', 'Mes', 'Dia_Mes', 'Hora_seno', 'Hora_coseno',
            'Transaccion_PAGO TARJETA DE CREDITO', 'Transaccion_REMESAS',
            'Transaccion_TRANSFER BANCOS', 'Transaccion_TRANSFER EXTERNAS',
            'Transaccion_TRANSFER INTERNACIONAL', 'Transaccion_TRANSFER INTERNAS']


@app.route('/')  # Ruta principal que renderiza la página principal
def index():
    # Cargar los datos normales y obtener la lista de clientes únicos
    data = pd.read_csv('client_stats_actualizado.csv')
    top_clients = data['Cod_Cliente'].unique().tolist()
    # Renderizar la plantilla HTML con los clientes
    return render_template('index.html', top_clients=top_clients)


# Ruta para manejar predicciones de transacciones individuales
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Obtener los datos enviados desde el frontend

    # Preprocesar la hora para convertirla a segundos
    time_str = data['Horas']
    h, m, s = map(int, time_str.split(':'))
    seconds = h * 3600 + m * 60 + s

    # Preprocesar los datos de la transacción para ser enviados al modelo
    data_preprocessed = {
        'Valor': float(data['Valor']),
        'Dia_Semana': data['Dia_Semana'],
        'Mes': data['Mes'],
        'Dia_Mes': data['Dia_Mes'],
        'Hora_seno': np.sin(2 * np.pi * seconds / 86400),
        'Hora_coseno': np.cos(2 * np.pi * seconds / 86400),
        'Transaccion_PAGO TARJETA DE CREDITO': 1 if data['Transaccion'] == 'PAGO TARJETA DE CREDITO' else 0,
        'Transaccion_REMESAS': 1 if data['Transaccion'] == 'REMESAS' else 0,
        'Transaccion_TRANSFER BANCOS': 1 if data['Transaccion'] == 'TRANSFER BANCOS' else 0,
        'Transaccion_TRANSFER EXTERNAS': 1 if data['Transaccion'] == 'TRANSFER EXTERNAS' else 0,
        'Transaccion_TRANSFER INTERNACIONAL': 1 if data['Transaccion'] == 'TRANSFER INTERNACIONAL' else 0,
        'Transaccion_TRANSFER INTERNAS': 1 if data['Transaccion'] == 'TRANSFER INTERNAS' else 0
    }

    # Convertir los datos en un DataFrame y escalar las características
    data_df = pd.DataFrame([data_preprocessed])
    data_scaled = scaler_gan.transform(data_df)

    # Convertir el código del cliente a un formato que el modelo pueda usar (one-hot encoding)
    client_labels = np.array([int(data['Cod_Cliente'])])
    max_clients = 3685  # Ajustar al número máximo de clientes esperados por el modelo
    client_labels = np.clip(client_labels, 0, max_clients - 1)
    client_one_hot = tf.keras.utils.to_categorical(
        client_labels, num_classes=max_clients)

    # Realizar la predicción con el modelo de discriminador
    try:
        probability = discriminator([data_scaled.astype(np.float32), client_one_hot.astype(
            np.float32)], training=False).numpy().flatten()[0]
    except Exception as e:
        # Manejar errores durante la inferencia
        return jsonify({'error': str(e)}), 500

    # Definir el umbral para determinar si la transacción es anómala
    threshold = 0.4081540815408154
    outlier = bool(probability < threshold)  # Convertir a booleano de Python

    # Obtener las estadísticas del cliente actual
    stats = client_stats[client_stats['Cod_Cliente']
                         == int(data['Cod_Cliente'])]
    if not stats.empty:
        stats = stats.iloc[0]
        hora_frecuente = int(stats['Hora_Mas_Frecuente'])
        hora_hhmmss = f"{hora_frecuente // 3600:02d}:{
        (hora_frecuente % 3600) // 60:02d}:{hora_frecuente % 60:02d}"

        # Convertir el día y mes más frecuentes a su representación en letras
        dia_semana_map = ['Domingo', 'Lunes', 'Martes',
                          'Miércoles', 'Jueves', 'Viernes', 'Sábado']
        mes_map = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

        dia_semana_frecuente = dia_semana_map[int(
            stats['Dia_Semana_Mas_Frecuente'])]
        mes_frecuente = mes_map[int(stats['Mes_Mas_Frecuente']) - 1]

        # Evaluar si la transacción es inusual en valor, hora, día o mes de la transacción
        valor_anomalo = float(data['Valor']) > (
                stats['Media_Valor'] + 2 * stats['Std_Valor'])
        hora_inusual = abs(seconds - hora_frecuente) > 6 * \
                       3600  # Diferencia mayor a 6 horas
        tipo_transaccion_columna = f"Num_{
        data['Transaccion'].replace(' ', '_')}"
        tipo_transaccion_nuevo = stats.get(tipo_transaccion_columna, 0) == 0
        dia_inusual = int(data['Dia_Semana']) != int(
            stats['Dia_Semana_Mas_Frecuente'])
        mes_inusual = int(data['Mes']) != int(stats['Mes_Mas_Frecuente'])

        # Crear una lista de diferencias encontradas
        diferencias = []
        if valor_anomalo:
            diferencias.append(
                "El valor de esta transacción es significativamente mayor que el promedio histórico.")
        if hora_inusual:
            diferencias.append(
                "La hora de esta transacción es inusual comparada con las horas frecuentes anteriores.")
        if tipo_transaccion_nuevo:
            diferencias.append(f"Este tipo de transacción ({
            data['Transaccion']}) no es común para este cliente.")
        if dia_inusual:
            diferencias.append(f"Esta transacción se realiza en un día ({dia_semana_map[int(
                data['Dia_Semana'])]}) que no es el más frecuente para este cliente.")
        if mes_inusual:
            diferencias.append(f"Esta transacción se realiza en un mes ({
            mes_map[int(data['Mes']) - 1]}) que no es el más frecuente para este cliente.")

        # Generar una explicación detallada para el resultado
        diferencias_str = " ".join(
            [f"⚠️ {d}" for d in diferencias]) if diferencias else "Esta transacción es atípica en comparación con el comportamiento anterior del cliente."

        explanation = (
            f"<strong>Cliente {data['Cod_Cliente']}</strong> realizó una transacción de <strong>{
            float(data['Valor']):.2f} USD</strong>, "
            f"clasificada como {
            '<span style=\"color: red;\">Anómala</span>' if outlier else '<span style=\"color: green;\">Normal</span>'}.<br>"
            f"<ul>"
            f"<li>Valor máximo de transacción previo: <strong>{
            float(stats['Max_Valor']):.2f} USD</strong></li>"
            f"<li>Valor mediano de transacción: <strong>{
            float(stats['Mediana_Valor']):.2f} USD</strong></li>"
            f"<li>Valor promedio de transacción: <strong>{
            float(stats['Media_Valor']):.2f} USD</strong></li>"
            f"<li>Desviación estándar del valor: <strong>{
            float(stats['Std_Valor']):.2f} USD</strong></li>"
            f"<li>Número total de transacciones: <strong>{
            int(stats['Num_Transacciones'])}</strong></li>"
            f"<li>Hora más frecuente de transacción: <strong>{
            hora_hhmmss}</strong></li>"
            f"<li>Día más frecuente de transacción: <strong>{
            dia_semana_frecuente}</strong></li>"
            f"<li>Mes más frecuente de transacción: <strong>{
            mes_frecuente}</strong></li>"
            f"<li><strong>Nueva transacción comparada con historial:</strong></li>"
            f"{diferencias_str}"
            f"<li>Número de transacciones de PAGO TARJETA DE CREDITO: <strong>{
            int(stats.get('Num_PAGO_TARJETA_CREDITO', 0))}</strong></li>"
            f"<li>Número de transacciones de REMESAS: <strong>{
            int(stats.get('Num_REMESAS', 0))}</strong></li>"
            f"<li>Número de transacciones de TRANSFER BANCOS: <strong>{
            int(stats.get('Num_TRANSFER_BANCOS', 0))}</strong></li>"
            f"<li>Número de transacciones de TRANSFER EXTERNAS: <strong>{
            int(stats.get('Num_TRANSFER_EXTERNAS', 0))}</strong></li>"
            f"<li>Número de transacciones de TRANSFER INTERNACIONAL: <strong>{
            int(stats.get('Num_TRANSFER_INTERNACIONAL', 0))}</strong></li>"
            f"<li>Número de transacciones de TRANSFER INTERNAS: <strong>{
            int(stats.get('Num_TRANSFER_INTERNAS', 0))}</strong></li>"
            f"</ul>"
            f"<br><strong>¿Qué significa esta probabilidad?</strong><br>"
            f"La probabilidad calculada es {
            probability:.2f}. Este valor representa la confianza que tiene el modelo en que la transacción es normal. "
            f"Si la probabilidad es menor que {threshold:.2f}, se considera que la transacción es anómala. En este caso, como la probabilidad es {
            'menor' if outlier else 'mayor'} que el umbral, "
            f"la transacción ha sido clasificada como {
            'anómala' if outlier else 'normal'}."
        )
    else:
        # En caso de que no haya estadísticas para el cliente
        explanation = (
            f"<strong>Cliente {data['Cod_Cliente']}</strong> realizó una transacción de <strong>{
            float(data['Valor']):.2f} USD</strong>, "
            f"clasificada como {
            '<span style=\"color: red;\">Anómala</span>' if outlier else '<span style=\"color: green;\">Normal</span>'}.<br>"
            f"No se encontraron estadísticas históricas para este cliente."
        )

    return jsonify({
        'message': f"La transacción es {'anómala' if outlier else 'normal'}. Probabilidad: {probability:.2f}. {explanation}",
        'confirm_needed': outlier
    })


# Ruta para manejar la carga de archivos CSV
@app.route('/cargar_csv', methods=['POST'])
def cargar_csv():
    import math

    file = request.files['file']  # Obtener el archivo CSV desde la solicitud
    data = pd.read_csv(file)  # Leer el archivo CSV en un DataFrame

    # Validar que las columnas requeridas estén presentes en el CSV
    if not all(feature in data.columns for feature in FEATURES) or 'Cod_Cliente' not in data.columns:
        return jsonify({'message': 'El archivo CSV no contiene las columnas necesarias.'}), 400

    # Filtrar las columnas relevantes y escalar las características
    data = data[['Cod_Cliente'] + FEATURES]
    data_scaled = scaler_gan.transform(data[FEATURES])

    # Convertir los códigos de clientes a formato one-hot
    client_labels = data['Cod_Cliente'].astype(int).map(
        {label: i for i, label in enumerate(data['Cod_Cliente'].unique())}).values
    max_clients = 3685  # Ajustar al número máximo de clientes esperados por el modelo
    client_labels = np.clip(client_labels, 0, max_clients - 1)
    client_one_hot = tf.keras.utils.to_categorical(
        client_labels, num_classes=max_clients)

    try:
        # Procesar los datos en lotes para evitar problemas de memoria
        batch_size = 1000
        probabilities = []
        for start in range(0, len(data_scaled), batch_size):
            end = start + batch_size
            batch_data_scaled = data_scaled[start:end]
            batch_client_one_hot = client_one_hot[start:end]
            batch_probabilities = discriminator([batch_data_scaled.astype(
                np.float32), batch_client_one_hot.astype(np.float32)], training=False).numpy().flatten()
            probabilities.extend(batch_probabilities)

        # Liberar memoria después de procesar cada lote
        del batch_data_scaled, batch_client_one_hot, batch_probabilities
        gc.collect()
    except Exception as e:
        # Manejar errores durante la inferencia
        return jsonify({'error': str(e)}), 500

    # Convertir las probabilidades a un array de numpy
    probabilities = np.array(probabilities)
    # Agregar las probabilidades calculadas al DataFrame
    data['Probability'] = probabilities

    # Determinar si las transacciones son anómalas o no basado en el umbral
    threshold = 0.4081540815408154
    data['Outlier'] = data['Probability'] < threshold

    # Convertir las horas seno y coseno a formato HH:MM:SS
    def convert_to_time(hour_sine, hour_cosine):
        angle = math.atan2(hour_sine, hour_cosine)
        if angle < 0:
            angle += 2 * math.pi
        seconds = angle * (86400 / (2 * math.pi))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Aplicar la conversión de hora a todas las filas
    data['Hora'] = data.apply(lambda row: convert_to_time(
        row['Hora_seno'], row['Hora_coseno']), axis=1)

    # Calcular información adicional para la respuesta
    num_clientes_unicos = data['Cod_Cliente'].nunique()
    clientes_con_mas_anomalias = data[data['Outlier'] ==
                                      True]['Cod_Cliente'].value_counts().head(5).to_dict()
    clientes_con_mas_normales = data[data['Outlier'] ==
                                     False]['Cod_Cliente'].value_counts().head(5).to_dict()
    distribucion_transacciones = data[FEATURES[6:]].sum().to_dict()

    total_anomalos = len(data)
    anomalos_clasificadas_anomalas = len(data[data['Outlier'] == True])
    anomalos_clasificadas_normales = len(data[data['Outlier'] == False])
    porcentaje_anomalos = (
                                  anomalos_clasificadas_anomalas / total_anomalos) * 100

    worst_anomalies = data[data['Outlier'] == True].nsmallest(5, 'Probability')

    # Formatear las peores transacciones anómalas para la respuesta
    worst_anomalies_formatted = [
        {
            'Cliente': str(row['Cod_Cliente']),
            'Valor': f"${row['Valor']:.2f}",
            'Fecha': f"{row['Dia_Mes']}/{row['Mes']} (Día de la Semana: {row['Dia_Semana']})",
            'Hora': row['Hora'],
            'Tipo de Transacción': FEATURES[6:][[row[feature] for feature in FEATURES[6:]].index(True)],
            'Probabilidad': f"{row['Probability']:.4f}",
            'Anómalo': 'Sí' if row['Outlier'] else 'No'
        } for _, row in worst_anomalies.iterrows()
    ]

    # Crear la respuesta JSON
    response = {
        'Total de datos evaluados': total_anomalos,
        'Datos detectados como anómalos': anomalos_clasificadas_anomalas,
        'Datos detectados como normales': anomalos_clasificadas_normales,
        'Porcentaje de aciertos (anómalos)': porcentaje_anomalos,
        'Clientes Únicos': num_clientes_unicos,
        'Clientes con Más Anomalías': clientes_con_mas_anomalias,
        'Clientes con Más Normales': clientes_con_mas_normales,
        'Distribución de Transacciones': distribucion_transacciones,
        'Peores transacciones anómalas': worst_anomalies_formatted
    }

    # Liberar memoria del DataFrame después del procesamiento
    del data, data_scaled, client_one_hot, probabilities
    gc.collect()

    return jsonify(response)  # Devolver la respuesta como JSON


@app.route('/update_model', methods=['POST'])
def update_model():
    data = request.json  # Obtener los datos enviados desde el frontend

    # Preprocesar la hora para convertirla a segundos
    time_str = data['Horas']
    h, m, s = map(int, time_str.split(':'))
    seconds = h * 3600 + m * 60 + s

    # Preprocesar los datos de la transacción para ser añadidos al modelo
    data_preprocessed = {
        'Valor': float(data['Valor']),
        'Dia_Semana': data['Dia_Semana'],
        'Mes': data['Mes'],
        'Dia_Mes': data['Dia_Mes'],
        'Hora_seno': np.sin(2 * np.pi * seconds / 86400),
        'Hora_coseno': np.cos(2 * np.pi * seconds / 86400),
        'Transaccion_PAGO TARJETA DE CREDITO': 1 if data['Transaccion'] == 'PAGO TARJETA DE CREDITO' else 0,
        'Transaccion_REMESAS': 1 if data['Transaccion'] == 'REMESAS' else 0,
        'Transaccion_TRANSFER BANCOS': 1 if data['Transaccion'] == 'TRANSFER BANCOS' else 0,
        'Transaccion_TRANSFER EXTERNAS': 1 if data['Transaccion'] == 'TRANSFER EXTERNAS' else 0,
        'Transaccion_TRANSFER INTERNACIONAL': 1 if data['Transaccion'] == 'TRANSFER INTERNACIONAL' else 0,
        'Transaccion_TRANSFER INTERNAS': 1 if data['Transaccion'] == 'TRANSFER INTERNAS' else 0
    }

    client_id = int(data['Cod_Cliente'])
    # Asegurarse de que el código del cliente no supere el valor máximo permitido
    max_clients = client_stats['Cod_Cliente'].max() + 1
    if client_id >= max_clients:
        return jsonify({'error': f'El código de cliente {client_id} supera el máximo permitido de {max_clients}.'}), 400

    # Verificar si el cliente existe en las estadísticas
    client_stats_row = client_stats[client_stats['Cod_Cliente'] == client_id]
    if client_stats_row.empty:
        return jsonify({'message': 'No se encontraron estadísticas históricas para este cliente. No se realizó ninguna actualización.'}), 400

    client_stats_row = client_stats_row.iloc[0]

    # Actualizar los valores de estadísticas (media, desviación estándar, etc.)
    num_transacciones = client_stats_row['Num_Transacciones'] + 1
    media_valor_actualizada = (
                                      (client_stats_row['Media_Valor'] * client_stats_row['Num_Transacciones']) + data_preprocessed['Valor']) / num_transacciones
    varianza_actualizada = ((client_stats_row['Std_Valor'] ** 2) * client_stats_row['Num_Transacciones'] +
                            (data_preprocessed['Valor'] - media_valor_actualizada) ** 2) / num_transacciones
    std_valor_actualizada = np.sqrt(varianza_actualizada)

    client_stats.loc[client_stats['Cod_Cliente'] == client_id, 'Num_Transacciones'] = num_transacciones
    client_stats.loc[client_stats['Cod_Cliente'] == client_id, 'Media_Valor'] = media_valor_actualizada
    client_stats.loc[client_stats['Cod_Cliente'] == client_id, 'Std_Valor'] = std_valor_actualizada
    client_stats.loc[client_stats['Cod_Cliente'] == client_id, 'Max_Valor'] = max(
        client_stats_row['Max_Valor'], data_preprocessed['Valor'])

    # Actualizar el archivo CSV con las nuevas estadísticas del cliente
    client_stats.to_csv('client_stats_actualizado.csv', index=False)

    # Reentrenar el discriminador para actualizar su percepción de transacciones normales
    data_df = pd.DataFrame([data_preprocessed])
    data_scaled = scaler_gan.transform(data_df)
    client_labels = np.array([client_id])
    client_one_hot = tf.keras.utils.to_categorical(
        client_labels, num_classes=max_clients)

    # Reentrenamiento del discriminador
    try:
        discriminator.train_on_batch([data_scaled.astype(np.float32), client_one_hot.astype(np.float32)], np.array([1.0]))
    except Exception as e:
        return jsonify({'error': f'Error durante el reentrenamiento: {str(e)}'}), 500

    return jsonify({'message': 'El modelo ha sido actualizado con éxito y la transacción ha sido registrada como normal.'})


if __name__ == '__main__':
    app.run(debug=True)  # Iniciar la aplicación Flask en modo de depuración
