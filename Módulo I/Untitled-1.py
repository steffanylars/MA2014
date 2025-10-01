import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Cargar datos
data = pd.read_csv('spam.csv', encoding='latin-1')

# Limpieza de datos
data_clean = data
data_clean['spam'] = data_clean['v1'].map({'ham' : 0, 'spam' : 1})
data_clean = data_clean.drop(columns=['v1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
data_clean = data_clean.rename(columns={'v2' : 'email'})

# Procesamiento de texto
data_clean['email'] = data_clean['email'].apply(lambda x : x.lower())
data_clean['email'] = data_clean['email'].apply(lambda x : re.sub('[^a-z0-9 ]+', ' ', x))

# Remover stop words
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
stop_words = stopwords.words('english')

def remove_stop_words(message):
    words = word_tokenize(message)
    words = [word for word in words if word not in stop_words]
    return words

data_clean['email'] = data_clean['email'].apply(remove_stop_words)

# Stemming
Porter = PorterStemmer()
data_clean['email'] = data_clean['email'].apply(lambda x : [Porter.stem(word) for word in x])

# División en conjuntos de entrenamiento y prueba
train_set = data_clean.sample(frac=0.8, random_state=1337)
test_set = data_clean.drop(train_set.index)

# Calcular probabilidades a priori
p_spam = train_set[train_set['spam'] == 1].shape[0] / train_set.shape[0]
p_not_spam = train_set[train_set['spam'] == 0].shape[0] / train_set.shape[0]

# Función bag of words
def bag_of_words(corpus):
    bag_of_words = {}
    for message in corpus["email"]:
        for word in message:
            if word in bag_of_words:
                value = bag_of_words[word] 
                value = int(value) + 1
                bag_of_words.update({word:value})
            else:
                bag_of_words[word] = 1
    return bag_of_words

# Función para calcular probabilidades de palabras
def probability_words(df):
    bag_of_words_dict = bag_of_words(df)
    
    sum = 0 
    for value in bag_of_words_dict.values():
        sum += value
    
    probability_words = {}
    for word, frequency in bag_of_words_dict.items():
        probability = frequency / sum
        probability_words.update({word: probability})
    
    return probability_words

# Calcular probabilidades de palabras para spam y no spam
probability_spam_words = probability_words(train_set[train_set['spam'] == 1])
probability_non_spam_words = probability_words(train_set[train_set['spam'] == 0])

# FUNCIÓN CORREGIDA SIN SUAVIZADO DE LAPLACE
def classify_email(email):
    """
    Clasifica un email como spam (1) o no spam (0) usando Naive Bayes.
    Sin suavizado de Laplace - asigna probabilidad 0 a palabras no vistas.
    """
    # Para cada mensaje en el email
    result = []
    for message in email:
        # Calcular P(spam|mensaje) y P(no_spam|mensaje)
        # Usamos log para evitar underflow con muchas multiplicaciones pequeñas
        log_p_spam = np.log(p_spam)
        log_p_nonspam = np.log(p_not_spam)
        
        # Flag para verificar si alguna palabra no está en el vocabulario
        all_words_in_spam_vocab = True
        all_words_in_nonspam_vocab = True
        
        for word in message:
            # P(palabra|spam)
            if word in probability_spam_words:
                log_p_spam += np.log(probability_spam_words[word])
            else:
                all_words_in_spam_vocab = False
                log_p_spam = -np.inf  # Si una palabra no está, probabilidad es 0
                
            # P(palabra|no_spam)
            if word in probability_non_spam_words:
                log_p_nonspam += np.log(probability_non_spam_words[word])
            else:
                all_words_in_nonspam_vocab = False
                log_p_nonspam = -np.inf  # Si una palabra no está, probabilidad es 0
        
        # Si ambas probabilidades son 0 (log = -inf), clasificar como no spam por defecto
        if log_p_spam == -np.inf and log_p_nonspam == -np.inf:
            result.append(0)  # Default a no spam
        # Clasificar según cuál probabilidad es mayor
        elif log_p_spam > log_p_nonspam:
            result.append(1)
        else:
            result.append(0)
    
    return result

# Aplicar clasificación al conjunto de prueba
test_set_hat = test_set.copy()
test_set_hat['prediction'] = classify_email(test_set['email'].values.reshape(-1, 1))

# Función de métricas de desempeño
def performance_metrics(results):
    positives = results[['spam', 'prediction']][results['spam'] == 1]
    negatives = results[['spam', 'prediction']][results['spam'] == 0]

    true_negatives = negatives[negatives['spam'] == negatives['prediction']].shape[0]
    false_positives = negatives[negatives['spam'] != negatives['prediction']].shape[0]
    true_positives = positives[positives['spam'] == positives['prediction']].shape[0]
    false_negatives = positives[positives['spam'] != positives['prediction']].shape[0]

    confusion_matrix = {'actual positives' : [true_positives, false_negatives],
                        'actual negatives' : [false_positives, true_negatives]}

    confusion_matrix_df = pd.DataFrame.from_dict(confusion_matrix, orient='index',
                                                 columns=['predicted positives', 'predicted negatives'])

    # Evitar división por cero
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {'Accuracy' : accuracy, 'Precision' : precision, 'Recall' : recall, 'F1 Score' : f1_score}
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Metrics'])

    return confusion_matrix_df, metrics_df

# Evaluar el modelo
confusion_matrix, metrics = performance_metrics(test_set_hat)
print("Confusion Matrix:")
print(confusion_matrix)
print("\nMetrics:")
print(metrics)

# GENERAR NUEVOS MENSAJES

# Generate a spam email with a length of 20 words
print("\n=== SPAM EMAIL GENERADO ===")
spam_words = list(probability_spam_words.keys())
spam_probs = list(probability_spam_words.values())
# Normalizar las probabilidades para que sumen 1
spam_probs = np.array(spam_probs) / np.sum(spam_probs)

spam_message = np.random.choice(spam_words, size=20, p=spam_probs)
print(' '.join(spam_message))

# Generate a non spam email composed of 20 random words
print("\n=== NON-SPAM EMAIL GENERADO ===")
non_spam_words = list(probability_non_spam_words.keys())
non_spam_probs = list(probability_non_spam_words.values())
# Normalizar las probabilidades para que sumen 1
non_spam_probs = np.array(non_spam_probs) / np.sum(non_spam_probs)

non_spam_message = np.random.choice(non_spam_words, size=20, p=non_spam_probs)
print(' '.join(non_spam_message))