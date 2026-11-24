import math
import random
import json
import os

MEMORY_FILE = "memory.json"

# 1. Функция активации (Сигмоида)
def sigmoid(x):
    return 1 / (1 + math.exp(-max(-50, min(50, x))))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Базовые знания при самом первом запуске
DEFAULT_DATA = {
    "vocab": ["привет", "здравствуй", "ку", "как", "дела", "кто", "ты", "пока", "до", "свидания"],
    "responses": [
        "Привет! Рад тебя видеть. Чем займемся?",
        "Я linesAI — самообучающаяся нейросеть, которую ты создал с нуля!",
        "До встречи! Моя память сохранена, жду следующей сессии."
    ],
    "training": [
        ["привет", 0],
        ["здравствуй", 0],
        ["ку", 0],
        ["как дела", 1],
        ["кто ты", 1],
        ["пока", 2],
        ["до свидания", 2]
    ]
}

# Загрузка памяти с диска (или создание новой)
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_DATA

def save_memory(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_words(text):
    for ch in "?!.,:;()\"'":
        text = text.replace(ch, "")
    return text.lower().split()

def text_to_vector(text, vocab):
    words = clean_words(text)
    return [1.0 if word in words else 0.0 for word in vocab]

# 3. Функция полного обучения нейросети на накопленных знаниях
def train_network(data):
    vocab = data["vocab"]
    num_classes = len(data["responses"])
    input_size = len(vocab)
    hidden_size = max(8, input_size)  # Скрытый слой растет вместе со словарем!
    output_size = num_classes

    random.seed(42)
    w_ih = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
    w_ho = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]

    learning_rate = 0.5
    epochs = 3000

    for _ in range(epochs):
        for phrase, class_idx in data["training"]:
            inputs = text_to_vector(phrase, vocab)
            target = [1.0 if i == class_idx else 0.0 for i in range(output_size)]

            # Прямой проход
            hidden = [sigmoid(sum(inputs[i] * w_ih[i][j] for i in range(input_size))) for j in range(hidden_size)]
            outputs = [sigmoid(sum(hidden[j] * w_ho[j][k] for j in range(hidden_size))) for k in range(output_size)]

            # Ошибка и обратное распространение (Backpropagation)
            out_errors = [target[k] - outputs[k] for k in range(output_size)]
            out_deltas = [out_errors[k] * sigmoid_derivative(outputs[k]) for k in range(output_size)]

            hid_errors = [sum(out_deltas[k] * w_ho[j][k] for k in range(output_size)) for j in range(hidden_size)]
            hid_deltas = [hid_errors[j] * sigmoid_derivative(hidden[j]) for j in range(hidden_size)]

            # Обновление связей (весов)
            for j in range(hidden_size):
                for k in range(output_size):
                    w_ho[j][k] += learning_rate * out_deltas[k] * hidden[j]
            for i in range(input_size):
                for j in range(hidden_size):
                    w_ih[i][j] += learning_rate * hid_deltas[j] * inputs[i]

    return w_ih, w_ho

# --- ЗАПУСК СИСТЕМЫ ---
memory = load_memory()
print("=== Инициализация нейросети linesAI 2.0 ===")
print(f"Нейронов в словаре: {len(memory['vocab'])} | Тем для ответа: {len(memory['responses'])}")
print("Обучение нейронных связей...")
w_ih, w_ho = train_network(memory)
print("Готово! Мозг нейросети настроен.\n")
print("Подсказка: напиши 'выход', чтобы закрыть, или 'статус', чтобы увидеть размер мозга.\n")

while True:
    user_text = input("Ты: ").strip()
    if not user_text:
        continue
    if user_text.lower() == "выход":
        save_memory(memory)
        print("linesAI: Память сохранена в memory.json. До связи!")
        break
    if user_text.lower() == "статус":
        total_weights = len(memory["vocab"]) * max(8, len(memory["vocab"])) + max(8, len(memory["vocab"])) * len(memory["responses"])
        print(f"linesAI [Статус]: Слов в памяти: {len(memory['vocab'])}, Параметров (связей): {total_weights}\n")
        continue

    vec = text_to_vector(user_text, memory["vocab"])
    known_words = sum(vec) > 0

    best_idx = 0
    confidence = 0.0

    if known_words:
        hidden_size = len(w_ho)
        output_size = len(memory["responses"])
        hidden = [sigmoid(sum(vec[i] * w_ih[i][j] for i in range(len(memory["vocab"])))) for j in range(hidden_size)]
        outputs = [sigmoid(sum(hidden[j] * w_ho[j][k] for j in range(hidden_size))) for k in range(output_size)]
        best_idx = outputs.index(max(outputs))
        confidence = outputs[best_idx] * 100

    # Если слова незнакомые или уверенность ниже 65% — просим пользователя научить!
    if not known_words or confidence < 65.0:
        print(f"linesAI: Я пока не уверен, как на это ответить (уверенность {confidence:.1f}%).")
        teach = input("Хочешь научить меня правильному ответу на эту фразу? (да/нет): ").strip().lower()
        if teach == "да":
            new_reply = input("Введи ответ, который я должен запомнить: ").strip()
            # Добавляем новые слова в словарь нейросети
            for w in clean_words(user_text):
                if w not in memory["vocab"]:
                    memory["vocab"].append(w)
            # Добавляем новый вариант ответа, если такого еще нет
            if new_reply not in memory["responses"]:
                memory["responses"].append(new_reply)
            resp_idx = memory["responses"].index(new_reply)
            memory["training"].append([user_text, resp_idx])
            save_memory(memory)

            print("linesAI: Перестраиваю нейроны и переобучаюсь...")
            w_ih, w_ho = train_network(memory)
            print("linesAI: Готово! Я запомнил это навсегда.\n")
        else:
            print("linesAI: Хорошо, пропустим.\n")
    else:
        print(f"linesAI ({confidence:.1f}%): {memory['responses'][best_idx]}\n")
