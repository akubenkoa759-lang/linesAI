import random
import json
import os

MEMORY_FILE = "gen_memory.json"

# 1. Загрузка или создание языковой модели
def load_brain():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"transitions": {}, "starters": []}

def save_brain(brain):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(brain, f, ensure_ascii=False, indent=2)

# 2. Обучение на тексте: ИИ запоминает, какие слова идут друг за другом и с каким весом
def learn_text(brain, text):
    # Разбиваем текст на предложения
    sentences = text.replace("!", ".").replace("?", ".").split(".")
    for s in sentences:
        words = s.strip().split()
        if len(words) < 2:
            continue
        
        # Запоминаем, с каких слов может начинаться мысль
        first_word = words[0]
        if first_word not in brain["starters"]:
            brain["starters"].append(first_word)
            
        # Строим связи между соседними словами (нейронные переходы)
        for i in range(len(words) - 1):
            curr_w = words[i].lower()
            next_w = words[i + 1]
            if curr_w not in brain["transitions"]:
                brain["transitions"][curr_w] = {}
            # Увеличиваем вес связи между этими двумя словами
            brain["transitions"][curr_w][next_w] = brain["transitions"][curr_w].get(next_w, 0) + 1

    save_brain(brain)

# 3. Генерация нового ответа слово за словом на основе контекста пользователя
def generate_reply(brain, user_text, max_words=15):
    if not brain["transitions"]:
        return "Мой мозг пока пуст! Напиши мне пару предложений, чтобы я обучился."

    # Ищем зацепку в словах пользователя
    user_words = [w.lower().strip("?!.,") for w in user_text.split()]
    valid_anchors = [w for w in user_words if w in brain["transitions"]]

    if valid_anchors:
        current = random.choice(valid_anchors)
        result = [current.capitalize()]
    else:
        current = random.choice(brain["starters"])
        result = [current]

    # Генерируем следующие слова, выбирая их с учетом весов (вероятностей)
    for _ in range(max_words - 1):
        key = current.lower()
        if key not in brain["transitions"]:
            break
        
        next_options = brain["transitions"][key]
        words = list(next_options.keys())
        weights = list(next_options.values())
        
        # Случайный выбор с учетом силы связи (веса)
        next_word = random.choices(words, weights=weights, k=1)[0]
        result.append(next_word)
        current = next_word

    return " ".join(result) + "."

# --- ЗАПУСК ГЕНЕРАТИВНОГО ИИ ---
brain = load_brain()
print("=== Генеративная нейросеть linesAI 3.0 ===")
print("Теперь я не повторяю шаблоны, а сам строю фразы из твоих слов!")
print("Команды: 'выход' — закрыть, 'мозг' — сколько связей выучено.\n")

# Стартовый мини-словарь, если запускаем впервые
if not brain["transitions"]:
    starter_corpus = (
        "Привет я твой искусственный интеллект linesAI. "
        "Я умею учиться на твоих сообщениях и придумывать новые ответы. "
        "Чем больше ты со мной общаешься тем умнее становится мой разум. "
        "Мы можем обсудить игры системы космос или программирование на Python."
    )
    learn_text(brain, starter_corpus)

while True:
    user_msg = input("Ты: ").strip()
    if not user_msg:
        continue
    if user_msg.lower() == "выход":
        print("linesAI: До связи! Все нейронные связи сохранены.")
        break
    if user_msg.lower() == "мозг":
        total_links = sum(len(v) for v in brain["transitions"].values())
        print(f"linesAI [Мозг]: Знаю слов: {len(brain['transitions'])}, Связей между ними: {total_links}\n")
        continue

    # 1. Сначала ИИ учится на твоем сообщении прямо в реальном времени!
    learn_text(brain, user_msg)

    # 2. Затем генерирует собственный уникальный ответ
    reply = generate_reply(brain, user_msg)
    print(f"linesAI: {reply}\n")
