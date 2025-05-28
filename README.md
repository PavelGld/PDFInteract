# PDF Chat Assistant 📚🤖

Streamlit-приложение для интерактивного обсуждения PDF-документов с помощью LLM через RAG (Retrieval-Augmented Generation) и OpenRouter API.

## 🎯 Возможности

- **Загрузка PDF файлов** - поддержка drag-and-drop интерфейса
- **Извлечение текста** - автоматическая обработка PDF документов
- **Векторный поиск** - создание эмбеддингов через Course API
- **Интеллектуальный чат** - ответы на вопросы на основе содержимого PDF
- **Множественные LLM модели** - выбор между GPT-3.5, Claude 3, DeepSeek и другими
- **RAG система** - контекстно-зависимые ответы с цитированием источников

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Установка Python пакетов
uv add streamlit PyPDF2 requests scikit-learn numpy openai langchain langchain-community langchain-openai tiktoken chromadb faiss-cpu

# Или через pip
pip install streamlit PyPDF2 requests scikit-learn numpy openai langchain langchain-community langchain-openai tiktoken chromadb faiss-cpu
```

### Настройка переменных окружения

Создайте файл `.env` со следующими ключами:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
COURSE_API_KEY=your_course_api_key_here
```

### Запуск приложения

```bash
streamlit run app.py --server.port 5000
```

Приложение будет доступно по адресу: `http://localhost:5000`

## 📁 Структура проекта

```
├── app.py                 # Главный файл Streamlit приложения
├── pdf_processor.py       # Обработка и извлечение текста из PDF
├── vector_store.py        # Векторное хранилище и поиск
├── openrouter_client.py   # Клиент для OpenRouter API
├── utils.py              # Вспомогательные функции
├── utils2.py             # Course API интеграция
├── README.md             # Документация проекта
└── .streamlit/
    └── config.toml       # Конфигурация Streamlit
```

## 🔧 Архитектура системы

### 1. Обработка PDF (pdf_processor.py)
- Извлечение текста с помощью PyPDF2
- Разбиение на фрагменты по 1000 символов с перекрытием 200 символов
- Очистка и нормализация текста

### 2. Векторное хранилище (vector_store.py)
- Создание эмбеддингов через Course API
- Использование косинусного сходства для поиска
- Хранение векторов в памяти с NumPy

### 3. LLM интеграция (openrouter_client.py)
- Поддержка множественных моделей через OpenRouter
- Создание контекстных промптов с найденными фрагментами
- Управление токенами и ограничениями API

### 4. Пользовательский интерфейс (app.py)
- Streamlit интерфейс с drag-and-drop загрузкой
- История чата и управление сессиями
- Отображение статистики и отладочной информации

## 🛠️ Конфигурация

### Streamlit настройки (.streamlit/config.toml)

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

### Поддерживаемые LLM модели (13 моделей)

- **OpenAI**: GPT-4o, GPT-4o Mini, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku  
- **Google**: Gemini Pro 1.5, Gemini Flash 1.5
- **Meta**: Llama 3.1 405B, Llama 3.1 70B
- **Mistral**: Mixtral 8x7B
- **Qwen**: Qwen 2.5 72B

## 📊 Рабочий процесс

1. **Загрузка PDF** → Пользователь загружает документ
2. **Извлечение текста** → PyPDF2 обрабатывает файл
3. **Создание фрагментов** → Текст разбивается на части
4. **Векторизация** → Course API создает эмбеддинги
5. **Индексация** → Векторы сохраняются для поиска
6. **Запрос пользователя** → Вопрос обрабатывается
7. **Поиск контекста** → Находятся релевантные фрагменты
8. **Генерация ответа** → LLM создает ответ на основе контекста

## 🔒 Безопасность

- API ключи хранятся в переменных окружения
- Валидация размера файлов (максимум 200 МБ)
- Обработка ошибок и исключений
- Безопасная очистка временных файлов

## 🐛 Отладка

В интерфейсе отображается отладочная информация:
- Количество обработанных фрагментов
- Статус векторного хранилища
- Найденные релевантные фрагменты
- Размер контекста для LLM

## 📝 Лицензия

Apache License Version 2.0

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для функции (`git checkout -b feature/AmazingFeature`)
3. Зафиксируйте изменения (`git commit -m 'Add some AmazingFeature'`)
4. Отправьте в ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📧 Поддержка

При возникновении проблем создайте issue в репозитории GitHub.

---

**Создано с ❤️ для удобного изучения документов с помощью ИИ**
