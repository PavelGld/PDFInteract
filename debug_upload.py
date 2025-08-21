"""
Диагностика проблем с загрузкой файлов в Streamlit
"""

import streamlit as st
import tempfile
import os

st.title("🔍 Диагностика загрузки файлов")

st.info("Эта страница поможет диагностировать проблемы с загрузкой PDF файлов")

# Тест базовой загрузки файлов
st.header("📁 Тест базовой загрузки")

uploaded_file = st.file_uploader(
    "Загрузите любой файл для тестирования",
    type=None,
    help="Попробуйте загрузить любой небольшой файл"
)

if uploaded_file is not None:
    st.success("✅ Файл успешно загружен!")
    
    # Показать информацию о файле
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Имя файла", uploaded_file.name)
    
    with col2:
        st.metric("Размер", f"{uploaded_file.size} байт")
    
    with col3:
        size_mb = uploaded_file.size / (1024 * 1024)
        st.metric("Размер (MB)", f"{size_mb:.2f}")
    
    # Попробуем прочитать файл
    try:
        file_content = uploaded_file.getvalue()
        st.success(f"✅ Содержимое файла успешно прочитано ({len(file_content)} байт)")
        
        # Показать первые несколько байт
        st.code(f"Первые 50 байт: {file_content[:50]}")
        
        # Попробуем создать временный файл
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            # Проверим, создался ли файл
            if os.path.exists(tmp_file_path):
                file_size = os.path.getsize(tmp_file_path)
                st.success(f"✅ Временный файл создан: {tmp_file_path} ({file_size} байт)")
                
                # Удалим временный файл
                os.unlink(tmp_file_path)
                st.info("🗑️ Временный файл удален")
            else:
                st.error("❌ Не удалось создать временный файл")
                
        except Exception as temp_error:
            st.error(f"❌ Ошибка при создании временного файла: {temp_error}")
        
    except Exception as read_error:
        st.error(f"❌ Ошибка при чтении файла: {read_error}")

# Тест специфично PDF
st.header("📄 Тест PDF файлов")

pdf_file = st.file_uploader(
    "Загрузите PDF файл для специального тестирования",
    type="pdf",
    help="Загрузите небольшой PDF файл для тестирования"
)

if pdf_file is not None:
    st.success("✅ PDF файл успешно загружен!")
    
    try:
        pdf_content = pdf_file.getvalue()
        
        # Проверим PDF заголовок
        if pdf_content.startswith(b'%PDF'):
            st.success("✅ Корректный PDF заголовок обнаружен")
            
            # Показать версию PDF
            header_line = pdf_content[:20].decode('utf-8', errors='ignore')
            st.info(f"📋 PDF заголовок: {header_line}")
        else:
            st.error("❌ Некорректный PDF заголовок")
            st.code(f"Первые 20 байт: {pdf_content[:20]}")
        
    except Exception as pdf_error:
        st.error(f"❌ Ошибка при обработке PDF: {pdf_error}")

# Тест системных ограничений
st.header("⚙️ Системная информация")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Streamlit конфигурация")
    st.code("""
    maxUploadSize: 200MB
    maxMessageSize: 200MB
    enableCORS: false
    enableXsrfProtection: false
    """)

with col2:
    st.subheader("Переменные окружения")
    
    env_vars = ["AITUNNEL_API_KEY", "OPENROUTER_API_KEY", "COURSE_API_KEY"]
    for var in env_vars:
        value = os.environ.get(var, "НЕ НАЙДЕН")
        if value != "НЕ НАЙДЕН":
            masked = value[:10] + "..." if len(value) > 10 else value
            st.text(f"{var}: {masked}")
        else:
            st.text(f"{var}: {value}")

st.divider()
st.info("💡 Если базовая загрузка файлов работает, но PDF Chat App не работает, проблема в обработке файлов, а не в самой загрузке.")