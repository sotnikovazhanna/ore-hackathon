# Шаг 6 — talc gate и финальное приложение

## 1. Обучить talc gate

Скопируйте `06_train_talc_gate.ipynb` в корень проекта и выполните `Run All`
с kernel `.venv (Python 3.11)`.

После обучения должен появиться файл:

```text
models/talc_gate_mobilenet_v3_small.pth
```

## 2. Скопировать приложение

Скопируйте в корень проекта:

```text
app.py
src/ore_pipeline.py
requirements_final.txt
```

Убедитесь, что в `models/` находятся:

```text
ordinary_fine_mobilenet_v3_small.pth
talc_gate_mobilenet_v3_small.pth
talc_lraspp_mobilenet_v3.pth
```

## 3. Установить Streamlit и запустить

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_final.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Для первого демо лучше использовать обычное изображение среднего размера.
Панорама будет обрабатываться дольше, потому что анализ выполняется по тайлам.
