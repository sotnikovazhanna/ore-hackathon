# OreScope AI

Интерпретируемая система анализа OM-изображений полированных шлифов.

Приложение:

- классифицирует руду как рядовую, труднообогатимую или оталькованную;
- выделяет предполагаемые обычные и тонкие срастания;
- строит экспериментальную маску областей оталькования;
- показывает количественные показатели;
- экспортирует результат в CSV, JSON и PNG.

## Возможности

- загрузка TIFF, PNG и JPEG;
- тайловая обработка больших изображений;
- классификация `ordinary / fine`;
- классификация `talc_over_10 / non_talc`;
- цветной overlay:
  - зелёный — обычные срастания;
  - красный — тонкие срастания;
  - синий — экспериментальная область оталькования;
- карта уверенности;
- локальный запуск через Streamlit.

## Требования

Рекомендуемая версия Python: **3.11**.

Проверить установленную версию:

```powershell
py -3.11 --version
```

## Быстрый запуск на Windows

### 1. Клонировать репозиторий

```powershell
git clone <ССЫЛКА_НА_РЕПОЗИТОРИЙ>
cd ore-hackathon
```

### 2. Создать виртуальное окружение

```powershell
py -3.11 -m venv .venv
```

Если PowerShell запрещает запуск скриптов:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Активировать окружение:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Установить зависимости

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements_final.txt
```

### 4. Проверить модели

В папке `models/` должны находиться:

```text
models/
├── ordinary_fine_mobilenet_v3_small.pth
├── talc_gate_mobilenet_v3_small.pth
└── talc_lraspp_mobilenet_v3.pth
```

Если модели хранятся отдельно, скачайте их по ссылке, указанной командой проекта, и поместите в папку `models`.

### 5. Запустить приложение

```powershell
python -m streamlit run app.py
```

После запуска Streamlit откроет приложение в браузере. Обычно используется адрес:

```text
http://localhost:8501
```

## Запуск на Linux/macOS

```bash
git clone <ССЫЛКА_НА_РЕПОЗИТОРИЙ>
cd ore-hackathon

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements_final.txt
python -m streamlit run app.py
```

## Как пользоваться приложением

1. Загрузите исходное изображение формата TIFF, PNG или JPEG.
2. Оставьте параметры анализа по умолчанию для первого запуска.
3. Нажмите **«Запустить анализ»**.
4. Дождитесь завершения тайловой обработки.
5. Проверьте:
   - итоговый класс руды;
   - вероятность превышения порога оталькования;
   - долю обычных и тонких срастаний;
   - цветной overlay;
   - карту уверенности.
6. При необходимости скачайте CSV, JSON или PNG.

Для первого теста рекомендуется использовать микрофотографию среднего размера. Панорамные изображения обрабатываются дольше.

## Логика классификации

```text
Если модель обнаруживает превышение порога оталькования:
    оталькованная руда
Иначе:
    ordinary преобладает → рядовая руда
    fine преобладает → труднообогатимая руда
```

## Структура проекта

```text
ore-hackathon/
├── app.py
├── src/
│   └── ore_pipeline.py
├── models/
├── 01_data_audit.ipynb
├── 02_prepare_training_data.ipynb
├── 03_train_intergrowth_classifier.ipynb
├── 04_train_talc_segmenter.ipynb
├── 05_train_talc_lraspp.ipynb
├── 06_train_talc_gate.ipynb
├── requirements.txt
├── requirements_step3.txt
├── requirements_final.txt
└── README.md
```

## Ноутбуки

- `01_data_audit.ipynb` — аудит изображений, поиск дублей и анализ классов.
- `02_prepare_training_data.ipynb` — подготовка масок и очищенных manifest-файлов.
- `03_train_intergrowth_classifier.ipynb` — обучение `ordinary / fine`.
- `04_train_talc_segmenter.ipynb` — пиксельный baseline сегментации.
- `05_train_talc_lraspp.ipynb` — семантическая сегментация LR-ASPP.
- `06_train_talc_gate.ipynb` — классификация `talc_over_10 / non_talc`.

Ноутбуки нужны для воспроизведения экспериментов. Для обычного запуска приложения достаточно `app.py`, папки `src`, весов моделей и `requirements_final.txt`.

## Текущие результаты baseline

На отложенных тестовых выборках:

| Модуль | Метрика | Результат |
|---|---|---:|
| Ordinary / fine | Macro F1 | около 0.83 |
| Ordinary / fine | ROC-AUC | около 0.88 |
| Talc gate | Accuracy при пороге 0.5 | около 0.93 |
| Talc gate | ROC-AUC | 1.00 |
| Сегментация оталькования | Mean IoU | около 0.34 |
| Сегментация оталькования | Pixel F1 | около 0.50 |

## Ограничения

Проект является хакатонным MVP.

- Сегментационная маска оталькования является экспериментальной и может захватывать обычную тёмную нерудную матрицу.
- Точная оценка площади талька пока не достигает требуемой ошибки ±3%.
- Светлые оталькованные образцы могут классифицироваться хуже.
- Панорамные изображения отличаются по яркости и масштабу от обучающих микрофотографий.
- Для промышленного применения требуется дополнительная экспертная разметка, калибровка и валидация на новых шлифах.

## Если приложение не запускается

Проверить Python:

```powershell
.\.venv\Scripts\python.exe --version
```

Проверить основные библиотеки:

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision, streamlit, cv2; print('OK')"
```

Проверить наличие моделей:

```powershell
Get-ChildItem .\models
```

Запустить Streamlit через интерпретатор окружения:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Команда

Проект разработан в рамках хакатона по автоматическому анализу полированных шлифов.
