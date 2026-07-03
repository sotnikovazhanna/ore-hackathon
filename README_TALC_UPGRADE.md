# Улучшение талька: ручная разметка → U-Net → обновлённый сайт

Это обновление исправляет две главные проблемы текущего MVP:

1. `talc gate` обучался на целых изображениях, а сайт подавал ему тайлы. Теперь обычные микрофотографии классифицируются целиком, как во время тестирования.
2. Старая сегментация обучалась на шумных автоматически заполненных контурах. Теперь предусмотрена ручная полигональная разметка 42 экспертных изображений и новая U-Net.

## Что входит

```text
tools/prepare_talc_annotation_project.py
tools/annotate_talc.py
tools/merge_talc_annotations.py
tools/validate_talc_masks.py
tools/build_talc_training_manifest.py
07_train_talc_unet.py
src/talc_unet.py
src/ore_pipeline.py      # заменить текущий
app.py                   # заменить текущий
requirements_talc_upgrade.txt
```

## Шаг 0. Создать ветку

В корне проекта:

```powershell
git checkout -b improve-talc-manual
```

Скопируйте содержимое архива в корень репозитория с заменой `app.py` и `src/ore_pipeline.py`.

Установите зависимости:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_talc_upgrade.txt
```

## Шаг 1. Исправить `.gitignore`

В текущем `.gitignore` найдите строку:

```text
data/
```

Удалите её и вставьте содержимое файла `GITIGNORE_MANUAL_TALC.txt`.

Так исходный датасет не попадёт в Git, но ручные маски команды будут храниться в репозитории.

## Шаг 2. Создать 42 задачи разметки

Подставьте путь к корню скачанного датасета:

```powershell
.\.venv\Scripts\python.exe tools\prepare_talc_annotation_project.py `
  --data-root "C:\ПУТЬ\К\ДАТАСЕТУ" `
  --assignees "Катя,Жанна,Имя3,Имя4"
```

Появится:

```text
data/manual_talc/tasks.csv
```

Задачи распределятся между участницами примерно поровну.

Закоммитьте общий список:

```powershell
git add data/manual_talc/tasks.csv tools src app.py 07_train_talc_unet.py requirements_talc_upgrade.txt README_TALC_UPGRADE.md

git commit -m "Add manual talc annotation workflow"
git push -u origin improve-talc-manual
```

## Шаг 3. Каждая участница размечает свою часть

Пример для Кати:

```powershell
.\.venv\Scripts\python.exe tools\annotate_talc.py `
  --data-root "C:\ПУТЬ\К\ДАТАСЕТУ" `
  --assignee "Катя"
```

В окне:

- **левый клик** — добавить вершину полигона;
- **Enter** — замкнуть и заполнить текущий полигон;
- можно создать несколько полигонов на одном изображении;
- **правый клик** или `U` — убрать последнюю точку;
- `C` — очистить маску текущего изображения;
- `K` — взять старую автоматическую маску как основу;
- `A` — показать/скрыть старую автоматическую маску оранжевым;
- `S` — сохранить маску и перейти дальше;
- `N` — пропустить;
- `B` — предыдущее изображение;
- `Q` — выйти с сохранением статуса.

Правило разметки:

> Размечается вся экспертная область оталькования целиком. Если синяя линия выходит к краю, полигон продолжается по краю изображения. Если зон несколько, создаются несколько полигонов.

Результаты участницы:

```text
data/manual_talc/annotations/Катя/masks/
data/manual_talc/annotations/Катя/overlays/
data/manual_talc/status/status_Катя.csv
```

Каждая участница делает собственный commit и push. Папки разделены по именам, поэтому конфликтов быть не должно.

## Шаг 4. Объединить и проверить маски

После получения всех изменений:

```powershell
git pull

.\.venv\Scripts\python.exe tools\merge_talc_annotations.py

.\.venv\Scripts\python.exe tools\validate_talc_masks.py `
  --data-root "C:\ПУТЬ\К\ДАТАСЕТУ"
```

Проверьте страницы:

```text
data/manual_talc/review/review_01.jpg
...
```

Слева оригинал, справа синяя ручная маска. Неверные кадры снова открыть в аннотаторе и исправить.

## Шаг 5. Добавить hard negatives и построить manifest

Команда автоматически выберет 60 рядовых/тонких изображений, включая самые тёмные, и назначит им нулевую маску:

```powershell
.\.venv\Scripts\python.exe tools\build_talc_training_manifest.py `
  --data-root "C:\ПУТЬ\К\ДАТАСЕТУ" `
  --negative-count 60
```

Появится:

```text
data/manual_talc/talc_segmentation_manifest.csv
```

Split выполняется по целым изображениям и группам, а не по патчам.

## Шаг 6. Обучить новую U-Net

```powershell
.\.venv\Scripts\python.exe 07_train_talc_unet.py
```

На CPU это может быть долго. Быстрый прогон:

```powershell
.\.venv\Scripts\python.exe 07_train_talc_unet.py --head-epochs 1 --finetune-epochs 5
```

Полный рекомендуемый прогон:

```powershell
.\.venv\Scripts\python.exe 07_train_talc_unet.py --head-epochs 2 --finetune-epochs 12
```

Результат:

```text
models/talc_unet_resnet18.pth
outputs/talc_unet/summary.json
outputs/talc_unet/test_metrics.csv
```

Основные метрики:

- mean IoU;
- Dice/F1;
- Hausdorff;
- Hausdorff 95%;
- средняя абсолютная ошибка площади в процентных пунктах.

## Шаг 7. Запустить обновлённый сайт

После появления `models/talc_unet_resnet18.pth` приложение автоматически выберет новую модель вместо старой LR-ASPP:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

До обучения новой U-Net сайт:

- использует `talc gate` на целом изображении;
- показывает старую синюю маску только как экспериментальную;
- не использует её площадь для итогового класса.

После обучения ручной U-Net:

- площадь маски участвует в правиле `>10%`;
- `talc gate` остаётся вторым независимым сигналом;
- сайт явно пишет причину итогового решения.

## Шаг 8. Зафиксировать результат в Git

Не коммитьте сырые данные и `.venv`. Коммитьте код, ручные маски, manifest и итоговые небольшие отчёты:

```powershell
git add .
git status
git commit -m "Train talc segmentation on manually corrected masks"
git push
```

Перед `git add .` убедитесь, что в списке нет `data/raw` и `.venv`.

## Что считать успехом

Нельзя заранее гарантировать конкретную метрику на новых месторождениях. Практическая цель новой итерации:

- IoU заметно выше старого baseline `0.34`;
- ошибка площади значительно ниже старых `~20 п.п.`;
- отсутствие систематической заливки всей тёмной матрицы;
- корректная работа на светлых оталькованных примерах;
- воспроизводимый test на полностью отложенных изображениях.
