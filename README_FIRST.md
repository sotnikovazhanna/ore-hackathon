# Первый запуск

1. Откройте папку проекта в VS Code.
2. Создайте виртуальное окружение:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Установите зависимости:

```powershell
pip install -r requirements.txt
```

4. Откройте `01_data_audit.ipynb`.
5. В первой конфигурационной ячейке замените `DATA_DIR` на путь к корню скачанного датасета.
6. Выполните `Run All`.

Исходные изображения не изменяются. Результаты появятся в `outputs/audit`.
