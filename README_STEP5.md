# Шаг 5 — улучшенная сегментация

1. Скопируйте `05_train_talc_lraspp.ipynb` в корень проекта.
2. Используйте `.venv (Python 3.11)`.
3. Дополнительные зависимости обычно не нужны, потому что PyTorch и
   torchvision уже установлены.
4. Нажмите `Run All`.

На CPU обучение может занять заметное время. На GPU будет существенно быстрее.

Результаты:
- `outputs/talc_lraspp/segmentation_summary.json`
- `outputs/talc_lraspp/test_review.png`
- `models/talc_lraspp_mobilenet_v3.pth`
