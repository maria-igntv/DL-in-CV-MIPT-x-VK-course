# Интеллектуальная система улучшения изображений

## Описание проекта

**Задача:** supervised image-to-image преобразование. На вход — RAW-фото, на выход — улучшенное фото в стиле профессионального ретушера (эксперт C).

**Датасет:** [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) — 5000 RAW-изображений + 5 экспертных ретушей.

**Ключевое требование:** инференс < 500 мс, модель лёгкая для массового применения.

**Основная модель:** Image-adaptive 3D LUT (~68K параметров, инференс ~20 мс на CPU).

**Альтернативная модель:** U-Net с MobileNetV2 энкодером (~2.87M параметров, инференс ~246 мс на CPU).

**Продакшн-развертывание:** Triton Inference Server (ONNX Runtime) с динамическим батчингом, HTTP/gRPC API, кастомными метриками Prometheus.


## Требования

- Python 3.10+
- PyTorch 2.0+
- OpenCV, NumPy, scikit-image, lpips, pandas, matplotlib
- Docker + Docker Compose (для Triton)
- Датасет MIT-Adobe FiveK (~8 GB)

Установка зависимостей:

```bash
# Для обучения и оценки
pip install -r hw_3_inference/requirements-train.txt

# Для клиентов Triton
pip install -r hw_3_inference/requirements-client.txt

# Полный набор
pip install -r hw_3_inference/requirements.txt
```

## Быстрый старт

Минимальный путь от данных до работающего сервиса:

```bash
cd hw_3_inference

# 1. Обучение 3D LUT
python model/train.py --data-root /path/to/adobe-fivek --epochs 40 --loss combined

# 2. Оценка
python model/evaluate.py --data-root /path/to/adobe-fivek --all

# 3. Экспорт в ONNX
python model/export_onnx.py --checkpoint model/checkpoints/best_lut_improved.pth

# 4. Запуск Triton
docker compose up --build

# 5. Тестирование (в другом терминале)
python clients/client_http.py --image test_photo.jpg --output enhanced.jpg
```

## Пошаговая инструкция

### 1. Подготовка данных

Скачайте датасет MIT-Adobe FiveK. Рекомендуется версия с Kaggle:

```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('weipengzhang/adobe-fivek')"
```


### 2. Обучение модели 3D LUT

```bash
cd hw_3_inference/model

# Основной вариант (combined loss: L1 + 0.1*LPIPS)
python train.py \
  --data-root /path/to/adobe-fivek \
  --epochs 40 \
  --batch-size 8 \
  --loss combined \
  --lr 1e-3 \
  --augment \
  --seed 42
```

Результаты сохраняются в `checkpoints/`:
- `best_lut_improved.pth` — лучшая модель
- `history.json` — кривые обучения
- `hparams.json` — гиперпараметры

**Параметры модели:**
- LUT-размер: 17³
- Количество LUT: 3
- Параметры CNN: 3→16→32→64 каналов
- Общее число параметров: ~68K

**Примечание:** на macOS с MPS обучение принудительно выполняется на CPU, так как MPS не поддерживает `grid_sampler_3d_backward`, необходимый для 3D LUT.

### 3. Обучение U-Net (опционально)

```bash
cd hw_3_inference/model

python train_unet.py \
  --data-root /path/to/adobe-fivek \
  --epochs 30 \
  --batch-size 4 \
  --loss combined \
  --freeze-encoder-epochs 10
```

Результаты в `checkpoints_unet/`:
- `best_unet.pth` — лучшая модель
- Progressive unfreezing: первые 10 эпох энкодер заморожен, затем разморозка с lr×0.1.

### 4. Оценка качества

Унифицированный скрипт оценивает все модели на одной тестовой выборке (последние 10% датасета):

```bash
cd hw_3_inference/model

# Оценить всё (RAW, классика, 3D LUT, U-Net)
python evaluate.py --data-root /path/to/adobe-fivek --all

# Только 3D LUT
python evaluate.py --data-root /path/to/adobe-fivek --lut-checkpoint checkpoints/best_lut_improved.pth
```

Результат: `evaluation_results.json` с метриками SSIM, PSNR, LPIPS, latency, FPS для каждой модели.

### 5. Анализ ошибок

Классификация ошибок по категориям контента изображения:

```bash
cd hw_3_inference/model

# Для 3D LUT
python analyze_errors.py \
  --data-root /path/to/adobe-fivek \
  --checkpoint checkpoints/best_lut_improved.pth \
  --model-type lut

# Для U-Net
python analyze_errors.py \
  --data-root /path/to/adobe-fivek \
  --checkpoint checkpoints_unet/best_unet.pth \
  --model-type unet
```

Результаты в `error_analysis/`:
- `error_analysis_summary.csv` — средние метрики по категориям
- `error_analysis_per_image.csv` — метрики для каждого изображения
- `best_worst/` — визуальные примеры (склейка RAW | Expert | Prediction)

**Категории анализа:**
- **Brightness:** night / twilight / day / overexposed
- **Color temp:** natural / warm_artificial / cool_artificial
- **Saturation:** low / medium / high


## Архитектура

### 3D LUT (основная модель)

**Принцип:** LUT-таблицы инициализированы близко к тождественному отображению. CNN предсказывает, как смешать 3 базовые LUT для каждого конкретного изображения. Trilinear interpolation гарантирует, что геометрия изображения не изменится — меняются только цвета пикселей.

### U-Net (альтернативная модель)

- **Encoder:** MobileNetV2 (pretrained on ImageNet), 5 уровней понижения разрешения
- **Decoder:** 5 уровней повышения с skip connections
- **Bottleneck:** Conv 1×1 (1280→256)
- **Выход:** Sigmoid → [0, 1]
- **Параметры:** ~2.87M

## Результаты

Тестовая выборка: 500 изображений (последние 10% датасета).

| Метод | SSIM | PSNR (дБ) | LPIPS | Параметры | Latency (мс) | FPS |
|-------|------|-----------|-------|-----------|--------------|-----|
| RAW (без обработки) | 0.648 ± 0.136 | 18.79 ± 3.48 | — | 0 | 0 | — |
| CLAHE | 0.546 ± 0.135 | 16.50 ± 2.54 | — | 0 | 0.8 | 1227 |
| Auto Gamma | 0.513 ± 0.205 | 15.71 ± 4.00 | — | 0 | 0.2 | 4703 |
| White Balance | 0.649 ± 0.136 | 19.03 ± 3.51 | — | 0 | 1.3 | 753 |
| Pipeline (WB+CLAHE) | 0.578 ± 0.137 | 17.45 ± 2.81 | — | 0 | 2.1 | 464 |
| **3D LUT (L1+LPIPS)** | **0.657 ± 0.140** | **19.81 ± 3.73** | **0.123 ± 0.066** | **67 996** | **19.2** | **52** |
| U-Net (MobileNetV2) | 0.674 ± 0.123 | 20.53 ± 3.30 | 0.129 ± 0.061 | 2 870 075 | 246.1 | 4.1 |

**Ключевые выводы:**
- 3D LUT превосходит RAW baseline по SSIM (+0.009) и PSNR (+1.02 дБ).
- Perceptual loss (LPIPS) критически важен: обучение только по L1 не даёт выигрыша над RAW.
- 3D LUT — оптимальный компромисс: 97.5% качества U-Net при 7.8% времени инференса и 2.4% размера модели.

## Воспроизводимость

Все гиперпараметры и настройки зафиксированы:

- **Seed:** 42 (torch, numpy, random, PYTHONHASHSEED)
- **Разбиение:** 80/10/10 (train/val/test) по алфавитному порядку файлов
- **Размер:** 512×512
- **Аугментации:** horizontal flip (50%), brightness jitter [0.9, 1.1], contrast jitter [0.9, 1.1]
- **Loss:** L1 + 0.1 × LPIPS
- **Optimizer:** Adam (lr = 1e-3)
- **Scheduler:** CosineAnnealingLR (T_max = 40)
- **Epochs:** 40 (3D LUT), 30 (U-Net)

Гиперпараметры сохранены в `checkpoints/hparams.json` и `checkpoints_unet/hparams.json`.
