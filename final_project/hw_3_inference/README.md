# Итоговый проект: Интеллектуальная система улучшения изображений

Система автоматического улучшения фотографий (яркость, контраст, цветовой баланс, насыщенность) на основе MIT-Adobe FiveK датасета (Expert C). Сравниваются image-adaptive 3D LUT (~68K параметров) и U-Net с MobileNetV2 (~2.3M параметров).

## Структура проекта

```
hw_3_inference/
├── model/
│   ├── dataset.py            # FiveKDataset с аугментациями
│   ├── utils.py              # device, seeds, метрики, loss, classical methods
│   ├── lut_model.py          # Learnable3DLUT (~68K параметров)
│   ├── unet_model.py         # U-Net с MobileNetV2 (~2.3M параметров)
│   ├── train.py              # Обучение 3D LUT
│   ├── train_unet.py         # Обучение U-Net
│   ├── evaluate.py           # Единая оценка всех моделей
│   ├── analyze_errors.py     # Анализ ошибок по категориям
│   └── export_onnx.py        # Экспорт в ONNX для Triton
├── clients/                  # HTTP/gRPC клиенты для Triton
├── triton/model_repository/  # Конфигурация Triton Inference Server
├── profiling/                # Нагрузочное тестирование
├── scripts/
│   └── inline_onnx_weights.py
├── Dockerfile
├── docker-compose.yml
├── docker-compose.gpu.yml
└── README.md
```

## Быстрый старт

### 1. Установка зависимостей

```bash
cd hw_3_inference
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
```

### 2. Загрузка датасета

```bash
# Датасет MIT-Adobe FiveK с Kaggle
python -c "import kagglehub; kagglehub.dataset_download('weipengzhang/adobe-fivek')"
# Путь: ~/.cache/kagglehub/datasets/weipengzhang/adobe-fivek/versions/1
```

### 3. Обучение моделей

```bash
cd model

# 3D LUT (основная модель) — ~12 часов на CPU, ~8 часов на MPS
python train.py --data-root ~/.cache/kagglehub/datasets/weipengzhang/adobe-fivek/versions/1 \
  --loss combined --epochs 100 --batch-size 8

# U-Net MobileNetV2 (для сравнения) — ~50 мин на CPU, ~30 мин на MPS
python train_unet.py --data-root ~/.cache/kagglehub/datasets/weipengzhang/adobe-fivek/versions/1 \
  --epochs 30 --batch-size 4
```

Чекпойнты сохраняются в:
- `checkpoints/best_lut_improved.pth` (3D LUT, combined loss)
- `checkpoints_unet/best_unet.pth` (U-Net)
- `checkpoints/hparams.json` и `checkpoints_unet/hparams.json` (гиперпараметры)

### 4. Оценка всех моделей

```bash
cd model
python evaluate.py --data-root ~/.cache/kagglehub/datasets/weipengzhang/adobe-fivek/versions/1 --all
```

Результаты сохраняются в `evaluation_results.json`.

### 5. Анализ ошибок

```bash
cd model
python analyze_errors.py \
  --data-root ~/.cache/kagglehub/datasets/weipengzhang/adobe-fivek/versions/1 \
  --checkpoint checkpoints/best_lut_improved.pth
```

Результаты: `error_analysis_summary.csv`, `error_analysis_per_image.csv`, `error_analysis/best_worst/`.

### 6. Итоговый ноутбук

Открыть `../final_project/final_project_notebook.ipynb` — загружает результаты оценки и строит таблицы/графики.

## Требования

- **Python:** 3.13
- **PyTorch:** 2.11.0
- **torchvision:** 0.26.0
- **Операционная система:** macOS (CPU/MPS) или Linux (CPU/CUDA)
- **Docker:** для Triton Inference Server (опционально)

Проверка версий:
```bash
python --version    # Python 3.13.x
python -c "import torch; print(torch.__version__)"  # 2.11.0
python -c "import torchvision; print(torchvision.__version__)"  # 0.26.0
```

## Развертывание через Triton Inference Server

```bash
# Экспорт модели в ONNX
cd model
python export_onnx.py --checkpoint checkpoints/best_lut_improved.pth \
  --out ../triton/model_repository/image_enhancer/1/model.onnx

# Запуск Triton
cd ..
docker compose build --no-cache
docker compose up -d

# Проверка
curl http://localhost:8000/v2/health/ready
python clients/client_http.py --image test_photo.jpg --output enhanced.jpg
```

## Кастомные метрики Triton

Прокси-модель `metrics_proxy` добавляет 2 метрики:
- `image_enhancer_processing_time_seconds` (COUNTER)
- `image_enhancer_current_requests` (GAUGE)

Эндпоинт Prometheus: `http://localhost:8002/metrics`

## Нагрузочное тестирование

```bash
# Через Docker SDK контейнер
docker compose run --rm sdk bash -lc \
  'perf_analyzer -m image_enhancer -u triton:8001 --concurrency-range 1:4:1'

# Или с хоста
cd profiling && URL=localhost:8001 bash run_perf_analyzer.sh
```

## Воспроизводимость

- **Seed:** 42 (torch, numpy, random, PYTHONHASHSEED)
- **Разбиение:** 80/10/10 (train/val/test, по алфавиту файлов)
- **Размер изображений:** 512x512
- **Аугментации:** horizontal flip (50%), brightness jitter (0.9-1.1), contrast jitter (0.9-1.1)
- **Гиперпараметры** сохраняются в `hparams.json` рядом с чекпойнтом

## Результаты Triton (CPU, macOS)

| Конкурентность | Avg Latency (ms) | P95 Latency (ms) | Throughput (inf/s) |
|---|---|---|---|
| 1 | 48.1 | 57.5 | 20.8 |
| 4 | 710.3 | 4467.5 | 1.4 |
| 8 | 318.8 | 482.6 | 3.1 |

## Проверка работоспособности
![img.png](img.png)
