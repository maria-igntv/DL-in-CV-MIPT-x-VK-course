# ДЗ3. Инференс модели улучшения изображений

## Описание

Image-adaptive 3D LUT модель для автоматического улучшения фотографий, развёрнутая через Triton Inference Server.

Модель обучена на датасете MIT-Adobe FiveK (Expert C), использует 3 обучаемые 3D LUT-таблицы с весами, предсказываемыми лёгким CNN (~68K параметров).

## Быстрый старт

### Требования
- Docker + NVIDIA Container Toolkit
- NVIDIA GPU

### Запуск одной командой

```bash
docker compose up
```

Или:

```bash
docker build -t triton-image-enhancer .
docker run --gpus all --net host \
  -v $(pwd)/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### Проверка работоспособности

```bash
# Health check
curl http://localhost:8000/v2/health/ready

# Отправить тестовый запрос
python clients/client_http.py --image test_photo.jpg --output enhanced.jpg
```

## Кастомные метрики

Прокси-модель `metrics_proxy` добавляет 2 кастомные метрики:


* `total_processing_time` - (COUNTER) Общее время процессинга запросов (с)
* `current_requests`- (GAUGE) Текущее число обрабатываемых запросов

Метрики доступны через Prometheus endpoint: `http://localhost:8002/metrics`

Запросы к метрикам отправляются через модель `metrics_proxy`, которая через BLS вызывает `image_enhancer`.

## Нагрузочное тестирование

### Performance Analyzer

```bash
# Из SDK-контейнера:
docker run --net host nvcr.io/nvidia/tritonserver:24.01-py3-sdk \
  perf_analyzer -m image_enhancer -u localhost:8001 --concurrency-range 1:8:1
```

### Model Analyzer

```bash
# Из контейнера с моделью:
bash profiling/run_model_analyzer.sh
```

Конфигурация тестирует:
- max_batch_size: [1, 4, 8]
- instance_group count: [1, 2]
- concurrency range: 1-8

## Результаты

### Performance Analyzer

| Конкурентность | Avg Latency (ms) | P95 Latency (ms) | Throughput (inf/s) |
|---|---|---|---|
| 1 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |

### Model Analyzer

| Конфигурация | max_batch | instances | Avg Latency (ms) | Throughput (inf/s) | GPU Mem (MB) |
|---|---|---|---|---|---|
| Default | 8 | 1 | — | — | — |
| Small | 1 | 1 | — | — | — |
| Large | 8 | 2 | — | — | — |

### Сравнительный анализ

