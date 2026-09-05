# Fraud Detection

ML-проект для обнаружения мошеннических банковских транзакций с использованием методов машинного обучения.

Основная задача проекта — построить бинарный классификатор, способный обнаруживать мошеннические транзакции при сильном дисбалансе классов.

В проекте реализован полный ML pipeline:

* EDA и анализ данных;
* preprocessing;
* baseline-модель Logistic Regression;
* обучение и tuning LightGBM;
* подбор threshold;
* оценка на отложенной test выборке;
* интерпретация модели с помощью SHAP;
* REST API на FastAPI;
* Docker-контейнеризация;
* автоматические тесты.

---

## Цель проекта

Построить модель бинарной классификации, которая определяет, является ли банковская транзакция мошеннической.

Основная сложность задачи — сильный дисбаланс классов: мошеннические транзакции составляют лишь небольшую часть всех операций.

Поэтому основное внимание уделяется не `accuracy`, а метрикам:

* Precision;
* Recall;
* F1-score;
* ROC-AUC;
* PR-AUC.

Также отдельно исследуется влияние threshold на баланс между Precision и Recall.

---

## Данные

В проекте используется датасет **Credit Card Fraud Detection**.

Данные содержат:

* `Time` — время транзакции;
* `V1...V28` — анонимизированные признаки;
* `Amount` — сумма транзакции;
* `Class` — целевая переменная:

  * `0` — нормальная транзакция;
  * `1` — мошенническая транзакция.

Перед обучением были удалены полные дубликаты.

После очистки наблюдается сильный дисбаланс классов: мошеннические транзакции составляют около **0.17%** данных.

---

## Exploratory Data Analysis

В ходе EDA были исследованы:

* распределение целевой переменной;
* распределение `Amount`;
* распределение `Time`;
* распределения признаков `V1...V28`;
* корреляции между признаками;
* различия распределений признаков для нормальных и мошеннических транзакций.

Основные выводы:

* пропущенные значения отсутствуют;
* обнаружены и удалены полные дубликаты;
* присутствует сильный дисбаланс классов;
* `Amount` имеет асимметричное распределение и выбросы;
* некоторые признаки имеют заметно различающиеся распределения для двух классов.

Результаты EDA находятся в:

```text
notebooks/eda.ipynb
```

---

## Preprocessing

Данные были разделены на три выборки:

* **Train — 70%**
* **Validation — 15%**
* **Test — 15%**

Для сохранения распределения редкого класса использовалась стратификация.

Для Logistic Regression признаки были стандартизированы с помощью `StandardScaler`.

Для LightGBM масштабирование не применялось.

---

## Модели

### Logistic Regression

В качестве baseline использовалась Logistic Regression с балансировкой классов:

```python
LogisticRegression(
    random_state=42,
    class_weight="balanced"
)
```

Результаты на validation set:

* **PR-AUC:** 0.6971
* **ROC-AUC:** 0.9719

Logistic Regression использовалась как baseline для сравнения с более сложной моделью.

---

### LightGBM

В качестве основной модели использовался LightGBM с балансировкой классов:

```python
LGBMClassifier(
    n_estimators=1100,
    num_leaves=31,
    learning_rate=0.02,
    min_child_samples=20,
    reg_lambda=0,
    class_weight="balanced",
    random_state=42
)
```

LightGBM показал значительное улучшение по сравнению с baseline.

---

## Hyperparameter Tuning

Подбор гиперпараметров выполнялся последовательно, чтобы оценить влияние отдельных параметров модели.

Исследовались:

* `n_estimators`;
* `learning_rate`;
* `num_leaves`;
* `min_child_samples`;
* `reg_lambda`.

Основной метрикой для выбора конфигурации был **PR-AUC**, поскольку задача характеризуется сильным дисбалансом классов.

Лучшая конфигурация:

| Parameter           |    Value |
| ------------------- | -------: |
| `n_estimators`      |     1100 |
| `learning_rate`     |     0.02 |
| `num_leaves`        |       31 |
| `min_child_samples` |       20 |
| `reg_lambda`        |        0 |
| `class_weight`      | balanced |

Результаты экспериментов сохраняются в:

```text
results/lgbm_tuning.csv
```

---

## Model Evaluation

Для оценки модели использовались:

* **Precision** — доля действительно мошеннических транзакций среди всех предсказанных мошенническими;
* **Recall** — доля обнаруженных мошеннических транзакций;
* **F1-score** — гармоническое среднее Precision и Recall;
* **ROC-AUC** — способность модели разделять два класса;
* **PR-AUC** — качество классификации редкого положительного класса.

### PR-AUC

В задаче обнаружения мошенничества положительный класс является очень редким.

При таком дисбалансе высокая ROC-AUC не всегда означает хорошее качество обнаружения мошеннических транзакций.

Поэтому **PR-AUC использовалась как основная метрика при сравнении моделей и подборе конфигурации**.

---

## Threshold Tuning

Стандартный threshold `0.5` не является оптимальным для данной задачи.

Поэтому на validation set был проведён отдельный анализ различных threshold.

Для каждого целевого значения Recall подбирался threshold, позволяющий контролировать баланс между Precision и Recall.

| Target Recall | Threshold | Precision | Recall |         F1 |
| ------------: | --------: | --------: | -----: | ---------: |
|           70% |    0.9939 |   100.00% | 73.24% |     84.55% |
|           75% |    0.9281 |    98.18% | 76.06% | **85.71%** |
|           80% |    0.0149 |    89.06% | 80.28% |     84.44% |
|           85% |    0.0004 |    53.51% | 85.92% |     65.95% |
|           90% |  0.000005 |     6.99% | 90.14% |     12.97% |
|           95% | 0.0000001 |     0.77% | 95.77% |      1.54% |

Для финальной оценки был выбран threshold:

```text
0.9281
```

Threshold был выбран **только на validation set** и зафиксирован перед оценкой на test set.

---

## Final Test Results

Финальная оценка проводилась на отложенной test выборке, которая не использовалась для подбора гиперпараметров и threshold.

| Metric    |      Score |
| --------- | ---------: |
| Precision | **94.64%** |
| Recall    | **74.65%** |
| F1-score  | **83.46%** |
| ROC-AUC   | **97.58%** |
| PR-AUC    | **81.52%** |

### Confusion Matrix

```text
[[42485     3]
 [   18    53]]
```

Где:

* **TN = 42,485**
* **FP = 3**
* **FN = 18**
* **TP = 53**

Модель обнаружила **53 из 71 мошеннической транзакции**, при этом допустила только **3 ложных срабатывания** среди нормальных транзакций.

Финальные результаты сохраняются в:

```text
results/final_metrics.csv
```

---

## Model Interpretability

Для интерпретации LightGBM использовался **SHAP**.

Были выполнены:

* глобальный анализ важности признаков;
* SHAP beeswarm plot;
* анализ влияния признака `V4`;
* локальное объяснение мошеннической транзакции;
* SHAP waterfall plot.

### Global Feature Importance

Наиболее значимые признаки по среднему абсолютному SHAP:

1. `V4`
2. `V12`
3. `V14`
4. `V3`
5. `V5`
6. `V15`
7. `V8`
8. `V22`
9. `V11`
10. `V26`

SHAP показывает вклад признаков в предсказания модели, но не говорит о причинно-следственной связи между признаками и мошенничеством.

Результаты SHAP анализа сохраняются в:

```text
results/
├── shap_summary_beeswarm.png
├── shap_summary_bar.png
├── shap_v4_impact.png
├── shap_fraud_waterfall.png
├── shap_global_importance.csv
└── shap_fraud_explanation.csv
```

---

## REST API

Для получения предсказаний реализован REST API на **FastAPI**.

### Endpoints

#### `GET /health`

Проверка состояния сервиса.

Пример ответа:

```json
{
  "status": "ok"
}
```

#### `POST /predict`

Принимает данные транзакции и возвращает вероятность мошенничества и итоговый класс.

Пример ответа:

```json
{
  "fraud_probability": 0.000017213248433306597,
  "is_fraud": false
}
```

Используемый в API threshold:

```text
0.9281
```

Исходный код API:

```text
src/api.py
```

---

## Docker

API контейнеризирован с помощью Docker.

Для запуска:

```bash
docker compose up --build
```

После запуска API доступен по адресу:

```text
http://localhost:8000
```

Документация FastAPI доступна через Swagger UI:

```text
http://localhost:8000/docs
```

Docker Compose конфигурация:

```text
docker-compose.yml
```

---

## Testing

Для API реализованы автоматические тесты с использованием `pytest`.

Проверяются:

* доступность `/health`;
* корректность ответа `/predict`;
* наличие `fraud_probability`;
* наличие `is_fraud`;
* корректность диапазона вероятности;
* тип итогового предсказания.

Запуск тестов:

```bash
pytest
```

Текущий результат:

```text
2 passed
```

Тесты находятся в:

```text
tests/test_api.py
```

---

## Структура проекта

```text
fraud-detection/
├── data/
│   └── creditcard.csv
├── models/
│   └── lgbm_model.pkl
├── notebooks/
│   └── eda.ipynb
├── results/
│   ├── final_metrics.csv
│   ├── lgbm_tuning.csv
│   ├── model_comparison.csv
│   ├── threshold_results.csv
│   ├── shap_global_importance.csv
│   ├── shap_fraud_explanation.csv
│   └── *.png
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── train_lgbm.py
│   ├── tuning_lgbm.py
│   ├── threshold_tuning.py
│   ├── evaluate.py
│   ├── evaluate_lgbm.py
│   ├── final_evaluation.py
│   ├── feature_importance.py
│   ├── shap_analysis.py
│   └── api.py
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Технологический стек

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **LightGBM**
* **SHAP**
* **Matplotlib**
* **Jupyter Notebook**
* **FastAPI**
* **Pytest**
* **Docker**
* **Git**

---

## Как запустить проект

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск тестов

```bash
pytest
```

### 3. Запуск API локально

```bash
uvicorn src.api:app --reload
```

После запуска:

```text
http://localhost:8000/docs
```

### 4. Запуск через Docker

```bash
docker compose up --build
```

После запуска:

```text
http://localhost:8000/docs
```

