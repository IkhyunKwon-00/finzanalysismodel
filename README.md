# Finz Analysis Model

**finzfinz의 랜덤 sentiment 함수를 대체할 멀티모달 딥러닝 주가 모멘텀 예측 모델**

## 아키텍처

```
┌──────────────┐   ┌──────────────────┐
│  News Text   │   │ Numerical Feats  │
│ (title+body) │   │ (stock + macro)  │
└──────┬───────┘   └───────┬──────────┘
       │                    │
┌──────▼───────┐   ┌───────▼──────────┐
│  DistilBERT  │   │   MLP Encoder    │
│  Text Enc.   │   │  (FC→GELU→FC)    │
└──────┬───────┘   └───────┬──────────┘
       │ (768-d)           │ (128-d)
       └─────────┬─────────┘
      ┌──────────▼──────────┐
      │  Cross-Attention    │
      │   Fusion Block      │
      └──────────┬──────────┘
                 │ (256-d)
      ┌──────────▼──────────┐
      │  Multi-Horizon Heads│
      │  30d / 180d / 360d  │
      │  (3-class + regress)│
      └─────────────────────┘
```

### Input
- **텍스트**: 영어 뉴스 제목 + 본문 (DistilBERT 인코딩)
- **수치 피처**: 종목 가격 롤링 피처 + 거시경제 지표 (S&P500, 금, WTI, BTC, VIX, DXY)

### Output (per horizon: 30d / 180d / 360d)
- **분류**: 긍정(2) / 중립(1) / 부정(0) — 3-class classification
- **회귀**: 예상 수익률 (%)
- **신뢰도**: softmax 확률 기반
- **핵심 문장 선별** (선택): Integrated Gradients 기반 문장별 인과 기여도 분석

### 문장 인과 분석 (Sentence Attribution)

`explain=true` 요청 시, 모델 예측에 가장 높은 인과적 영향을 미친 문장을 선별합니다.

- **방법**: Integrated Gradients (Sundararajan et al., 2017)
- 토큰별 gradient attribution → 문장 단위 집계 → 기여도 0~100 점수화
- Attention weight 단순 추출과 달리, 실제 출력에 대한 **인과적 기여도**를 측정
- Sensitivity, Implementation Invariance 공리를 만족하는 이론적으로 가장 엄밀한 방법

### 학습 데이터
| 소스 | 내용 |
|------|------|
| Yahoo Finance (yfinance) | 종목별 최근 뉴스 기사 (런타임 자동 수집) |
| Financial PhraseBank | 전문가 라벨링 금융 문장 ~5,000개 (`data/raw/financial_phrasebank.txt`) |
| CSV datasets | `data/raw/*.csv`에 드롭하면 자동 인식 (Kaggle 등) |
| yfinance | 종목별 OHLCV, 시가총액, 매출, 영업이익 |
| yfinance | S&P500, 금, WTI, BTC, VIX, 달러인덱스 |

> **참고**: Supabase 뉴스 데이터는 현재 학습에 사용하지 않습니다. 모든 뉴스는 위 외부 소스에서 수집됩니다.

## 프로젝트 구조

```
finzanalysismodel/
├── config/
│   └── default.yaml          # 전체 설정 (모델, 학습, 데이터, API)
├── src/
│   ├── config.py              # 설정 로더
│   ├── data/
│   │   ├── external_data.py   # 뉴스 수집 (Yahoo Finance + PhraseBank + CSV)
│   │   ├── news_collector.py  # (호환용 래퍼 → external_data.py)
│   │   ├── price_collector.py # yfinance 주가/매크로 수집
│   │   ├── dataset_builder.py # 학습 데이터셋 빌드 (뉴스↔가격 정렬)
│   │   └── dataset.py         # PyTorch Dataset
│   ├── models/
│   │   ├── momentum_model.py       # 멀티모달 모델 아키텍처
│   │   └── sentence_attribution.py # Integrated Gradients 문장 선별
│   ├── train.py               # 학습 파이프라인
│   └── api/
│       └── server.py          # FastAPI 추론 서버
├── scripts/
│   ├── collect_data.py        # 데이터 수집 CLI
│   └── serve.py               # API 서버 실행 CLI
├── notebooks/                 # 실험 노트북
├── data/
│   ├── raw/                   # (gitignored)
│   └── processed/             # (gitignored)
├── models/
│   ├── checkpoints/           # (gitignored)
│   └── best/                  # (gitignored)
├── pyproject.toml
├── .env.example               # (선택) W&B 키 등
└── README.md
```

## 빠른 시작

### 1. 환경 설정

```bash
pip install -e .
```

> **환경 변수**: 현재 필수 환경 변수는 없습니다. 선택적으로 W&B 학습 추적이 필요하면 `.env.example`을 참고하세요.

### 2. 데이터 수집

```bash
python scripts/collect_data.py
```

### 3. 모델 학습

```bash
python -m src.train
```

> **⚠️ GPU 권장**: DistilBERT 기반 모델(6700만 파라미터)이므로 GPU 환경에서 학습하세요.
> - **Google Colab (추천)**: `notebooks/train_colab.ipynb`를 Colab에서 열어 T4 GPU로 학습
> - **로컬 GPU**: NVIDIA GTX 1660+ (VRAM 6GB 이상)
> - CPU 전용 시 `config/default.yaml`에서 `batch_size: 4`, `max_length: 128`, `freeze_layers: 5`로 변경

### 4. API 서버 실행

```bash
python scripts/serve.py
# → http://localhost:8000
```

### 5. 예측 요청

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Tesla reports record quarterly revenue on strong EV demand",
    "summary": "Electric vehicle maker Tesla Inc reported record revenue...",
    "symbol": "TSLA",
    "explain": true
  }'
```

응답:
```json
{
  "predictions": [
    {
      "horizon_days": 30,
      "predicted_return_pct": 5.2,
      "label": "positive",
      "confidence_pct": 78.3,
      "direction": "up"
    },
    {
      "horizon_days": 180,
      "predicted_return_pct": 2.1,
      "label": "neutral",
      "confidence_pct": 65.1,
      "direction": "up"
    },
    {
      "horizon_days": 360,
      "predicted_return_pct": -1.5,
      "label": "neutral",
      "confidence_pct": 55.8,
      "direction": "down"
    }
  ],
  "model_version": "finz-momentum-v0.1",
  "sentiment": {
    "horizonDays": 30,
    "expectedMovePct": 5.2,
    "confidencePct": 78.3,
    "direction": "up"
  },
  "key_sentences": [
    {
      "sentence": "Tesla reports record quarterly revenue on strong EV demand",
      "attribution_score": 100.0,
      "rank": 1
    },
    {
      "sentence": "Electric vehicle maker Tesla Inc reported record revenue...",
      "attribution_score": 72.4,
      "rank": 2
    }
  ]
}
```

## finzfinz 연동

현재 finzfinz의 mock → 이 API로 교체:

```typescript
// finzfinz에서 교체할 부분
async function getModelSentiment(title: string, summary: string, symbol: string) {
  const res = await fetch(`${ANALYSIS_MODEL_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, summary, symbol }),
  });
  const data = await res.json();
  return data.sentiment; // { horizonDays, expectedMovePct, confidencePct, direction }
}
```

## DB 요구사항

현재 Supabase 등 외부 DB 의존성은 **없습니다**. 학습 데이터는 Yahoo Finance API + 로컬 파일(`data/raw/`)에서 수집되며, 빌드된 데이터셋은 `data/processed/dataset.parquet`에 저장됩니다.

향후 예측 캐시가 필요하면:

```sql
CREATE TABLE IF NOT EXISTS momentum_predictions (
  id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  news_url      TEXT NOT NULL,
  symbol        TEXT,
  horizon_days  INT NOT NULL,
  predicted_return_pct FLOAT,
  label         TEXT,
  confidence_pct FLOAT,
  direction     TEXT,
  model_version TEXT,
  created_at    TIMESTAMPTZ DEFAULT now()
);
```
