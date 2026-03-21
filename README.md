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

### 학습 데이터
| 소스 | 내용 |
|------|------|
| Supabase (finzfinz) | 뉴스 기사 (영어 원문 + 한국어 번역) |
| yfinance | 종목별 OHLCV, 시가총액, 매출, 영업이익 |
| yfinance | S&P500, 금, WTI, BTC, VIX, 달러인덱스 |

## 프로젝트 구조

```
finzanalysismodel/
├── config/
│   └── default.yaml          # 전체 설정 (모델, 학습, 데이터, API)
├── src/
│   ├── config.py              # 설정 로더
│   ├── data/
│   │   ├── news_collector.py  # Supabase 뉴스 수집
│   │   ├── price_collector.py # yfinance 주가/매크로 수집
│   │   ├── dataset_builder.py # 학습 데이터셋 빌드 (뉴스↔가격 정렬)
│   │   └── dataset.py         # PyTorch Dataset
│   ├── models/
│   │   └── momentum_model.py  # 멀티모달 모델 아키텍처
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
├── .env.example
└── README.md
```

## 빠른 시작

### 1. 환경 설정

```bash
pip install -e .

cp .env.example .env
# .env 파일에 SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY 입력
```

### 2. 데이터 수집

```bash
python scripts/collect_data.py
```

### 3. 모델 학습

```bash
python src/train.py
```

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
    "symbol": "TSLA"
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
  }
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

**추가 테이블 불필요** — finzfinz의 기존 Supabase `news_articles` 테이블을 읽기 전용으로 사용합니다.

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
