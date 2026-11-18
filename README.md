# Streaming Algorithms: Bloom Filter and Count-Min Sketch

MovieLens 1M 평점 스트림 1,000,209건을 한 줄씩 처리하며 Bloom Filter와 Count-Min Sketch의 정확도·메모리·처리시간을 비교하는 과제 코드입니다.

## 구현 범위

| 알고리즘 | 목적 | Ground Truth |
|---|---|---|
| Bloom Filter | `(user_id, movie_id)` 쌍의 포함 여부 판정 | Python `set` |
| Count-Min Sketch | 영화별·사용자별 평점 횟수 추정 | Python `Counter` |

Bloom Filter는 비트 배열과 이중 해싱을, Count-Min Sketch는 `depth × width` 카운터 배열과 행별 해시 함수를 사용합니다. 대표 성능 실험은 파일을 한 줄씩 읽으며 각 레코드마다 `add()`를 호출합니다. 파라미터 스윕에서는 실행 시간을 줄이기 위해 같은 해시·갱신 원리의 `add_many()`를 사용합니다.

## 파일

| 파일 | 역할 |
|---|---|
| `algorithms.py` | Bloom Filter와 Count-Min Sketch 구현 |
| `download_data.py` | MovieLens 1M 다운로드 및 압축 해제 |
| `run_experiments.py` | 전체 실험 및 결과·그림 생성 |
| `run_extra.py` | 추가 실험(CMS Top-K 복원율, Bloom 용량 초과, 병합 검증, 메모리 확장성, 시간 5회 반복) |
| `outputs/results.json` | 대표 실험·파라미터 스윕 원시 결과 |
| `outputs/extra.json` | 추가 실험 원시 결과 |

## 실행 방법

Python 3.10 이상에서 저장소 루트를 기준으로 실행합니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python download_data.py
python run_experiments.py
python run_extra.py      # 추가 실험 (run_experiments.py와 독립적으로 실행 가능)
```

실험 결과는 `outputs/results.json`, 그림은 `outputs/fig_*.png`에 생성됩니다. 난수 seed는 42로 고정되며, 시간과 처리량은 실행 환경에 따라 달라질 수 있습니다.

## 과제 요구사항 대응

| 요구사항 | 구현 내용 |
|---|---|
| 100,000건 이상 | MovieLens 1M 1,000,209건 |
| 스트림 처리 | `stream()` 제너레이터로 한 줄씩 처리 |
| 2개 알고리즘 | Bloom Filter, Count-Min Sketch |
| 정확도 | FPR·FNR, 절대오차·상대오차·오차 상한 |
| 메모리 | 자료구조 크기와 `tracemalloc` 피크 |
| 시간 | 전체 처리시간과 초당 처리량 |
| 파라미터 비교 | Bloom의 비트 수·해시 수, CMS의 width·depth |
