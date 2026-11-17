# Streaming Algorithms (Bloom / Count-Min / HyperLogLog / Reservoir) on MovieLens 1M

MovieLens 1M 평점 스트림(1,000,209건)을 한 줄씩 처리하며 4개 스트리밍 알고리즘을 직접 구현하고,
정확도·메모리·시간을 정확 계산(set/Counter)과 비교한 과제 코드입니다. 라이브러리 없이 `algorithms.py`에 직접 구현했습니다.

## 파일
| 파일 | 역할 |
|---|---|
| `algorithms.py` | BloomFilter, CountMinSketch, HyperLogLog, ReservoirSampler (R / L) 구현 |
| `download_data.py` | MovieLens 1M 다운로드·압축해제 (`data/ml-1m/ratings.dat`) |
| `run_experiments.py` | 전체 실험 → `outputs/results.json`, `outputs/fig_*.png` |
| `outputs/results.json` | 보고서 표와 분석에 사용한 원시 실험 결과 |
| `outputs/fig_*.png` | 보고서에 사용한 파라미터·성능 비교 그림 |

## 실행 방법 (Windows / macOS / Linux 공통)
Python 3.10 이상 권장 (개발 환경: Python 3.14, numpy 2.4). 저장소 루트에서:

```bash
# 1) (선택) 가상환경
python -m venv .venv
#   Windows PowerShell: .venv\Scripts\Activate.ps1   /   macOS·Linux: source .venv/bin/activate

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 데이터 다운로드 (약 6MB, 인터넷 필요)
python download_data.py

# 4) 실험 실행 (약 5~10분, 결과는 outputs/ 에 저장)
python run_experiments.py

```

- 어느 위치에서 실행해도 되도록 모든 경로는 스크립트 파일 기준입니다. 한글 경로여도 동작합니다.
- 난수 seed=42 고정. 정확도 수치는 재현되지만 **시간/처리량은 CPU에 따라 달라집니다**.
- `python run_experiments.py --figures-only` : 저장된 `results.json`으로 그림만 다시 그림.
- PDF 보고서는 제출용 별도 산출물이며, 이 저장소는 보고서의 수치와 그림을 재현하는 소스코드·원시 결과를 제공합니다.

## 과제 요구사항 대응

| 요구사항 | 구현 위치 |
|---|---|
| 100,000건 이상, 한 줄씩 스트림 처리 | MovieLens 1M 1,000,209건, `stream()` 제너레이터 |
| 2개 이상 직접 구현 | `algorithms.py`의 Bloom Filter, Count-Min Sketch, HyperLogLog, Reservoir Sampling |
| Ground Truth 비교 | `set`, `Counter`, 전체 평점 분포와 각각 비교 |
| 정확도·메모리·시간 측정 | `run_experiments.py` Part A, `outputs/results.json`의 `A` |
| 파라미터 비교 실험 | `run_experiments.py` Part B, `outputs/results.json`의 `B` |

대표 벤치마크는 파일을 한 줄씩 읽으면서 각 알고리즘의 `add()`를 호출합니다. 파라미터 스윕은 실행 시간을 줄이기 위해 같은 해시·갱신 로직의 `add_many()`를 사용하며, 실행 중 두 방식의 결과가 같은지 검증합니다.
