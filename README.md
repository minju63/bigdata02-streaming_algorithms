# Streaming Algorithms (Bloom / Count-Min / HyperLogLog / Reservoir) on MovieLens 1M

MovieLens 1M 평점 스트림(1,000,209건)을 한 줄씩 처리하며 4개 스트리밍 알고리즘을 직접 구현하고,
정확도·메모리·시간을 정확 계산(set/Counter)과 비교한 과제 코드입니다. 라이브러리 없이 `algorithms.py`에 직접 구현했습니다.

## 파일
| 파일 | 역할 |
|---|---|
| `algorithms.py` | BloomFilter, CountMinSketch, HyperLogLog, ReservoirSampler (R / L) 구현 |
| `download_data.py` | MovieLens 1M 다운로드·압축해제 (`data/ml-1m/ratings.dat`) |
| `run_experiments.py` | 전체 실험 → `outputs/results.json`, `outputs/fig_*.png` |

## 실행 방법 (Windows / macOS / Linux 공통)
Python 3.10 이상 권장 (개발 환경: Python 3.14, numpy 2.4). 이 폴더(`streaming_algorithms/`)에서:

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
