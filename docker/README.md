# Docker strategy (fallback) — Step 3 scaffold

SoT (policy):
- 기본 전략: 단일 env (pyproject.toml + requirements.lock)
- fallback: 의존성 충돌 시 모델별 Docker 이미지로 실행하고 결과(run_dir)만 수집

> Step 3에서는 실제 docker orchestration을 구현하지 않습니다.
> 다만 Dockerfile 템플릿 + 규약을 고정해 두고, Step 6에서 runner가 이를 사용합니다.

---

## Build

예시(모델별 이미지):
```bash
cd /mnt/data/bench

docker build -f docker/kalmannet_tsp.Dockerfile -t bench/kalmannet_tsp:dev .
docker build -f docker/adaptive_knet.Dockerfile -t bench/adaptive_knet:dev .
docker build -f docker/maml_knet.Dockerfile -t bench/maml_knet:dev .
docker build -f docker/split_knet.Dockerfile -t bench/split_knet:dev .
docker build -f docker/my_model.Dockerfile -t bench/my_model:dev .
```
# Run (규약)
## Mount 규약

-v /mnt/data/bench:/workspace/bench (벤치 코드/설정)

-v /mnt/data:/workspace/data (SoT suite/decisions + 데이터 캐시/출력 위치)

결과(run_dir)는 suite의 output_dir_template에 따라 /workspace/data/runs/...에 생성되는 것을 기본으로 가정.

## GPU

NVIDIA runtime:

--gpus all (또는 --gpus device=0)

CUDA 버전은 호스트/드라이버에 따라 다르므로, BASE_IMAGE는 필요 시 교체.

## MAML 주의

MAML-KalmanNet은 README 기준 데이터 생성 시 use_cuda=False 권고가 있음.

Step 6 orchestration에서 “데이터 생성 단계는 CPU로 수행”하도록 분리하는 것을 권장.

## Notes

Dockerfile은 최소 템플릿이며, third_party repos가 실제로 붙으면
모델별 추가 의존성을 여기에 반영하거나, requirements.lock을 갱신합니다.
