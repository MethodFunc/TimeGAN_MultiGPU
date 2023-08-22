# TimeGAN MultiGPU

----

## 1. requierments
- tensorflow 2.5
- scikit-learn
- numpy == 1.19.5
- matplotlib
- pandas
- tqdm
- nvidia gpu

## 2. Describe
- 현재 제작 중인 스크립트 입니다.
- 다중 GPU와 싱글 GPU, CPU 모두 지원하는 모델을 완성 하는 것이 목적입니다.
- 현재 싱글  GPU / 다중 GPU가 돌아가는 것을 확인했습니다.
- 다음 스탭은 exp에 뭉쳐있는 모델을 모델 폴더로 빼는 작업입니다. (완료)
- 로깅 작업 예정
- 설정은 setting.yaml에서 하면 됩니다. (추후 arguments로 변경 예정)

## 3. Bug
- os.environ을 사용하지 않으면 싱글 그래픽카드 사용 불가 버그(fixed)