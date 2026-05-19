# Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking

## 📋 논문 정보
- **제목**: Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking
- **저자**: Dengming Zhang, Xiaowen Ma, Zhenliang Ni, Zhenkai Wu, Han Shu, Xin Jiang, Xinghao Chen
- **소속**: Zhejiang University, Huawei Noah's Ark Lab
- **발표**: Preprint (arXiv:2509.25712v1, 30 Sep 2025)
- **코드**: https://github.com/Littleor/ExpertMerging

---

## 🎯 핵심 요약

여러 도메인 전문가 모델을 하나로 합치는 모델 병합(Model Merging)에서, 기존 방법들은 손으로 고정한 계수를 쓰거나 레이어 이질성을 무시하는 한계가 있었다. **Expert Merging**은 레이블 없는 5~10개 샘플만으로 레이어별 계수를 학습해 병합 모델의 내부 표현과 출력을 전문가 모델에 맞춘다. **Expert Merging++**는 레이어 중요도를 측정해 중요한 레이어에만 더 세밀한 청크 단위 계수를 부여한다. InternVL, Qwen2-VL, Mistral 전 백본에서 기존 학습 기반 병합 기법을 모두 상회하며, 일부 설정에서는 supervised Mixture Training까지 초과한다.

---

## 🧩 핵심 흐름 (직관적 이해)

### 1단계 — 문제 정의: 기존 모델 병합은 왜 충분하지 않은가?

여러 전문가 모델(수학, 코딩, OCR 등)을 하나로 합치면 각 도메인의 능력을 동시에 가진 모델을 얻을 수 있다. 문제는 **어떻게 합치느냐**이다.

- **Training-free 방법** (Task Arithmetic 등): `θ_merged = θ_base + Σ λ_k τ_k` 형태로 λ를 사람이 직접 조정. 고정 계수라 데이터에 맞게 조정되지 않고, 레이어마다 같은 비율을 적용.
- **Training-based 방법** (AdaMerging 등): entropy 최소화로 계수를 학습하지만, 전문가 모델에 명시적으로 맞추는 신호가 없고, 레이어당 단일 계수 하나로 이질성을 포착하지 못함.

> **비유**: 여러 요리사의 레시피를 합칠 때, 재료 비율을 감으로 정하거나(training-free) 맛을 보지 않고 칼로리만 맞추는(training-based) 것과 같다. 실제로 원본 요리와 맛이 같은지 비교해가며 비율을 조정해야 한다.

### 2단계 — 발견: 레이어마다 중요도가 크게 다르다

*그렇다면 학습 방향을 어떻게 잡아야 할까?*

학습된 계수를 분석하면, 레이어마다 최적 계수 크기가 크게 다르다. 특히:
- **MLP(Gate/Up/Down)** 성분이 attention보다 중요도가 높음
- **후반부 레이어(late-stage)**일수록 중요도가 높음
- 레이어 타입마다 파라미터 수도 크게 다름

→ 모든 레이어에 하나의 계수를 쓰는 것은 명백히 낭비이다.

> **비유**: 오케스트라에서 모든 악기 볼륨을 같은 비율로 조정하는 것과 같다. 현악기군은 세밀하게, 타악기는 단순하게 조정해야 전체 균형이 맞는다.

### 3단계 — Expert Merging: 레이블 없는 데이터로 레이어별 계수를 학습

*어떻게 "원본 전문가 모델과 같은 내부 표현"을 목표로 학습할 수 있을까?*

각 태스크 도메인의 비레이블 입력 5~10개만 준비한다. 베이스 모델과 전문가 모델은 **고정(frozen)**한 채, K × L개의 계수만 학습한다.

손실함수는 두 가지를 동시에 최소화:

```
L_hid^(k) = Σ_ℓ E_x ||h_ℓ(x; θ_merged) - h_ℓ^(k)(x)||²   ← hidden state 맞추기
L_logit^(k) = T² · KL(softmax(z^(k)/T) || softmax(z/T))    ← 출력 분포 맞추기
```

계수가 Task Arithmetic 초기값에서 너무 벗어나지 않도록 정규화 항도 추가한다.

> **결과**: 정규화 없이 학습하면 InternVL에서 평균 -10.54점 급락. 정규화가 안정성의 핵심.

> **실제 예시**: Mistral-7B에서 Expert Merging은 AlpacaEval 77.84, 최고 Code 점수 53.66을 기록 — 프롬프트별 최적화 없이 AdaMerging++보다 높음.

### 4단계 — Expert Merging++: 중요한 레이어에만 더 세밀한 계수를

*레이어별 단일 계수로도 충분하지 않다면?*

중요한 레이어는 파라미터를 청크(chunk)로 나눠 청크마다 별도 계수를 부여한다. 중요도 낮은 레이어는 계수 하나로 유지.

레이어 중요도 계산:
```
I_ℓ = Norm(Σ_k |α_k^ℓ| · s_k^ℓ · n_ℓ)
  - α: 학습된 계수 크기
  - s: task-vector의 평균 절댓값
  - n: 레이어 파라미터 수
```

청크 수 배분:
```
m_ℓ = floor(B · I_ℓ^κ / Σ I_j^κ)   (B: 총 예산, κ: 집중도)
```

> **핵심**: 모든 레이어를 2× 청킹하면 오히려 성능 하락. 중요한 레이어만 선별적으로 세밀하게 제어하는 것이 핵심.

> **실제 예시**: Expert Merging++ vs Expert Merging — InternVL +0.34, Qwen2-VL +0.29, Mistral +1.28. 특히 Mistral에서 HumanEval(코드) +7.3점 향상.

### 5단계 — 반대 증명: 왜 무분별한 청킹은 실패하는가?

"청크가 많을수록 좋다"는 직관과 달리, 모든 레이어를 동일하게 2× 청킹하면 InternVL 56.76, Mistral 46.82로 Expert Merging++보다 낮아진다. 중요하지 않은 레이어에 파라미터를 낭비하면 과적합이 발생한다. 이는 **중요도 기반 선택적 할당**이 단순한 파라미터 증가보다 훨씬 효과적임을 입증한다.

---

## 📖 주요 내용

### 연구 배경

- 도메인 전문화를 위한 SFT 모델을 여러 개 운영하면 메모리·서빙 비용이 급증
- 모델 병합은 단일 모델로 다중 능력을 얻는 대안
- 기존 방법의 두 한계: (1) 고정 계수의 손 조정 의존, (2) 레이어 이질성 무시

### 방법: Expert Merging

**파라미터화**:
```
θ_merged^ℓ = θ_base^ℓ + Σ_k α_k^ℓ · τ_k^ℓ
```
- K × L개의 학습 가능한 계수 (K: 전문가 수, L: 레이어 수)
- 베이스·전문가 모델은 고정, 계수만 학습

**태스크 가중치 β_k**: 도메인별 중요도를 명시적으로 조절 가능한 knob 제공

### 방법: Expert Merging++

**레이어 중요도 → 청크 수 배분 → 청크별 계수 학습**의 3단계 파이프라인

총 계수 수는 Expert Merging과 거의 동일(B ≈ 0.9~1.2)하게 유지하면서 성능만 향상.

### 실험 설정

| 백본 | 크기 | 전문가 도메인 |
|------|------|-------------|
| InternVL2.5 | 1B | VQA, Geometry, Chart, OCR, Grounding |
| Qwen2-VL | 7B | VQA, Geometry, Chart, OCR, Grounding |
| Mistral | 7B | Chat, Math, Code |

비레이블 캘리브레이션 데이터: 태스크당 5~10개 샘플 (텍스트 or 이미지+텍스트)

---

## 📊 실험 결과

### MLLM: InternVL2.5 (Table 1, 평균 점수)

| 방법 | 평균 | vs. Expert Merging++ |
|------|------|---------------------|
| WUDI v2 (최강 baseline) | 56.86 | -1.49 |
| AdaMerging++ | 56.91 | -1.54 |
| **Expert Merging** | **58.11** | -0.34 |
| **Expert Merging++** | **58.45** | — |
| Mixture Training (supervised) | 57.66 | +0.79 |

- Grounding에서 특히 두드러짐: RefCOCO 80.05/80.53, RefCOCO+ 73.85/74.37

### MLLM: Qwen2-VL (Table 2, 평균 점수)

| 방법 | 평균 |
|------|------|
| WUDI v2 | 62.23 |
| Mixture Training | 62.23 |
| **Expert Merging++** | **63.63** |

### LLM: Mistral-7B (Table 3, 평균 점수)

| 방법 | 평균 |
|------|------|
| WUDI v2 | 45.48 |
| AdaMerging++ | 45.10 |
| **Expert Merging** | **47.43** |
| **Expert Merging++** | **48.71** |

### Ablation (Table 4)

| 제거 항목 | InternVL 변화 | Mistral 변화 |
|-----------|-------------|-------------|
| hidden alignment 제거 | -0.30 | -0.28 |
| logit alignment 제거 | -3.39 | -3.81 |
| coefficient regularization 제거 | **-10.54** | **-6.62** |
| task trade-off weights 제거 | -0.40 | -0.70 |
| chunking 제거 (Expert Merging++) | -0.34 | -1.28 |
| 전 레이어 2× 청킹 | -1.69 | -1.89 |

---

## 💡 주요 인사이트

1. **Task alignment > Parameter alignment**: hidden state + logit을 명시적으로 맞추는 것이 파라미터 공간 정렬보다 효과적
2. **정규화의 결정적 역할**: 비지도 최적화에서 계수가 폭주하면 그라운딩 같은 고용량 태스크에서 파국적 망각 발생
3. **레이어 이질성은 실재한다**: 후반부 MLP 레이어가 일관되게 중요 → 이를 무시한 단일 계수 방식은 구조적 한계
4. **데이터 효율성**: 5~10개 비레이블 샘플만으로 supervised Mixture Training과 경쟁 — 실용적 배포 비용 극적 절감
5. **Selective > Uniform**: 무차별 과파라미터화는 오히려 해롭고, 중요도 기반 선택이 핵심

---

## 🔬 기술적 세부사항

### 전체 목적함수

```
min_{α} L_align + γ R(α)

L_align = Σ_k β_k (L_hid^(k) + L_logit^(k))
R(α) = (1/KL) Σ_k Σ_ℓ |α_k^ℓ - ᾱ_k|   (ᾱ_k: Task Arithmetic 초기값)
```

### Expert Merging++ 청크 병합

```
θ_merged^ℓ = θ_base^ℓ + Σ_k Σ_s α_{k,s}^ℓ · τ_{k,s}^ℓ
```
- τ_{k,s}^ℓ: 전문가 k의 레이어 ℓ에서 청크 s
- α_{k,s}^ℓ: 해당 청크의 독립 학습 계수

### 레이어 중요도 분석 결과

- **Gate/Up** (MLP): 일관되게 가장 높은 중요도
- **K/V** (Attention): 가장 낮은 중요도
- **Down** (MLP): 백본별 편차가 가장 큼 (InternVL에서 최고, Mistral에서 최저)
- 깊이: Early → Middle → Late 순으로 단조 증가 경향 (특히 Mistral에서 뚜렷)

---

## 📌 한계점

1. **데이터 의존**: 비레이블이지만 태스크별 파티셔닝된 캘리브레이션 셋 필요 → 완전히 데이터 없는 시나리오에는 적용 불가
2. **이종 모델**: 현재는 동일 백본 기반 전문가만 실험. 서로 다른 아키텍처 모델 병합은 미래 과제
3. **κ 하이퍼파라미터**: 청크 집중도 κ의 최적값이 백본마다 달라 추가 탐색 필요
4. **분석 범위**: 레이어 중요도 분석이 3개 백본에 국한, 더 큰 스케일에서의 일반화는 미확인

---

## 🎓 결론

Expert Merging은 비레이블 데이터만으로 전문가 모델의 내부 표현에 병합 모델을 맞추는 training-light 병합 프레임워크다. Expert Merging++는 여기에 중요도 기반 청크 단위 계수를 더해 inter-layer 이질성 문제를 해결한다. 적은 데이터로 supervised Mixture Training을 넘어서는 결과는 실용적 다중 전문가 통합의 새로운 기준을 제시한다.
