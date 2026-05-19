# Visual Position Prompt for MLLM based Visual Grounding

## 📋 논문 정보
- **제목**: Visual Position Prompt for MLLM based Visual Grounding
- **저자**: Wei Tang, Yanpeng Sun, Qinying Gu, Zechao Li (Nanjing University of Science and Technology, Shanghai AI Lab)
- **발표**: arXiv:2503.15426v4 [cs.CV] 16 Jul 2025
- **코드**: https://github.com/WayneTomas/VPP-LLaVA

## 🎯 핵심 요약
Multimodal Large Language Models (MLLMs)은 다양한 이미지 관련 작업에서 뛰어난 성능을 보이지만, visual grounding과 같은 위치 인식 작업에서 좌표와 공간 정보를 정확하게 정렬하는데 어려움을 겪는다. 본 논문은 Visual Position Prompt (VPP)를 도입한 VPP-LLaVA를 제안하여 이 문제를 해결한다. Global VPP는 축 형태의 학습 가능한 텐서를 입력 이미지에 오버레이하여 구조화된 공간 단서를 제공하고, Local VPP는 위치 인식 쿼리를 통합하여 세밀한 위치 파악을 지원한다. 0.6M 샘플의 컴팩트한 VPP-SFT 데이터셋으로 학습하여, 기존 모델들(예: MiniGPT-v2의 21M 샘플)보다 훨씬 적은 데이터로 SOTA 성능을 달성하고 강력한 zero-shot 일반화 능력을 보인다.

## 📖 주요 내용

### 연구 배경 및 문제점

**MLLMs의 공간 이해 한계**
- MLLMs은 다양한 vision-language 작업에서 인상적인 결과를 달성했지만, visual grounding (특히 Referring Expression Comprehension, REC)에서 정확한 위치 파악에 어려움을 겪음
- Visual grounding은 자유 형식의 언어 표현을 기반으로 이미지 내의 위치를 정확하게 식별하는 작업
- 예시: "brown toy"라는 표현에 대해 LLaVA-v1.5는 부정확한 bounding box를 출력 (크기와 형태의 부정확성)

**근본적인 두 가지 원인**
1. **명시적 공간 참조 부족**: MLLMs는 텍스트 설명을 정확한 이미지 위치와 연결하기 어려움
2. **전역 컨텍스트 우선**: 특징 추출 과정에서 전역 컨텍스트를 우선시하고 세밀한 공간 세부 정보를 간과하여 약한 위치 파악 능력으로 이어짐

**해결 방안의 통찰**
- 좌표 축 형태의 위치 참조를 제공하면 모델의 공간 관계 이해가 크게 향상됨
- 명시적 공간 가이드가 있을 때, 예측된 bounding box가 대상 객체와 더 잘 정렬됨

### 제안 방법: VPP-LLaVA

#### 전체 아키텍처
VPP-LLaVA는 LLaVA-v1.5 프레임워크를 기반으로 구성:
- **Visual Encoder**: CLIP-L/336 (이미지 특징 추출)
- **Projector**: 2-layer MLP (시각 특징을 LLM 특징 공간으로 매핑)
- **LLM Backbone**: Vicuna-v1.5 (텍스트 프롬프트 처리)
- **핵심 추가 요소**: Global VPP + Local VPP

#### Global Visual Position Prompt

**초기화 및 구조**
```
δg_i ← T_v(X_axis) ∈ R^(3×336×336)
```
- 축 형태 이미지(axis-like image)로 초기화되는 학습 가능한 프롬프트
- RGB 3채널 이미지: 흰색 배경, 검은색 축/좌표 숫자/눈금
- 기본 설정: 0.1 단위 스케일, 가장자리를 따라 축 배치

**오버레이 프로세스**
```
X_gp = α · T_v(X) + (1 - α) · T_ipt(δg_i ⊙ M_w)
```
- `α`: 글로벌 VPP의 강도를 제어하는 tradeoff 매개변수 (기본값: 0.95)
- `T_ipt`: 입력 크기에 맞게 스케일링하는 보간 연산
- `M_w`: 폭 w의 바이너리 마스크 (기본값: 30 pixels)
  - 가장자리는 1, 나머지는 0으로 설정하여 좌표만 보이도록 함
  - 반투명 회색 오버레이로 표시되어 보이면서 학습 가능함을 나타냄

**특징 추출**
```
F_gp = CLIP(X_gp)
F'_gp = MLP(F_gp)
```

**설계 원리**
- LLaVA 학습 데이터의 좌표 데이터 형식과 일치하여 MLLM에 전역 위치 참조 제공
- 명시적 제약 없이 최적화되어 학습 중 유연한 적응 가능
- 학습 가능하게 만들어 데이터 기반 방식으로 위치 관련 단서를 더 잘 인코딩

#### Local Visual Position Prompt

**생성 프로세스**
```
F_lp = DETR(X_gp, O)
```
- DETR (Detection Transformer) 사용
  - ResNet-101 backbone
  - 6-layer transformer encoder
  - 6-layer transformer decoder
- 100개의 object query embeddings 생성 (객체 위치 + 의미 정보 캡처)

**특징 매핑**
```
F'_lp = MLP(F_lp)
```

**기존 방법과의 차별점**
1. **동적 생성**: 사전 학습된 detector의 bounding box 제안에 크게 의존하는 기존 2단계 방법과 달리, 공간 및 의미 정보를 모두 캡처하는 객체 위치 임베딩을 동적으로 생성
2. **순수 MLLM**: ContextDET처럼 DETR을 통합하고 추가 box decoder를 추가하면서 LLM을 융합에만 사용하는 것과 달리, 순수 MLLM 방식으로 아키텍처를 단순화하고 성능 향상

#### Instruction Tuning

**특징 융합 및 처리**
```
F' = concatenate([F'_gp, F'_lp])
P(X_a | F', X_q) = ∏_{i=1}^L P_θ(x_i | F', X_q, X_a,<i)
```
- Global 및 Local VPP 특징을 직접 연결 (concatenation)
- 쿼리 텍스트 지시와 함께 LLM에 입력
- Autoregressive language modeling을 통해 응답 생성

**VPP 관련 텍스트 지시**
- 샘플 레벨에 포함: "Each image is accompanied by axes. If the question pertains to the bounding box coordinates, refer to the axes for the response."
- 명시적 텍스트 가이드가 global VPP를 효과적으로 활용하는데 필수적

### 데이터셋 구축: VPP-SFT

**규모 및 구성**
- 약 0.6M 샘플 (기존 모델들보다 훨씬 작음)
  - MiniGPT-v2: 21M 샘플
  - KOSMOS-2: 20M 샘플
  - Shikra: 4M 샘플

**데이터 소스**
1. **LLaVA-665K 서브셋**: 134,864 대화
   - Region captions 및 visual grounding 대화 추출
2. **CB-GRD**: 264,516 대화
   - 특수 포맷 체계 제거, referring expressions만 보존
3. **CB-REF**: 87,091 대화
   - Region captioning 데이터로 LLaVA의 언어 능력 유지
4. **Genixer**: 130,000 대화
   - LLaVA-665K의 grounding 템플릿에 따라 대화 구성

**데이터 전처리**
- 모든 소스의 bounding box 좌표를 LLaVA 형식으로 변환
- 긴 변을 기준으로 이미지 패딩
- (x1, y1, x2, y2) 좌표를 정규화하여 패딩된 정사각형 입력과 호환되도록 함

**VPP-SFT의 차별점**
- 단순한 데이터셋 병합이 아님: 흩어진 visual grounding 데이터셋들을 MLLMs를 위해 통합
- 복잡한 형식과 특수 위치 토큰이 많은 기존 데이터셋과 달리 간단하고 직관적인 형식
- 높은 재사용성: 새로운 데이터셋 생성이 용이하고 직접적인 fine-tuning 가능
- 고품질: Zero-shot GSEval-BBox 벤치마크에서 part-object 및 multi-object 시나리오에 대한 인상적인 일반화 능력 입증

### 실험 설정 및 구현 세부사항

**벤치마크**
1. **RefCOCO**: 50,000 객체, 19,994 이미지, 142,209 표현 쿼리 (평균 3.61 단어)
   - testA: 사람 설명 중심
   - testB: 사람 외 객체 중심
2. **RefCOCO+**: 49,856 객체, 19,992 이미지, 141,564 표현 (평균 3.53 단어)
   - 절대 위치 표현 (right, top 등) 금지, 시각적 속성 강조
3. **RefCOCOg**: 49,822 객체, 25,799 이미지, 95,010 표현 (평균 8.43 단어)
   - RefCOCOg-umd 버전 사용
4. **ReferIt**: 20,000 이미지 (zero-shot 평가용)
   - 모호한 쿼리 포함, 일부 부정확한 bounding box 레이블
5. **GSEval-BBox**: 3,715 샘플 (zero-shot 평가용)
   - 4가지 fine-grained 표현 유형: stuff, part-object, multi-object, single-object
   - Part-object 및 multi-object 참조가 특히 도전적

**학습 세부사항**
- **초기화**: 사전 학습된 LLaVA-v1.5-7B/13B 파라미터로 초기화
- **Optimizer**: AdamW with cosine annealing scheduler
- **Learning rates**:
  - LLM (Vicuna-v1.5): 2e-5
  - Global VPP: 2e-4
  - Local VPP generator: 2e-5
  - Projector (local VPP → LLM): 2e-4
  - Visual encoder: 2e-6 (unfrozen)
- **학습 설정**:
  - Epochs: 3
  - Global batch size: 64
  - 학습 시간: 약 30시간 (8× NVIDIA A100 80GB GPUs)

**하이퍼파라미터**
- α (global VPP 강도): 0.95
- w (바이너리 마스크 폭): 30 pixels
- 폰트 크기 (축 숫자): 10

## 💡 주요 인사이트

1. **적은 데이터로 높은 성능**: 0.6M 샘플로 21M 샘플을 사용한 모델들을 능가하는 성능 달성 - 데이터 품질과 효율적 설계의 중요성 입증

2. **명시적 위치 참조의 중요성**: 좌표 축 형태의 명시적 공간 가이드가 MLLMs의 공간 이해를 크게 향상시킴

3. **Global-Local 상호보완**: 전역 공간 구조와 지역 객체별 위치 파악의 시너지 효과

4. **제로샷 일반화**: 학습 시 보지 못한 데이터셋에서도 강력한 성능 - 특히 GSEval-BBox의 part-object (69.9%) 및 multi-object (82.7%) 시나리오에서 SOTA

5. **학습 가능한 프롬프트**: Global VPP를 학습 가능하게 만들어 데이터 기반 방식으로 위치 관련 단서를 최적화

6. **전이 가능성**: VPP 메커니즘이 LLaVA-NeXT 등 다른 MLLM 아키텍처로 효과적으로 전이 가능

## 🔬 기술적 세부사항

### Global VPP 초기화 변형 실험

4가지 초기화 조건 비교:
1. **Default**: 0.1 단위 스케일, 가장자리를 따라 축 배치 → 최고 성능
2. **Internal-0.05**: 0.05 단위 스케일, 가장자리 축
3. **Cross-axis-0.1**: 0.1 단위 스케일, 중앙 십자형 축
4. **External-0.1**: 0.1 단위 스케일, 외부 패딩된 가장자리 축
   - 스케일 불일치로 성능 저하 (실제 이미지는 336×336 중 276×276만 차지)

**교훈**: 간격 스케일은 기본 MLLM의 내부 좌표 시스템과 정렬되어야 함

### Local VPP Generator 비교

| Generator | 특징 | 성능 |
|-----------|------|------|
| Vanilla-DETR | 100 queries | 최고 성능 (기본 설정) |
| Anchor-DETR | 300 queries (기본) | 하위 성능 |
| Anchor-DETR* | 100 queries (조정) | 성능 향상 (쿼리 수 감소로) |
| YOLO-World* | 116 queries | 경쟁력 있는 성능 |

**관찰**: 과도한 제안 수가 LLM의 공간 추론을 방해할 수 있음

### Fusion Strategy 비교

| Strategy | 설명 | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|----------|------|-------------|--------------|--------------|
| Cross-Attention-1 | F'_lp를 query, F'_gp를 key로 | 40.37 | 26.34 | 29.98 |
| Cross-Attention-2 | F'_gp를 query, F'_lp를 key로 | 82.74 | 73.71 | 76.33 |
| Concatenation | 직접 연결 (채택) | **85.29** | **77.68** | **79.80** |

**이유**: Cross-attention 모듈은 처음부터 학습되어 LLM의 사전 학습된 표현과의 정렬이 제한됨

### Visual Encoder Freezing vs Unfreezing

| 설정 | RefCOCO testB | RefCOCO+ testA | RefCOCOg test |
|------|---------------|----------------|---------------|
| Frozen | 80.06 | 84.75 | 80.36 |
| Unfrozen | **81.59** | **86.22** | **82.14** |

**이유**: Global VPP는 MLLM의 사전 학습 데이터에 자연스럽게 존재하지 않으므로, visual encoder를 unfreeze하면 이 새로운 입력 형태를 더 잘 학습하고 적응할 수 있음

### 구성 요소 Ablation Study

| Global VPP | Local VPP | RefCOCO val | RefCOCO testB | RefCOCOg test | 개선폭 |
|------------|-----------|-------------|---------------|---------------|--------|
| ✗ | ✗ | 84.58 | 78.07 | 78.63 | Baseline |
| ✓ | ✗ | 84.95 | 78.78 | 79.79 | +1.16 |
| ✗ | ✓ | 84.46 | 78.49 | 79.36 | +0.73 |
| ✓ | ✓ | **85.29** | **80.06** | **80.36** | **+1.73** |

**통찰**:
- Global VPP 단독: 특히 RefCOCOg에서 1.5% 향상, 안정적인 공간 참조 제공
- Local VPP 단독: 혼합된 결과, 제한된 데이터로 DETR 특징 정렬이 어려움
- 결합 시: 시너지 효과로 약 2% 향상 (RefCOCO testB)

## 📊 성능 비교

### RefCOCO/RefCOCO+/RefCOCOg 벤치마크

**VPP-LLaVA-7B 성능 (Generalist 모델 중)**

| Model | Data Scale | RefCOCO testA | RefCOCO testB | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg val |
|-------|------------|---------------|---------------|----------------|----------------|--------------|
| MiniGPT-v2-7B | ~21M | 88.06 | 91.29 | 84.30 | 79.58 | 85.52 |
| Qwen-VL-7B† | ~21M | 88.55 | 92.27 | 84.51 | 82.82 | 88.59 |
| Groma | ~26M | 89.53 | 92.09 | 86.26 | 83.90 | 88.91 |
| **VPP-LLaVA-7B** | **~0.6M** | **90.37** | **92.89** | **85.77** | **84.65** | **89.84** |

† 더 큰 visual encoder (1.9B ViT-BigG) 사용

**Specialist 모델과 비교**

| Model | RefCOCO testA | RefCOCO testB | RefCOCO+ testA | RefCOCO+ testB |
|-------|---------------|---------------|----------------|----------------|
| UNINEXT-L | 91.43 | 93.73 | 88.93 | 83.09 |
| **VPP-LLaVA-7B** | **90.37** | **92.89** | **85.77** | **84.65** |

- RefCOCO+의 3개 split에서 UNINEXT-L 초과 (2.56%, 1.94%, 0.84%)

**VPP-LLaVA-13B 성능**

| Model | Data Scale | RefCOCO testA | RefCOCO+ testA | RefCOCOg val |
|-------|------------|---------------|----------------|--------------|
| Griffon-v2-13B | ~13M | 89.60 | 86.50 | 85.50 |
| Lion-12B† | 7.2M | 89.80 | 85.57 | 89.22 |
| **VPP-LLaVA-13B** | **~0.6M** | **90.32** | **86.34** | **90.78** |

† 1.1B EVA-G visual encoder 사용

### Zero-Shot 일반화 성능

**ReferIt Dataset**

| Model | ReferIt val | ReferIt test |
|-------|-------------|--------------|
| LLaVA-v1.5-7B | 48.95 | 47.42 |
| VPP-LLaVA-7B | **57.55** (+8.6) | **56.53** (+9.1) |

**GSEval-BBox Dataset (Zero-Shot)**

| Model | Stuff | Part | Multi | Single | All |
|-------|-------|------|-------|--------|-----|
| InternVL2.5-78B | 85.3 | 63.2 | 55.7 | 16.8 | 62.2 |
| Qwen2.5-VL-72B | 88.4 | 42.8 | 64.6 | 31.2 | 62.5 |
| Qwen2.5-VL-7B | 93.0 | 75.9 | 59.1 | 17.6 | 66.7 |
| Ferret-7B | 82.1 | 56.0 | 43.2 | 17.6 | 53.3 |
| Ferret-13B | 80.6 | 58.0 | 46.6 | 21.3 | 55.1 |
| **VPP-LLaVA-7B** | 52.1 | **82.7** | **68.7** | **67.9** | **67.0** |
| **VPP-LLaVA-13B** | 54.5 | **83.8** | **69.5** | **69.9** | **68.4** |

**주목할 점**:
- Part-object에서 압도적 성능 (83.8% vs Qwen2.5-VL-7B 75.9%)
- Multi-object에서도 최고 성능 (69.5% vs Qwen2.5-VL-72B 64.6%)
- Stuff 카테고리는 상대적으로 낮지만 전체적으로 최고 성능

### Region Captioning 성능 (RefCOCOg)

| Model | METEOR | BERTScore-F1 | CIDER |
|-------|--------|--------------|-------|
| GPT4ROI | 9.7 | 87.3 | 60.3 |
| Kosmos-2 | 12.2 | 87.1 | 73.1 |
| LLaVA-v1.5-7B | 12.0 | 86.9 | 72.5 |
| Griffon-v2-13B | 12.1 | 88.0 | - |
| **VPP-LLaVA-7B** | **12.1** | **87.1** | **73.1** |

**의의**: Visual grounding에 특화되었음에도 강력한 언어 생성 능력 유지

### Transferability Study

**LLaVA-NeXT에 VPP 적용**

| Model | RefCOCO val | RefCOCO testA | RefCOCO+ testA | RefCOCOg val |
|-------|-------------|---------------|----------------|--------------|
| LLaVA-NeXT | 84.74 | 89.67 | 77.27 | 80.12 |
| LLaVA-NeXT+VPP | **90.28** (+5.54) | **93.26** (+3.59) | **84.75** (+7.48) | **85.60** (+5.48) |

**의의**: VPP가 다른 MLLM 아키텍처에도 효과적으로 전이됨을 입증

## 🔗 관련 연구

### Visual Grounding 방법론

**전통적 방법**
1. **Two-stage 방법** (EARN, MattNet): 표준 detection 네트워크로 region proposals 생성 후 cross-modal matching으로 선택
2. **One-stage 방법** (FAOA): YOLO 기반 네트워크로 융합된 특징에서 직접 bounding box 생성
3. **Transformer 기반** (TransVG, VLTVG, TransCP, MDETR, CLIP-VG, OFA, UNINEXT-L): Vision-Language Pre-trained (VLP) 모델 활용, 앵커 독립적

**MLLM 기반 방법**
1. **좌표 직접 출력 방식**:
   - Shikra, Kosmos, MiniGPT-v2: 좌표를 특수 토큰으로 이산화
   - Ferret: Spatial-Aware Visual Sampler 도입
   - PINK: Self-consistent bootstrapping 접근법

2. **Task-specific decoder 방식**:
   - LISA, GLaMM: SAM decoder로 pixel-level segmentation
   - LLaVA grounding: 추가 grounding 모듈로 bounding box 예측

**본 연구의 차별점**:
- 명시적 위치 참조 제공으로 좌표-공간 정보 정렬 지원
- 훨씬 작은 데이터셋(0.6M vs 21M)으로 SOTA 달성

### Visual Prompt 연구

**VLP 모델**
- VPT, MaPLe, CMPA: CLIP visual embeddings 전에 학습 가능한 토큰 추가로 few-shot classification 전이
- PEVL, CPT: Visual grounding과 같은 position-sensitive vision-language 작업을 mask token prediction으로 적응

**Large Vision Models**
- SAM 및 variants: Image segmentation 가이드를 위한 visual prompts

**MLLMs**
- DetToolChain, Scaffolding: GPT-4V를 위한 position-guided visual prompts (Chain-of-Thought 능력에 크게 의존)
- Ferret, ViP-LLaVA: 낙서, 화살표 등 손으로 그린 자유 형식 visual prompts
- Transferable visual prompt: 단일 모델에서 학습 후 다른 모델로 전이 가능
- External knowledge: Segmentation masks를 visual prompts로 통합하여 시각적 이해 향상

## 🚀 응용 분야 및 활용 방안

1. **인간-로봇 상호작용**: "왼쪽에 있는 빨간 공 가져와"와 같은 자연어 명령을 정확한 객체 위치로 변환

2. **이미지 편집 및 조작**: 텍스트 설명으로 이미지의 특정 영역 선택 및 편집

3. **원격 감지 (Remote Sensing)**: 위성 이미지에서 "해안 근처의 배" 등의 설명으로 객체 위치 파악

4. **자율 주행**: "앞쪽 차선에 있는 파란 차"와 같은 설명으로 주변 환경의 객체 인식

5. **의료 이미징**: "좌상엽의 결절"과 같은 의학적 설명으로 의료 이미지의 관심 영역 정확히 위치 파악

6. **교육 및 접근성**: 시각 장애인을 위한 이미지 콘텐츠 설명 및 위치 파악 지원

7. **E-commerce**: "두 번째 행 세 번째 제품"과 같은 설명으로 제품 검색 및 선택

8. **증강 현실 (AR)**: 자연어 쿼리를 기반으로 실제 환경의 객체와 정보 오버레이 정확히 연결

## 📌 한계점 및 향후 연구 방향

### 현재 한계점

1. **복잡한 관계 추론**: 객체 간 복잡한 관계를 포함하는 쿼리에서 정확한 bounding box 제공 어려움
   - 예: "왼쪽 사람이 잡고 있는 물건 오른쪽의 객체"

2. **모호한 쿼리 처리**: 이미지 내 여러 영역을 지칭할 수 있는 모호한 쿼리에서 어려움
   - 특히 subject가 누락된 경우 더욱 어려움

3. **부정 표현**: "빨간색이 아닌 모자"와 같은 부정을 포함하는 쿼리에서 차선의 성능
   - MLLMs의 부정 관련 의미 이해 및 추론 능력 한계

4. **Stuff 카테고리**: GSEval-BBox의 stuff 카테고리에서 상대적으로 낮은 성능
   - 제한된 데이터 커버리지 및 데이터셋 편향 가능성

5. **Ground Truth 품질**: 일부 데이터셋의 ground truth 주석이 완벽하게 정확하지 않음

6. **다중 모달 작업 범위**: 좌표 관련 작업에서 강력한 성능을 보이지만, 더 광범위한 다중 모달 작업에 대한 능력은 아직 충분히 탐구되지 않음

### 향후 연구 방향

1. **더 다양하고 대규모의 supervision 통합**:
   - Stuff 카테고리 및 복잡한 관계 추론을 개선하기 위한 추가 학습 데이터
   - 부정 표현 및 모호한 쿼리 처리를 위한 특수 데이터셋 구축

2. **강화 학습 방법 도입**:
   - 일반적인 MLLM 능력 향상
   - 복잡한 추론 작업에 대한 성능 개선

3. **더 복잡한 공간 관계 모델링**:
   - 그래프 기반 접근법으로 객체 간 관계 명시적 모델링
   - Attention mechanism 개선으로 관계 추론 강화

4. **다중 모달 작업 확장**:
   - Visual Question Answering (VQA)
   - Image Captioning
   - Visual Reasoning
   - Multi-task learning 프레임워크 개발

5. **더 효율적인 아키텍처**:
   - 모바일 및 엣지 디바이스를 위한 경량화
   - 추론 속도 개선

6. **해석 가능성 향상**:
   - 모델이 특정 위치를 선택한 이유에 대한 설명 제공
   - Attention visualization 및 분석 도구 개발

7. **3D 공간 이해로 확장**:
   - 3D 환경에서의 visual grounding
   - Depth 정보 통합

8. **다국어 지원**:
   - 현재 주로 영어 기반, 다국어 referring expressions 처리 능력 확장

## 🎓 결론

VPP-LLaVA는 Visual Position Prompt (VPP)를 통해 MLLMs의 visual grounding 능력을 크게 향상시키는 효율적이고 효과적인 프레임워크이다. Global VPP는 축 형태의 학습 가능한 전역 위치 참조를 제공하고, Local VPP는 동적 객체 위치 임베딩으로 세밀한 위치 파악을 지원한다.

**주요 성과**:
- 0.6M 샘플의 컴팩트한 VPP-SFT 데이터셋으로 학습
- 21M+ 샘플을 사용한 기존 모델들을 능가하는 SOTA 성능
- RefCOCO/RefCOCO+/RefCOCOg 벤치마크에서 최고 또는 경쟁력 있는 성능
- ReferIt 및 GSEval-BBox에서 강력한 zero-shot 일반화 능력
- 특히 part-object (83.8%) 및 multi-object (69.5%) 시나리오에서 탁월
- 다른 MLLM 아키텍처(LLaVA-NeXT)로의 효과적인 전이 가능성 입증

**핵심 기여**:
1. 명시적 위치 참조를 통한 공간 정보-좌표 정렬 개선
2. 고품질 VPP-SFT 데이터셋으로 데이터 효율성 입증
3. Global-Local VPP의 상호보완적 설계로 시너지 효과 창출
4. 다양한 벤치마크에서 우수한 성능 및 일반화 능력 검증

본 연구는 MLLMs에서 위치 인식 작업을 개선하기 위한 효율적이고 확장 가능한 접근법을 제시하며, visual grounding 연구의 새로운 방향을 제시한다.
