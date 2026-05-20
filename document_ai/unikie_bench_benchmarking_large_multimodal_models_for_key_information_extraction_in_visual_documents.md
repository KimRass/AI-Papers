# UniKIE-BENCH: Benchmarking Large Multimodal Models for Key Information Extraction in Visual Documents

## 📋 논문 정보
- **제목**: UniKIE-BENCH: Benchmarking Large Multimodal Models for Key Information Extraction in Visual Documents
- **저자**: Yifan Ji, Zhipeng Xu, Zhenghao Liu, Zulong Chen, Qian Zhang, Zhibo Yang, Junyang Lin, Yu Gu, Ge Yu, Maosong Sun
- **소속**: Northeastern University, Tsinghua University, Alibaba Group
- **발표**: arXiv:2602.07038v2 (24 Apr 2026)
- **코드**: https://github.com/NEUIR/UNIKIE-BENCH

---

## 🎯 핵심 요약

Key Information Extraction(KIE)을 위한 기존 벤치마크들은 OCR 의존, 단일 문서 타입 특화, 이질적인 평가 체계로 인해 LMM의 end-to-end KIE 능력을 체계적으로 평가하기 어렵다. **UniKIE-BENCH**는 스키마 가이드 구조적 예측 방식으로 KIE를 통일하고, 시나리오 predefined 스키마를 사용하는 **Constrained-Category** 트랙(4,472문서, 3도메인, 11시나리오)과 문서별 고유 스키마를 사용하는 **Open-Category** 트랙(1,661문서, 4타입, 2언어)으로 구성된다. 15개 LMM 평가 결과, 현재 SOTA 모델도 다양한 스키마·롱테일 필드·복잡 레이아웃에서 심각한 성능 저하를 보이며, 한국어(중국어)·영어 간 격차와 오픈소스·클로즈드 소스 간 격차가 뚜렷하다.

---

## 🧩 핵심 흐름 (직관적 이해)

### 1단계 — 문제: 기존 KIE 벤치마크는 왜 LMM 평가에 적합하지 않은가?

LMM이 문서 이미지에서 직접 정보를 추출하는 end-to-end KIE가 주목받고 있지만, 기존 벤치마크들은 세 가지 구조적 한계를 갖는다.

- **OCR 의존** (DocILE, RealKIE): OCR 어노테이션 기반 → LMM의 순수 추출 능력 평가 불가
- **단일 타입 특화**: 영수증, 청구서 등 특정 문서에만 집중 → 일반화 측정 불가
- **QA 방식의 비효율**: 필드마다 별도 쿼리를 날리는 방식 → N개 필드면 N번 inference, 필드 간 구조 관계 포착 불가

> **비유**: "이름이 뭐야?", "금액이 얼마야?"를 따로따로 묻는 것과, "이 계약서에서 필요한 모든 항목을 한 번에 채워줘"라고 묻는 것의 차이.

### 2단계 — 발견: KIE를 스키마 가이드 구조적 예측으로 재정의

*그렇다면 LMM의 실제 KIE 능력을 어떻게 단일 추론으로 측정할까?*

기존 QA 방식과 달리, 스키마 `s = (F, R)`(추출 필드 집합 F + 필드 간 관계 R)를 문서 이미지와 함께 입력해 한 번에 구조화된 출력을 생성하도록 요구한다.

```
기존 QA:  y_f = M(x, q(f)),  f ∈ F    ← 필드마다 별도 inference
UniKIE:   y^SG = M(x, s)               ← 스키마 통째로 입력, 단일 inference
```

이 방식은 필드 간 의존 관계를 포착하고, 실제 애플리케이션에서 쓰는 방식과 일치한다.

### 3단계 — 두 트랙으로 상보적 평가

*시나리오별 특화 능력과 범용 추출 능력을 어떻게 동시에 측정할까?*

| 트랙 | 스키마 방식 | 규모 | 측정 대상 |
|------|-----------|------|---------|
| Constrained-Category | 시나리오별 predefined 스키마 | 4,472문서, 11시나리오 | 실용적 application 성능 |
| Open-Category | 문서별 고유 스키마 | 1,661문서, 영/중 2언어 | 범용 추출 능력 |

Constrained 트랙은 실제 업무(세금신고, 의료기록, 영수증 등)에서 요구되는 정형 스키마로, Open 트랙은 각 문서가 자체 스키마를 갖기 때문에 모델이 처음 보는 필드 유형에도 대응해야 한다.

> **실제 예시** (Figure 1): 인보이스 이미지 + 스키마(`store_name`, `invoice_num`, `billing_to` 등) → 모델이 `"Athletics Store"`, `"A-2024-INV-0312"` 등을 한 번에 추출.

### 4단계 — 15개 LMM 평가: 격차의 구조

*최신 LMM들은 얼마나 잘 하는가?*

**Constrained 트랙 최고 성능 (Table 4)**:
- 클로즈드: Gemini-3-Pro 82.37, Qwen3-VL-Plus 80.20
- 오픈소스: Qwen3-VL-8B **79.12** (일부 클로즈드 모델 추월)
- 최하: SmolVLM2-2.2B 23.69

**Open 트랙 (Table 5)**:
- 모든 모델에서 영어 → 중국어로 가면 성능 급락
- 한자의 고밀도 구조·명시적 단어 경계 없음이 시각 인식 어렵게 함
- Form 문서가 가장 어렵고, Receipt가 가장 쉬움

**공통 발견**: 스케일만으로 성능 차이 설명 불가 — 같은 크기 모델 간 격차가 크다.

### 5단계 — 오류 분석: 4가지 실패 유형

*왜 성능이 떨어지는가? 어떤 종류의 실수를 하는가?*

| 오류 유형 | 설명 | 실제 예시 |
|---------|------|---------|
| **Visual Perception Failure** | 시각 인식 오류 | 세금 필드: 정답 2.32 → 예측 4.32 (숫자 혼동) |
| **Layout Perception Failure** | 레이아웃 지각 실패 | 주소 필드: 인접한 다른 줄 선택 |
| **Field Interpretation Error** | 필드 의미 혼동 | net total을 gross amount로 매핑 |
| **Hallucinated Prediction** | 환각 예측 | 문서에 없는 주소값 생성 |

> **핵심 발견**: Faithfulness(문서 내용에 근거)가 높을수록 F1이 높지만, faithfulness가 높아도 F1이 낮은 경우 존재 → 정확한 지각만으로는 충분하지 않고, 필드 경계 구분과 올바른 인스턴스 선택도 필요.

---

## 📖 주요 내용

### 벤치마크 구성

**Constrained-Category KIE Track**:
- 3 도메인: Business Transactions, Public Services, Regulated Records
- 11 시나리오: Commercial, Retail, Catering, Accommodation, Administrative, Education, Postal Label, Advertisement, Tax-Compliant, Medical Services, Nutrition Label
- 데이터 수집: 공개 데이터셋 통합 + 시나리오별 스키마 매핑 + 재어노테이션

**Open-Category KIE Track**:
- 4 문서 타입 × 2 언어 (영/중): Receipt, Form, Invoice, Contract
- 문서 재구성 파이프라인: HTML 코드 생성 → 렌더링 → noise 추가 → OCR로 ground truth 추출
- 영어 문서가 중국어 대비 복잡한 스키마 (Form/Invoice는 평균 필드 수 15+)

### 평가 방식

- **메트릭**: 필드 단위 F1 score (exact match)
- **구현**: 모든 모델에 통일된 프롬프트 템플릿, temperature=0, 최대 이미지 해상도 1,605,632 픽셀
- 클로즈드 소스: 공식 API, 오픈소스: vLLM + Flash-Attention

---

## 📊 실험 결과

### Constrained-Category (Table 4, 평균 F1)

| 모델 | 평균 | 비고 |
|------|------|------|
| Gemini-3-Pro | 82.37 | 클로즈드 최강 |
| Qwen3-VL-Plus | 80.20 | |
| Qwen-VL-Max | 77.77 | |
| **Qwen3-VL-8B** | **79.12** | 오픈소스 최강, 일부 클로즈드 초과 |
| MiMo-VL-7B-RL | 68.88 | |
| InternVL3.5-8B | 67.70 | |
| SmolVLM2-2.2B | 23.69 | 최하 |

### Open-Category (Table 5, 평균 F1)

| 모델 | 평균 | 중국어 Avg | 영어 Avg |
|------|------|----------|---------|
| Gemini-3-Pro | 81.65 | 75.96 | 87.06 |
| Qwen3-VL-Plus | 70.61 | 70.95 | 73.45 |
| **Qwen3-VL-8B** | **67.36** | 64.95 | 67.34 |
| MiniCPM-V4.5-8B | 52.60 | 39.46 | 56.44 |
| SmolVLM2-2.2B | 12.83 | 4.10 | 22.36 |

### 기존 KIE 벤치마크 비교 (Table 3)

| 벤치마크 | LMM-Ready | Constrained | Open | #Domains | Size |
|---------|-----------|-------------|------|----------|------|
| OCRBench | ✓ | ✓ | ✗ | — | 200 |
| OCRBenchV2 | ✓ | ✓ | ✗ | — | 800 |
| CC-OCR | ✓ | ✓ | ✓ | — | 2,008 |
| **UniKIE-BENCH** | **✓** | **✓** | **✓** | **3** | **6,133** |

---

## 💡 주요 인사이트

1. **스키마 다양성이 가장 큰 도전**: 롱테일 필드, 다양한 스키마 정의 시 성능 급락 → 단순 스케일업으로 해결 불가
2. **레이아웃 이해가 병목**: 텍스트 인식은 되어도 올바른 공간적 매핑 실패가 많음
3. **한중 언어 격차 구조적**: 중국어 고밀도 자소 구조 + 단어 경계 부재 → 시각적 인식 자체가 더 어려움
4. **Faithfulness ≠ 정확도**: 문서 내 근거를 찾더라도 필드 경계와 올바른 인스턴스 선택이 별도 능력
5. **오픈소스 Qwen3-VL-8B의 선전**: 일부 클로즈드 모델을 초과 — 아키텍처·학습 전략이 스케일보다 중요

---

## 🔬 기술적 세부사항

### 스키마 가이드 KIE 공식화

```
s = (F, R)
  F: 추출 대상 필드 집합
  R: 필드 간 관계 (중첩, 포함 등 구조 관계)

y^SG = M(x, s)   ← 문서 이미지 x, 스키마 s → 구조화된 출력
```

### Open-Category 데이터 생성 파이프라인

```
1. 실제 문서 샘플에서 대표 예시 큐레이션
2. LLM으로 문서 내용·레이아웃 설명 생성
3. LLM으로 HTML 코드 생성 → 렌더링 → 문서 이미지
4. Lightweight noise 추가 (시각적 실사성 강화)
5. OCR로 텍스트 추출 → LMM으로 ground truth key-value 어노테이션
```

---

## 📌 한계점

1. **단일 페이지 한정**: 긴 문서 KIE 미포함 — 페이지 검색·크로스페이지 집계가 필요한 롱 문서는 별도 평가 인프라 필요
2. **Context 길이 제약**: LMM의 컨텍스트 제약 때문에 롱 문서에서 내재적 KIE 능력 격리 어려움
3. **평가 지표 한계**: Exact match F1 → 의미적으로 동등한 표현(날짜 포맷 등) 오답 처리

---

## 🎓 결론

UniKIE-BENCH는 스키마 가이드 단일 추론 방식으로 KIE를 통일해 LMM의 실질적 문서 이해 능력을 체계적으로 측정한다. 15개 모델 실험은 현재 LMM이 다양한 스키마·복잡 레이아웃·중국어 문서에서 뚜렷한 한계를 가짐을 보이며, 레이아웃 인식과 필드 의미 이해 강화가 향후 핵심 과제임을 제시한다.
