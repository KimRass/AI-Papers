# UniKIE-BENCH: Benchmarking Large Multimodal Models for Key Information Extraction in Visual Documents

## 📋 논문 정보
- **제목**: UniKIE-BENCH: Benchmarking Large Multimodal Models for Key Information Extraction in Visual Documents
- **저자**: Yifan Ji, Zhipeng Xu, Zhenghao Liu, Zulong Chen, Qian Zhang, Zhibo Yang, Junyang Lin, Yu Gu, Ge Yu, Maosong Sun
- **소속**: Northeastern University, Tsinghua University, Alibaba Group
- **발표**: arXiv:2602.07038v2 (24 Apr 2026)
- **코드**: https://github.com/NEUIR/UNIKIE-BENCH

---

## 🎯 핵심 요약

Key Information Extraction(KIE)을 위한 기존 벤치마크들은 OCR 의존, 단일 문서 타입 특화, 이질적인 평가 체계로 인해 LMM의 end-to-end KIE 능력을 체계적으로 평가하기 어렵다. **UniKIE-BENCH**는 스키마 가이드 구조적 예측 방식으로 KIE를 통일하고, 시나리오별 predefined 스키마를 사용하는 **Constrained-Category** 트랙(4,472문서, 3도메인, 11시나리오)과 문서별 고유 스키마를 사용하는 **Open-Category** 트랙(1,661문서, 4타입, 2언어)으로 구성된다. 15개 LMM 평가 결과, 현재 SOTA 모델도 다양한 스키마·롱테일 필드·복잡 레이아웃에서 심각한 성능 저하를 보이며, 중국어·영어 간 격차와 오픈소스·클로즈드 소스 간 격차가 뚜렷하다.

---

## 🧩 핵심 흐름 (직관적 이해)

### 1단계 — 문제: 기존 KIE 벤치마크는 왜 LMM 평가에 적합하지 않은가?

LMM이 문서 이미지에서 직접 정보를 추출하는 end-to-end KIE가 주목받고 있지만, 기존 벤치마크들은 세 가지 구조적 한계를 갖는다.

- **OCR 의존** (DocILE, RealKIE): OCR 어노테이션 기반 → LMM의 순수 추출 능력 평가 불가
- **단일 타입 특화**: 영수증·청구서 등 특정 문서에만 집중 → 일반화 측정 불가
- **QA 방식의 비효율**: 필드마다 별도 쿼리 → N개 필드면 N번 inference, 필드 간 구조 관계 포착 불가

> **비유**: "이름이 뭐야?", "금액이 얼마야?"를 따로따로 묻는 것과, "이 계약서에서 필요한 모든 항목을 한 번에 채워줘"라고 묻는 것의 차이.

### 2단계 — 발견: KIE를 스키마 가이드 구조적 예측으로 재정의

*그렇다면 LMM의 실제 KIE 능력을 어떻게 단일 추론으로 측정할까?*

스키마 `s = (F, R)`(추출 필드 집합 F + 필드 간 관계 R)를 문서 이미지와 함께 입력해 한 번에 구조화된 출력을 생성하도록 요구한다.

```
기존 QA:  y_f = M(x, q(f)),  f ∈ F    ← 필드마다 별도 inference
UniKIE:   y^SG = M(x, s)               ← 스키마 통째로 입력, 단일 inference
```

> **실제 예시** (Figure 1): 인보이스 이미지 + 스키마(`store_name`, `invoice_num`, `billing_to` 등) → `"Athletics Store"`, `"A-2024-INV-0312"` 등을 한 번에 추출.

### 3단계 — 두 트랙으로 상보적 평가

| 트랙 | 스키마 방식 | 규모 | 측정 대상 |
|------|-----------|------|---------|
| Constrained-Category | 시나리오별 predefined 스키마 | 4,472문서, 11시나리오 | 실용적 application 성능 |
| Open-Category | 문서별 고유 스키마 | 1,661문서, 영/중 2언어 | 범용 추출 능력 |

Constrained 트랙의 데이터 소스는 다양하며, **Postal Label 시나리오는 HW-FORMS(필기체 폼 데이터셋)를 포함**한다 (Appendix A.2, Table 6).

### 4단계 — 15개 LMM 평가: 격차의 구조

**Constrained 트랙 최고 성능**: Gemini-3-Pro 82.37, Qwen3-VL-8B(오픈소스) 79.12로 일부 클로즈드 모델 초과.

**Open 트랙**: 모든 모델에서 영어 → 중국어 성능 급락. 한자의 고밀도 구조·단어 경계 부재가 원인.

문서 타입별 난이도: **Form > Invoice/Contract > Receipt** (Form이 가장 어렵고, Receipt가 가장 쉬움).

> **주목할 실패 사례** (Figure 13, Medical Services): 의료 문서의 개인정보 마스킹(redaction) 영역에서 LMM들이 그럴듯하지만 틀린 값을 생성(환각). 보안 처리된 문서에서의 구조적 취약점.

### 5단계 — 오류 분석: 4가지 실패 유형과 내부 메커니즘

| 오류 유형 | 설명 | 실제 예시 |
|---------|------|---------|
| **Visual Perception Failure** | 숫자·문자 혼동 | 세금 2.32 → 예측 4.32 |
| **Layout Perception Failure** | 인접 줄 혼동 | 주소 필드에 다른 줄 선택 |
| **Field Interpretation Error** | 필드 의미 혼동 | net total → gross amount 매핑 |
| **Hallucinated Prediction** | 문서에 없는 값 생성 | 마스킹된 이름 자리에 임의 이름 |

**Attention 분석** (Appendix A.8, Qwen3-VL-8B): 얕은 레이어는 필드명 레이블에 집중 → 깊은 레이어는 해당 값 위치로 attention 이동. 필드명이 없는 경우 깊은 레이어의 attention이 여러 후보로 분산되어 정확도 저하.

---

## 📖 주요 내용

### 벤치마크 구성

**Constrained-Category KIE Track** (Table 6):

| 도메인 | 시나리오 | 데이터 소스 | 샘플 수 |
|------|---------|-----------|-------|
| Business Transactions | Commercial | SIBR, DocILE | 620 |
| | Retail | SROIE | 347 |
| | Catering Services | CORD, CELL | 212 |
| | Accommodation | SIBR | 40 |
| Public Services | Administrative | CELL, FUNSD | 385 |
| | Education | EPHOIE, CELL | 320 |
| | Postal Label | **HW-FORMS** (필기체) | 500 |
| | Advertisement | DeepForm | 71 |
| Regulated Records | Tax-Compliant | Nanonets-KIE | 987 |
| | Medical Services | SIBR | 240 |
| | Nutrition Label | POIE | 750 |

**Open-Category KIE Track**: 영/중 × {Receipt, Form, Invoice, Contract} = 8개 조합, 총 1,661문서

### 데이터 생성 파이프라인 (Open-Category)

1. 실제 문서 예시 큐레이션 → GPT-4o로 문서 구조 설명 생성
2. LLM으로 HTML 생성 → 렌더링 → 문서 이미지
3. **Blender로 3D 렌더링**: 현실적 조명·접힘·그림자 시뮬레이션
4. 카메라 촬영 + 모션 블러·가우시안 블러·원근 왜곡 적용
5. 프린터 아티팩트(잉크 번짐, 드럼 노이즈 등) 추가
6. OCR → LMM으로 ground truth 키-값 어노테이션

### 어노테이션 지침 (Appendix A.10)

| | Constrained Track | Open Track |
|---|---|---|
| 입력 | 문서 이미지 + 시나리오 + predefined 스키마 | 문서 이미지 + OCR 결과 |
| 작업 | 스키마 필드에 값 채우기 | OCR 교정 → 스키마 직접 설계 |
| 제약 | 보이는 텍스트에만 근거, 없으면 missing 표기 | 추론·완성·환각 금지 |

### 평가 프로토콜

- **메트릭**: 필드 단위 F1 score (exact match)
- **프롬프트** (Figure 7): "JSON 스키마의 빈칸을 이미지 정보로 채워라. 출력은 valid JSON만"
- temperature=0, 최대 해상도 1,605,632픽셀
- 오픈소스: vLLM + Flash-Attention, NVIDIA A100 GPU × 2

---

## 📊 실험 결과

### Constrained-Category (Table 4, 평균 F1)

| 모델 | 평균 | 비고 |
|------|------|------|
| Gemini-3-Pro | 82.37 | 클로즈드 최강 |
| Qwen3-VL-Plus | 80.20 | |
| Qwen-VL-Max | 77.77 | |
| GPT-4o | 69.15 | |
| Claude-Sonnet-4.5 | 66.39 | |
| **Qwen3-VL-8B** | **79.12** | **오픈소스 최강, Qwen-VL-Max 초과** |
| MiMo-VL-7B-RL | 68.88 | |
| InternVL3.5-8B | 67.70 | |
| MiniCPM-V4.5-8B | 67.15 | |
| SmolVLM2-2.2B | 23.69 | 최하 |

### Open-Category (Table 5, 평균 F1)

| 모델 | 전체 평균 | 중국어 평균 | 영어 평균 |
|------|---------|----------|---------|
| Gemini-3-Pro | 81.65 | 75.96 | 87.06 |
| Qwen3-VL-Plus | 70.61 | 70.95 | 73.45 |
| **Qwen3-VL-8B** | **67.36** | 64.95 | 67.34 |
| Kimi-VL-A3B | 58.56 | 53.65 | 64.27 |
| InternVL3.5-8B | 49.38 | 39.43 | 50.06 |
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

1. **스키마 다양성이 가장 큰 도전**: 롱테일 필드·다양한 스키마 시 성능 급락 — 단순 스케일업으로 해결 불가
2. **필기체 포함**: Postal Label 시나리오(HW-FORMS)로 인쇄체에 편중된 기존 벤치마크의 한계를 보완
3. **레이아웃 이해가 병목**: Form 문서의 체크박스·셀렉션 필드, Advertisement의 복잡한 표 레이아웃에서 LMM이 특히 취약
4. **중국어 문서 구조적 열세**: 고밀도 자소·단어 경계 부재 → 시각 인식 자체가 더 어렵고, 영문 중심 사전학습 데이터 불균형
5. **Privacy redaction = 환각 트리거**: 마스킹된 필드를 모델이 채워야 한다고 판단해 그럴듯한 값을 생성
6. **Faithfulness ≠ 정확도**: 문서 내 근거를 찾더라도 필드 경계 구분과 올바른 인스턴스 선택이 별도로 필요
7. **Qwen3-VL-8B의 선전**: 오픈소스로 일부 클로즈드 모델 초과 — 아키텍처·학습 전략이 스케일보다 중요

---

## 🔬 기술적 세부사항

### 스키마 가이드 KIE 공식화

```
s = (F, R)
  F: 추출 대상 필드 집합
  R: 필드 간 관계 (그룹핑, 포함 등 계층 구조)

y^SG = M(x, s)   ← 문서 이미지 x + 스키마 s → 구조화된 JSON 출력
```

### Attention 레이어별 행동 (Appendix A.8)

**필드명이 명시적인 경우** (Figure 8):
- 얕은 레이어: 필드 레이블 위치에 집중 → 의미적 앵커 확립
- 깊은 레이어: 레이블에서 대응 값 위치로 이동 → 추출 완료

**필드명이 암묵적인 경우** (Figure 9):
- 얕은 레이어: 의미적으로 관련된 후보 구문들에 분산
- 깊은 레이어: 여러 후보 위치로 분산 유지 → 비교·추론을 통해 정답 선택

### 평가 지표

- **Sen.Acc**: 줄 단위 완전 일치 정확도
- **F1**: 필드 단위 (exact match, character mismatch = 오답)
- **Faithfulness Rate**: 예측값이 OCR 결과에 존재하는 비율 (환각 탐지)

---

## 📌 한계점

1. **단일 페이지 한정**: 롱 문서 KIE 미포함 — 크로스페이지 집계 필요한 시나리오 제외
2. **Context 길이 제약**: LMM 컨텍스트 제한으로 롱 문서에서의 내재적 능력 격리 어려움
3. **Exact match 한계**: 날짜 포맷·단위 표기 등 의미적으로 동등한 표현을 오답 처리
4. **합성 문서(Open Track)**: 실제 문서와 시각적으로 유사하지만 완전히 동일하지 않을 수 있음

---

## ❓ Q&A

**Q. 어떤 필드를 뽑으라는 지시를 어떻게 내렸는가?**

Section 4 + Appendix A.6 (Figure 7). 프롬프트 템플릿에 JSON 스키마를 직접 삽입하는 방식이다.

```
<image>
Suppose you are an information extraction expert.
Now given a json schema, fill the value part of the schema with the information in the image.
Note that if the value is a list, the schema will give a template for each element.
Finally, only legal json is required as the output.
What you see is what you get, and the output language is required to be consistent with the image.
No explanation is required.
Please output the results as required. The input json schema content is as follows:
<schema>
```

필드명을 key로, 빈 문자열 `""`를 value로 채운 JSON 스키마를 넘기면 모델이 빈칸을 채운다. 필드 설명(description)은 의도적으로 제거해 모델이 필드명과 맥락만으로 의미를 추론하게 했다.

---

**Q. Constrained와 Open 트랙에서 필드 지시 방식이 어떻게 다른가?**

Section 3.1 + Appendix A.4, A.6.

**Constrained-Category**: 시나리오마다 공통으로 정의된 predefined 스키마를 사용한다. 같은 시나리오의 모든 문서에 동일한 필드 집합이 적용된다.

```json
// 예) Tax-Compliant 시나리오 — 모든 세금 문서에 동일한 키 적용
{ "taxpayer_name": "", "tax_id": "", "filing_date": "", ... }
```

**Open-Category**: 문서마다 그 문서 고유의 스키마가 붙는다. 어노테이터가 각 문서를 보고 직접 설계한 것으로, 같은 Receipt라도 문서마다 필드가 달라질 수 있다.

```json
// 예) Figure 10 — 특정 영수증 한 장의 고유 스키마
{ "receipt_number": "", "items": [{"subtotal": "", "unit_price": "",
  "item_name": "", "quantity": ""}], "total": "",
  "transaction_time": "", "store_address": "", "store_name": "" }
```

**공통점**: 두 트랙 모두 동일한 프롬프트 템플릿(Figure 7)에 스키마만 교체하는 방식이며, 필드 설명은 제공하지 않는다.

---

## 🎓 결론

UniKIE-BENCH는 스키마 가이드 단일 추론 방식으로 KIE를 통일해 LMM의 실질적 문서 이해 능력을 체계적으로 측정한다. 필기체(HW-FORMS)를 포함한 다양한 문서 타입, 영/중 이중 언어, 6,133개 문서 규모로 기존 벤치마크의 한계를 극복한다. 15개 모델 실험은 현재 LMM이 복잡 레이아웃·중국어 문서·롱테일 필드에서 뚜렷한 한계를 가짐을 보이며, 레이아웃 인식과 필드 의미 이해 강화가 향후 핵심 과제임을 제시한다.
