# GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing

## 📋 논문 정보
- **제목**: GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing
- **저자**: Tong Wang, Ting Liu, Xiaochao Qu, Chengjing Wu, Luoqi Liu, Xiaolin Hu
- **소속**: MT Lab, Meitu Inc. / Tsinghua University (BNRist, IDG/McGovern Institute)
- **발표**: CVPR 2025
- **arXiv**: (CVPR Open Access)

---

## 🎯 핵심 요약

Scene text editing은 이미지 속 텍스트를 원하는 내용으로 바꾸되 스타일과 주변 배경을 유지하는 태스크다. 기존 diffusion 기반 방법들은 OCR 특징을 그대로 가져다 쓰기 때문에 한자처럼 복잡한 획 구조를 가진 글자를 제대로 생성하지 못했다. **GlyphMastero**는 개별 글자(로컬)와 텍스트 줄 전체(글로벌) 두 레벨의 glyph 특징을 교차 주의(cross-attention)로 융합하는 전용 glyph 인코더를 제안한다. 기존 SOTA 대비 문장 정확도(Sen.Acc) +18.02%, FID 53.28% 감소를 달성했다.

---

## 🧩 핵심 흐름 (직관적 이해)

### 1단계 — 문제: OCR 특징은 "읽는" 도구이지 "그리는" 도구가 아니다

기존 방법(AnyText, DiffUTE)은 사전학습된 OCR 모델의 특징을 diffusion 모델의 가이던스로 그대로 활용한다. 문제는 OCR이 인식에 최적화된 도구라 **획의 계층적 구조** — 개별 획 → 획 간 상호작용 → 글자 전체 구조 — 를 표현하지 못한다는 것이다. 영문자처럼 단순한 글자에서는 괜찮지만, 한자처럼 복잡한 획 구조를 가진 언어에서 치명적이다.

> **비유**: OCR은 글자를 "얼마나 정확히 인식하는가"로 훈련되었지, "획을 어떤 순서로 어떻게 그리는가"로 훈련된 것이 아니다. 필기 선생님과 독해 선생님의 차이.

> **실제 예시** (Figure 5): AnyText와 DiffUTE는 한자 "赶海(beachcombing)", "各位(everybody)"에서 획이 뭉개지거나 다른 글자로 변형되는 오류 발생. GlyphMastero는 원본 스타일을 유지하며 정확히 생성.

### 2단계 — 발견: 텍스트 구조에는 두 레벨이 필요하다

*그렇다면 획 수준 정밀도를 위해 어떤 정보가 필요한가?*

글자 하나를 정확히 생성하려면:
- **로컬 정보**: 각 글자 하나하나의 획 세부사항 → 글자를 낱개로 렌더링한 이미지에서 추출
- **글로벌 정보**: 텍스트 줄 전체의 레이아웃과 맥락 → 전체 줄을 하나의 이미지로 렌더링해 추출

두 정보를 합쳐야 비로소 "이 글자가 이 줄에서 어떤 모양이어야 하는가"를 표현할 수 있다.

> **비유**: 악보 한 음표를 정확히 연주하려면, 해당 음표 자체의 음고(로컬)도 알아야 하고, 앞뒤 음표와의 흐름(글로벌)도 알아야 한다.

### 3단계 — Glyph Attention Module: 로컬 글자가 글로벌 줄을 참조한다

*두 레벨 특징을 어떻게 연결하는가?*

로컬 특징(각 글자)을 **Query**, 글로벌 특징(전체 줄)을 **Key/Value**로 사용해 multi-head cross-attention을 수행한다. 글로벌 특징은 로컬의 시퀀스 길이 N에 맞게 반복(repeat)된 뒤 RoPE 위치 임베딩을 적용한다.

```
o = Attention(Q=l^p, K=g^p, V=g^p)
  l^p = ψ_l(l),  g^p = ψ_g(ĝ)   (ĝ: g를 N번 repeat)
```

이 모듈은 neck 특징(T_n)과 backbone 특징(T_b) 두 레벨에 각각 독립적으로 적용된다.

> **결과 (Ablation)**: T_n만 제거해도 Sen.Acc 평균 43.49% 하락. 글로벌 neck 특징(g_n)만 단독 사용(AnyText 방식과 동등)하면 Sen.Acc 0.10/0.07로 폭락. 로컬-글로벌 교차 주의가 핵심.

### 4단계 — FPN: 얕은 층의 해상도 + 깊은 층의 의미를 동시에

*글로벌 스트림의 OCR 특징만으로는 부족하지 않은가?*

글로벌 스트림에 FPN(Feature Pyramid Network)을 추가해 OCR backbone의 5개 계층 특징을 피라미드 방식으로 융합한다. 얕은 층의 고해상도 획 정보와 깊은 층의 의미 풍부 정보를 모두 살린다.

```
p_i = g_i(u(p_{i+1}) + c_i)   (i = 4,3,2,1)
```

> **결과 (Ablation)**: FPN 제거 시 Sen.Acc 평균 22.42% 하락, CER 증가. 멀티스케일 융합이 전역 텍스트 구조 표현에 필수.

### 5단계 — 최종 가이던스: 두 스트림을 합쳐 diffusion UNet을 조건화

Aggregator가 neck 출력(o_n)과 backbone 출력(o_b)을 연결·투영해 최종 조건 임베딩 `c ∈ R^{N×D}`를 생성한다. 이 임베딩이 SD 2.1 기반 inpainting UNet의 cross-attention을 통해 획 수준 정밀도를 제공한다.

> **실제 예시** (Figure 5): "Xiaxia" → DiffUTE는 "Xiuxia"처럼 변형, AnyText는 폰트 부정확. GlyphMastero는 원본 스타일을 그대로 유지하며 정확히 삽입.

---

## 📖 주요 내용

### 연구 배경

Scene text editing은 두 단계 접근(텍스트 제거 → 폰트 매칭 삽입)으로 시작했으나 실제 장면의 관점 왜곡·조명 변화에 취약. GAN 기반 방법들은 복잡한 스타일 변형에서 한계. Diffusion 기반이 주류가 되었지만 DiffUTE처럼 고정 길이 OCR 특징을 쓰면 복잡 글자에서 품질 저하.

### 전체 아키텍처

```
텍스트 입력 y
  ├── 로컬 스트림: 글자별 렌더링 x_l → OCR backbone → l_b, l_n
  └── 글로벌 스트림: 전체 줄 렌더링 x_g → OCR backbone + FPN → g_n, g_b

Glyph Attention Module T_n: (l_n, g_n) → o_n
Glyph Attention Module T_b: (l_b, g_b) → o_b

Aggregator A: (o_n, o_b) → c ∈ R^{N×D}

c → SD 2.1 inpainting UNet (cross-attention)
```

### 학습 설정

- 베이스: Stable Diffusion 2.1 inpainting
- OCR: PaddleOCR-v4 (frozen)
- 데이터: AnyWord-3M (~3.5M 이미지)
- 배치: 256, 15 epochs, 8× V100-32G
- Glyph Attention: 4-head, d̄=512, d_o=1024

---

## 📊 실험 결과

### 다국어 비교 (Table 1, English / Chinese)

| 방법 | Sen.Acc↑ | CER↓ | FID↓ | LPIPS↓ |
|------|---------|------|------|--------|
| DiffUTE | 0.332 / 0.252 | 0.319 / 0.405 | 14.3 / 24.9 | 0.131 / 0.206 |
| AnyText | 0.607 / 0.580 | 0.173 / 0.209 | 10.4 / 24.9 | 0.110 / 0.198 |
| **GlyphMastero** | **0.817 / 0.730** | **0.074 / 0.134** | **4.61 / 11.89** | **0.055 / 0.101** |

- DiffUTE 대비: Sen.Acc +18.02% (평균), FID 57.95% 감소
- AnyText 대비: Sen.Acc +8.68% (평균), FID 53.28% 감소

### 영어 전용 비교 (Table 2)

| 방법 | Sen.Acc↑ | CER↓ | FID↓ | LPIPS↓ |
|------|---------|------|------|--------|
| TextCtrl | 0.765 | 0.094 | **4.15** | **0.043** |
| **GlyphMastero** | **0.817** | **0.074** | 4.61 | 0.055 |

- 텍스트 정확도 전 방법 최고, FID/LPIPS는 TextCtrl에 소폭 열세

### Ablation (Table 3, English / Chinese Sen.Acc)

| 구성 | Sen.Acc (En/Zh) | 평균 변화 |
|------|----------------|---------|
| Full model | 0.549 / 0.512 | — |
| − FPN | 0.454 / 0.370 | **-22.42%** |
| − T_b | 0.507 / 0.427 | -13.68% |
| − T_n (w/ l_n만) | 0.326 / 0.274 | **-43.49%** |
| − T_n (w/ g_n만) | 0.100 / 0.072 | **최악** |

---

## 💡 주요 인사이트

1. **OCR 특징의 구조적 한계**: 인식용 특징은 생성 가이던스로 충분하지 않음 — 전용 학습 가이던스 필요
2. **계층적 glyph 모델링**: 로컬(획) × 글로벌(줄) 교차 참조가 복잡 문자 생성의 핵심
3. **g_n 단독 사용 = 실패**: 시퀀스 길이 N을 1로 압축하는 순간 세부 획 정보 소실 — AnyText의 구조적 한계
4. **FPN의 필수성**: 멀티스케일 없이 단일 레이어 특징만으로는 전역 텍스트 구조 부족
5. **한자가 기준점**: 영문 전용이 아닌 한자 포함 평가가 glyph 인코더 품질의 진정한 검증

---

## 🔬 기술적 세부사항

### Glyph Attention Module 수식

```
ĝ = repeat(g, N)                       # g를 N번 복제 (시퀀스 맞춤)
l^p = ψ_l(l),  g^p = ψ_g(ĝ)           # 선형 투영
l̄^p, ḡ^p = RoPE(l^p, g^p)             # 위치 임베딩
z = MultiHeadAttention(Q=l̄^p, K=ḡ^p, V=ḡ^p)
o = ψ_o(LayerNorm(z))                  # 출력 투영
```

### FPN 융합

```
p_5 = c_5
p_i = g_i(upsample(p_{i+1}) + c_i),   i = 4,3,2,1
# g_i: 3×3 conv, c_i: 1×1 lateral connection
# 최종 g_b: channel projection + downsampling
```

### 평가 지표

- **Sen.Acc**: 줄 단위 정확도 (OCR 인식 기반)
- **CER**: 글자 단위 오류율
- **FID**: 생성 분포와 실제 분포의 거리 (스타일 유사도)
- **LPIPS**: 샘플 단위 지각적 유사도

---

## 📌 한계점

1. **긴 텍스트 정확도**: 짧은 텍스트 대비 긴 텍스트에서 Sen.Acc 여전히 낮음 — 학습 데이터와 512×512 해상도 제약
2. **해상도 제약**: SD 2.1의 512×512 한계 → 고해상도 장면에서 세밀도 제한
3. **데이터 의존**: AnyWord-3M 특화 학습 → 다른 도메인 일반화 미검증

---

## ❓ Q&A

**Q. 어디에 위치한 어떤 문자를 다른 문자로 대체할지 어떻게 컨트롤하는가?**

두 가지 정보로 독립적으로 제어한다 (Section 3.1, 4.1).

**어디에** → Polygon `P_ij`

각 학습 샘플은 `(T_ij, P_ij)` 쌍으로 구성된다. `P_ij`는 4개의 꼭짓점으로 이루어진 사각형 polygon으로, 이미지에서 수정할 텍스트 영역의 위치를 지정한다. 이 polygon으로부터 binary mask `m`을 생성해 해당 영역만 inpainting 대상으로 지정한다.

```
x_m = x ⊙ (1 - m)          ← mask 영역을 0으로 처리 (기존 텍스트 제거)
z̃_t = [z_t ; m ; E(x_m)]   ← latent에 mask와 masked image를 채널로 concat
```

**어떤 문자로** → Target text string `T_ij`

`T_ij`는 원하는 새 텍스트 문자열이다. 이를 렌더링해 glyph 이미지로 변환하고, GlyphMastero로 처리해 conditioning embedding `c`를 생성한다. `c`가 UNet의 cross-attention을 통해 해당 mask 영역에 해당 글자를 생성하도록 가이드한다.

---

## 🎓 결론

GlyphMastero는 장면 텍스트 편집을 위한 전용 glyph 인코더로, 개별 글자의 로컬 획 정보와 줄 전체의 글로벌 구조 정보를 glyph attention으로 융합해 획 수준 정밀도의 가이던스를 생성한다. 특히 한자 같은 복잡한 문자 체계에서 기존 방법 대비 명확한 우위를 보이며, OCR 특징 직접 활용의 구조적 한계를 전용 학습 가이던스로 극복한 사례를 제시한다.
