# Think Twice, Act Once: Verifier-Guided Action Selection For Embodied Agents

## 📋 논문 정보
- **제목**: Think Twice, Act Once: Verifier-Guided Action Selection For Embodied Agents
- **저자**: Nishad Singhi, Christian Bialas, Snehal Jauhri, Vignesh Prasad, Georgia Chalvatzaki, Marcus Rohrbach, Anna Rohrbach
- **소속**: Technical University of Darmstadt & hessian.AI
- **발표**: arXiv:2605.12620v1 (12 May 2026)
- **코드**: 논문 내 GitHub 링크 제공

---

## 🎯 핵심 요약

MLLM 기반 embodied agent는 CoT 추론을 갖춰도 매 타임스텝마다 단일 행동을 greedy하게 선택하기 때문에 OOD 시나리오와 롱-호라이즌 태스크에서 실수를 스스로 교정할 수 없다. **VEGAS(Verifier-Guided Action Selection)**는 inference 시 N개의 후보 행동을 샘플링하고, 전용 학습된 생성형 검증자(generative verifier)가 각 후보를 평가해 가장 신뢰할 수 있는 행동만 실행하는 test-time 프레임워크다. 핵심 발견은 off-the-shelf MLLM을 검증자로 쓰면 성능 향상이 없다는 것으로, 이를 해결하기 위해 LLM이 자동으로 다양한 실패 궤적과 검증 어노테이션을 합성하는 파이프라인을 도입했다. LangR에서 CoT 대비 65%→71%, Multiple Objects 태스크에서 **+36% 상대 성능 향상**, 3B 검증자가 20× 큰 72B 정책까지 개선하는 크로스 모델 일반화를 달성한다.

---

## 🧩 핵심 흐름 (직관적 이해)

### 1단계 — 문제: Greedy 선택은 실수를 교정할 기회가 없다

MLLM 기반 에이전트는 매 타임스텝마다 CoT 추론 후 가장 확률 높은 행동 하나를 즉시 실행한다. 인간은 여러 선택지를 머릿속으로 평가한 뒤 가장 유망한 것을 선택하는 반면, 에이전트는 **단 하나의 greedy 선택에 즉시 커밋**한다.

- "banana 가져와"는 성공해도 "yellow curved fruit 가져와"라는 paraphrase에서 실패
- 단일 객체 pick-and-place 학습 후 "apple 청소 → cabinet 수납"처럼 멀티스텝으로 확장되면 연쇄 오류 발생

> **비유**: 체스에서 첫 직감으로 바로 말을 두는 것과, 가능한 수를 몇 가지 탐색해 최선을 고른 후 두는 것의 차이.

> **실제 예시** (Figure 1): "Find a sports object and place it on the counter" 태스크 — 표준 정책은 `pick(sponge)` 선택. 스펀지는 스포츠 객체가 아니므로 실패. VEGAS는 후보 중 `pick(ball)`을 검증자가 선택해 올바르게 실행.

그래서 다음 단계에서는 "단일 greedy 선택" 대신 여러 후보를 샘플링하면 어떤 이점이 있는지 살펴본다.

### 2단계 — 발견: N개를 샘플링하면 정답이 포함될 확률이 급등한다

*그렇다면 여러 후보를 뽑으면 얼마나 유리해질까?*

동일한 정책에서 N개의 후보를 샘플링하면, 그 중 적어도 하나의 정답 행동이 포함될 확률이 N에 따라 급격히 높아진다.

| N (후보 수) | 정답 포함 확률 |
|-----------|-------------|
| 2 | 68.1% |
| 4 | 79.4% |
| 10 | **89.4%** |

이제 문제는 N개 중 **어떤 후보가 정답인지 골라내는 것**이다. 이를 위해 검증자가 필요하지만, 여기서 중요한 발견이 있다.

> **비유**: 시험 답을 두 가지로 압축했을 때, 어느 쪽이 더 타당한지 검토하는 과정.

그래서 다음 단계에서는 off-the-shelf MLLM을 검증자로 쓸 수 없는 이유와, 전용 학습이 왜 필수인지를 다룬다.

### 3단계 — 핵심 발견: Off-the-shelf MLLM 검증자는 효과가 없다

*범용 MLLM을 zero-shot으로 검증자에 쓰면 어떨까?*

실험 결과, Zero-Shot 검증자(+ZS Verifier)는 CoT 기준선(65%) 대비 오히려 소폭 하락(64%)한다. 범용 언어 이해만으로는 embodied 환경의 행동 정확성을 판별하기에 부족하다.

전용 검증자 학습을 위해 성공/실패 예시가 모두 필요하지만, **기존 데이터셋에는 실패 예시가 없다**. 이를 해결하는 LLM 기반 합성 파이프라인을 도입한다:

```
성공 궤적 τ+
  → 교사 LLM(o3)으로 각 행동에 CoT 추가 → D+_CoT
  → o3로 실패 궤적 τ- 합성 (틀린 객체·receptacle, 전제조건 위반 등)
  → τ+, τ- 모든 행동에 verification 어노테이션 자동 생성
  → 정답/오답 균형 데이터셋 → verifier fine-tuning
```

> **실제 예시** (Figure 2): `τ+`에서 "find TennisRacket → pick TennisRacket" 순서 정상 실행. `τ-`는 탐색 없이 바로 `pick(TennisRacket)` 시도 → verification: "Pre-condition(racket located and within reach) is violated. `is_action_correct: no`"

그래서 다음 단계에서는 학습된 검증자로 Best-of-N 선택을 어떻게 수행하는지를 다룬다.

### 4단계 — VEGAS 작동: N개 샘플링 → 검증 → 최고점 실행

*학습된 생성형 검증자는 어떻게 후보를 평가하는가?*

각 타임스텝 t에서 VEGAS는 다음 절차를 반복한다:

1. 정책에서 N=16개 후보 `(c_t^(n), a_t^(n))` 샘플링 (CoT rationale + 행동)
2. 각 후보를 검증자에 M=5회 통과 → verification CoT + `action_is_correct: yes/no` 생성
3. `yes`→1, `no`→0 변환 후 평균 → 후보별 점수 σ
4. 최고점 행동 실행: `a_t = argmax σ_t^(n)`

Generative verifier는 단순 점수가 아니라 **추론 과정을 생성**한 후 판단 → discriminative 방식보다 일관되게 우수.

> **비유**: 수학 시험에서 답만 맞히는 것(discriminative)과, 풀이 과정을 쓰고 검토하는 것(generative)의 차이.

그래서 다음 단계에서는 이 작은 검증자가 더 큰 모델에도 효과적인지를 확인한다.

### 5단계 — 크로스 모델 일반화: 3B 검증자가 72B 정책을 개선한다

*검증자가 학습에 없던 더 큰 정책에도 효과가 있을까?*

Qwen2.5-VL-3B 검증자를 zero-shot 대형 정책과 페어링한 결과 (EB-ALFRED):

| 정책 (zero-shot) | 기본 | +VEGAS |
|----------------|------|--------|
| Qwen2.5-VL-72B (20× 큰 모델) | 30% | **38%** |
| Gemma-3-27B | 19% | **25%** |
| InternVL3.5-38B | 24% | **35%** |

검증 능력은 모델 스케일보다 **전용 학습 데이터의 질**에 의존한다. 실용적 함의: 대형 모델 fine-tuning이 어려운 환경에서도 소형 전용 검증자만으로 성능을 끌어올릴 수 있다.

---

## 📖 주요 내용

### 연구 배경 및 문제점

- MLLM 기반 embodied agent가 CoT 추론으로 성능 향상 중이나, OOD 시나리오에서 여전히 취약
- 핵심 원인: **greedy 선택 + 자기교정 불가** — 실수를 인식하지 못한 채 다음 행동으로 진행
- scaling test-time compute(Best-of-N + verifier)가 코딩·수학에서 유효하다고 알려졌으나, embodied reasoning에 적용한 연구는 미비
- 기존 데이터셋에 실패 예시 부재 → 검증자 학습 데이터 구축 자체가 도전

### 제안 방법

**문제 공식화**:
```
상태: 부분 관측 순차 의사결정 (POMDP)
에이전트: π(a_t | I, o_{1:t}, a_{1:t-1})
행동 공간 A: pick(object), navigate(receptacle), place(object), open/close(receptacle) 등
저수준 실행: oracle low-level policy에 위임
```

**정책 학습**: 전문가 시연에서 imitation learning, 출력 토큰에만 손실 적용

**합성 실패 데이터 생성**:
- 교사 LLM(OpenAI o3)으로 성공 궤적에 CoT 추가
- o3로 현실적 실패 궤적 자동 생성 (wrong object, wrong receptacle, precondition violation 등)
- 생성된 τ+, τ- 모든 행동에 `action_is_correct: yes/no` 어노테이션

**VEGAS 추론**:
```
σ_t^(n) = (1/M) Σ_m score(verify(c_t^(n), a_t^(n)))   (yes→1, no→0)
a_t = argmax_{n∈[N]} σ_t^(n)
```

### 실험 환경

| 요소 | 설정 |
|------|------|
| 정책·검증자 | Qwen2.5-VL-3B-Instruct (각각 별도 파인튜닝) |
| 벤치마크 | LangR (Habitat 2.0), EB-ALFRED (AI2-THOR) |
| N (후보 수) | 16 |
| M (검증 횟수/후보) | 5 |
| 학습 GPU | 8×NVIDIA L40 |
| 추론 GPU | L40 (LangR), A100 80GB (ALFRED) |
| 프레임워크 | LLaMAFactory, vLLM |

---

## 📊 실험 결과

### LangR 벤치마크 (Table 1)

| 방법 | 평균 | Rephrasing | Irrelevant Text | Multiple Objects | Conditional |
|------|------|-----------|----------------|----------------|------------|
| No-CoT | 58 | 93 | 39 | 17 | 28 |
| w/ CoT | 65 | 98 | 50 | 25 | 42 |
| + ZS Verifier | 64 | 98 | 50 | 30 | 40 |
| **+ FT Verifier (VEGAS)** | **71** | **99** | **52** | **34** | **48** |

- Multiple Objects(멀티 객체): CoT 25% → VEGAS 34% = **+36% 상대 향상**
- ZS Verifier는 CoT 대비 오히려 소폭 하락 → 전용 학습 필수

### EB-ALFRED 벤치마크 (Table 2, 파인튜닝 정책)

| 방법 | 평균 | Long Horizon | Visual Appearance |
|------|------|-------------|-----------------|
| Qwen3B w/ CoT | 44 | 22 | 48 |
| +ZS Verifier | 44 | 24 | 46 |
| **+VEGAS** | **49** | **34** | **41** |
| Gemma-4B w/ CoT | 48 | 28 | 56 |
| **+VEGAS** | **51** | **25** | **53** |

### 후보 수 스케일링 (Figure 6, EB-ALFRED)

동일 LLM 호출 수 기준으로 VEGAS vs Self-Consistency 비교:
- VEGAS: N 증가에 따라 가파르고 일관된 향상 (N=16에서 ~49%)
- Self-Consistency: 완만하게 증가 후 정체 (~46%)
- → 검증자의 질이 다수결 앙상블보다 test-time compute를 더 효율적으로 활용

### Teacher Model 민감도 (Table 4, LangR)

| 교사 모델 | 평균 성공률 |
|---------|----------|
| w/ CoT (검증자 없음) | 65% |
| + Qwen3-VL-8B-thinking 교사 | 69% |
| + o3 교사 (VEGAS) | **71%** |

저렴한 교사도 의미 있는 향상 제공 → 파이프라인 접근성 확보.

---

## 💡 주요 인사이트

1. **Zero-shot 검증은 역효과**: Best-of-N 패러다임만으로는 부족 — embodied 검증에는 범용 MLLM 언어 이해로 불충분하며 전용 학습 필수
2. **Negative data가 학습 핵심**: 기존 데이터셋은 성공 사례만 있어 검증자 학습 불가 → LLM 기반 실패 자동 합성이 이 논문의 핵심 기여
3. **Generative verifier > Discriminative**: 추론 과정을 생성하면서 판단하는 방식이 일관되게 우수 — 점수만 출력하는 방식보다 해석 가능성도 높음
4. **Cross-model 일반화**: 3B 전용 검증자가 20× 큰 zero-shot 정책까지 개선 — 검증 능력은 모델 스케일보다 데이터 질에 의존
5. **텍스트 전용 검증자도 유효**: CoT가 시각 장면을 충분히 언어로 기술하므로 이미지 입력 제거해도 성능 거의 동일 (LangR: 71% vs 71%)
6. **레이턴시 실용적**: N=16, M=5에서 wall-clock 2배 증가(3s→6s) — 병렬 샘플링으로 총 96 LLM 호출을 8초 내 처리

---

## 🔬 기술적 세부사항

### VEGAS 점수 계산

```
후보 샘플링: (c_t^(1), a_t^(1)), ..., (c_t^(N), a_t^(N))  [온도 0.7]

검증: v_t^(n,m) = Verifier(I, o_{1:t}, a_{1:t-1}, c_t^(n), a_t^(n))
                → verification CoT + "action_is_correct: yes/no"

점수: score = yes→1, no→0
      σ_t^(n) = (1/M) Σ_{m=1}^{M} score(v_t^(n,m))

선택: a_t = argmax_{n∈[N]} σ_t^(n)
```

### CoT 데이터 형식 (Appendix 10.1)

각 행동에 구조화된 추론 태그 부착:
```xml
<task>전체 태스크 설명</task>
<plan>[남은 고수준 단계 목록]</plan>
<subtask>현재 실행할 서브태스크</subtask>
<subtask_reason>이 서브태스크를 선택한 이유</subtask_reason>
<action>pick(apple)</action>
```

### 실패 궤적 생성 유형 (Appendix 10.2)

LLM이 생성하는 현실적 실패 패턴:
- **Wrong object/receptacle**: apple 대신 orange, bed 대신 sofa 선택
- **Precondition violation**: find/open 없이 pick/put 시도
- **Action ordering error**: 세척 전에 객체를 다른 곳에 배치
- **Instance confusion**: 동일 카테고리 내 다른 인스턴스 혼동
- **Partial execution**: 지시의 일부만 수행

### 학습 하이퍼파라미터 (Table 6)

| 파라미터 | 값 |
|---------|---|
| GPU | 8×NVIDIA L40 |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 4 |
| effective batch size | 64 |
| learning_rate | 1e-5 |
| num_train_epochs | 3 |
| lr_scheduler | cosine |
| precision | bf16 |

---

## 📌 한계점

1. **High-level 행동 공간 한정**: oracle low-level policy에 의존 — 저수준 모터 제어의 검증은 미포함
2. **시뮬레이션 환경 한정**: Habitat 2.0, AI2-THOR에서만 검증 — 실제 로봇 환경(real-world)으로의 이전 미확인
3. **교사 모델 의존**: 합성 데이터 품질이 강력한 교사 LLM(o3)에 의존 (단, Qwen3-VL-8B-thinking으로도 의미 있는 향상 가능)
4. **시각적 검증 한계**: CoT가 시각 장면을 언어로 기술하므로 text-only 검증자도 유사 성능 → 미세한 시각 구분(occlusion, fine-grained appearance)이 필요한 태스크에서는 이미지 기반 검증의 추가 이점이 클 것으로 예상

---

## 🎓 결론

VEGAS는 embodied agent의 test-time 강건성을 높이기 위해 N개의 후보 행동을 샘플링하고, 전용 학습된 생성형 검증자가 최선의 행동을 선택하는 프레임워크다. Off-the-shelf MLLM 검증의 한계를 실증하고, LLM 기반 자동 합성 실패 데이터로 이를 극복해 LangR·EB-ALFRED 전반에 걸쳐 일관된 성능 향상과 크로스 모델 일반화를 달성했다. Test-time compute 확장이 embodied agent 성능 개선의 유효한 방향임을 보이며, 작은 전용 검증자가 대형 정책을 보완할 수 있다는 실용적 함의를 제시한다.
