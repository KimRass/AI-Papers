# ask-pdf

특정 PDF 논문의 내용에 대해 질문하는 스킬입니다.

## 사용법

```bash
/ask-pdf <pdf_파일_경로> <질문>
```

## 예시

```bash
/ask-pdf llm/foundation_models/refusal_in_language_models_is_mediated_by_a_single_direction.pdf Difference-in-Means 방법을 자세히 설명해줘
/ask-pdf generative/image_editing/glyphmastero.pdf FPN을 제거하면 성능이 얼마나 떨어져?
/ask-pdf ./my_paper.pdf 이 논문의 한계점은 뭐야?
```

## 동작 방식

1. 지정된 PDF 파일을 읽는다
2. 질문과 관련된 내용을 PDF에서 찾는다
3. 한국어로 답변한다

## 답변 원칙

- **근거 우선**: 답변은 반드시 PDF 본문에서 찾은 내용을 바탕으로 한다
- **출처 명시**: 답변에 사용된 내용이 논문 몇 섹션/페이지에 해당하는지 밝힌다
- **모르면 모른다고**: PDF에 없는 내용은 추측하지 않고 "논문에서 확인되지 않음"이라고 명시한다
- **수식·수치 정확히**: 수식이나 실험 수치는 논문 원문 그대로 인용한다
- **간결하게**: 질문에 직접 답하는 내용만, 불필요한 배경 설명은 생략한다

## 참고사항

- 같은 대화에서 추가 질문을 이어서 할 수 있다 (PDF를 다시 읽지 않아도 됨)
- PDF가 길면 질문과 관련된 섹션을 우선적으로 읽는다
- 같은 폴더에 `.md` 요약 파일이 있으면 함께 참고한다
