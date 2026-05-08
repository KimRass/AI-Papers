#!/usr/bin/env python3
"""
개별 PDF 파일을 마크다운으로 요약
"""

import sys
import subprocess
from pathlib import Path
import re

def extract_text_from_pdf(pdf_path: Path) -> str:
    """PDF에서 텍스트 추출"""
    try:
        result = subprocess.run(
            ['pdftotext', str(pdf_path), '-'],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return ""

def parse_paper_structure(text: str) -> dict:
    """논문 텍스트에서 주요 섹션 추출"""
    sections = {
        'title': '',
        'abstract': '',
        'introduction': '',
        'related_work': '',
        'method': '',
        'experiments': '',
        'results': '',
        'conclusion': ''
    }

    # 제목 추출
    lines = text.split('\n')
    non_empty_lines = [l.strip() for l in lines if l.strip()]
    if non_empty_lines:
        sections['title'] = non_empty_lines[0]

    # Abstract
    abstract_match = re.search(
        r'(?:Abstract|ABSTRACT)\s*\n(.*?)(?:\n\s*\n|(?:\n\s*)?(?:1\s+Introduction|Introduction|1\.|Keywords))',
        text, re.DOTALL | re.IGNORECASE
    )
    if abstract_match:
        sections['abstract'] = abstract_match.group(1).strip()

    # Introduction
    intro_match = re.search(
        r'(?:1\s+Introduction|Introduction|1\.)\s*\n(.*?)(?:\n\s*(?:2\s+|2\.|Related Work|Background|Method|Approach))',
        text, re.DOTALL | re.IGNORECASE
    )
    if intro_match:
        sections['introduction'] = intro_match.group(1).strip()

    # Related Work
    related_match = re.search(
        r'(?:2\s+Related Work|Related Work|2\.|Background)\s*\n(.*?)(?:\n\s*(?:3\s+|3\.|Method|Approach|Methodology))',
        text, re.DOTALL | re.IGNORECASE
    )
    if related_match:
        sections['related_work'] = related_match.group(1).strip()

    # Method
    method_match = re.search(
        r'(?:3\s+Method|Method|Approach|Methodology|3\.)\s*\n(.*?)(?:\n\s*(?:4\s+|4\.|Experiment|Results|Evaluation))',
        text, re.DOTALL | re.IGNORECASE
    )
    if method_match:
        sections['method'] = method_match.group(1).strip()

    # Experiments
    exp_match = re.search(
        r'(?:4\s+Experiment|Experiments|Evaluation|4\.)\s*\n(.*?)(?:\n\s*(?:5\s+|5\.|Results|Discussion|Conclusion))',
        text, re.DOTALL | re.IGNORECASE
    )
    if exp_match:
        sections['experiments'] = exp_match.group(1).strip()

    # Results
    results_match = re.search(
        r'(?:Results|Discussion)\s*\n(.*?)(?:\n\s*(?:Conclusion|6\s+|6\.|References))',
        text, re.DOTALL | re.IGNORECASE
    )
    if results_match:
        sections['results'] = results_match.group(1).strip()

    # Conclusion
    conclusion_match = re.search(
        r'(?:Conclusion|Conclusions|Summary)\s*\n(.*?)(?:\n\s*(?:References|Acknowledgment|Appendix))',
        text, re.DOTALL | re.IGNORECASE
    )
    if conclusion_match:
        sections['conclusion'] = conclusion_match.group(1).strip()

    return sections

def truncate(text: str, max_length: int = 3000) -> str:
    """텍스트가 너무 길면 자르기"""
    if not text or text == '':
        return None
    if len(text) > max_length:
        return text[:max_length] + "\n\n...(계속)"
    return text

def create_markdown(pdf_path: Path, sections: dict) -> str:
    """마크다운 문서 생성"""
    pdf_name = pdf_path.stem

    md_content = f"""# {sections['title'] or pdf_name}

## 논문 정보

**파일명**: `{pdf_path.name}`

**위치**: `{pdf_path}`

---

## Abstract

{sections['abstract'] or '(추출 실패)'}

---

## 1. Introduction

{truncate(sections['introduction']) or '(추출 실패)'}

---

## 2. Related Work

{truncate(sections['related_work']) or '(이 섹션은 논문에 없거나 추출에 실패했습니다)'}

---

## 3. Method / Approach

{truncate(sections['method']) or '(추출 실패)'}

---

## 4. Experiments

{truncate(sections['experiments']) or '(추출 실패)'}

---

## 5. Results

{truncate(sections['results']) or '(이 섹션은 논문에 없거나 추출에 실패했습니다)'}

---

## 6. Conclusion

{truncate(sections['conclusion']) or '(추출 실패)'}

---

## Notes

- 📝 이 문서는 PDF에서 자동 추출되었습니다
- 📄 전체 내용은 원본 PDF를 참조하세요
- ⚠️ 수식, 그림, 표는 추출되지 않았습니다

---

## References

📎 원본 논문: [{pdf_path.name}](./{pdf_path.name})
"""

    return md_content

def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_pdf.py <pdf_file>")
        print("Example: python summarize_pdf.py paper.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != '.pdf':
        print(f"❌ Not a PDF file: {pdf_path}")
        sys.exit(1)

    print(f"📄 Processing: {pdf_path.name}")

    # PDF에서 텍스트 추출
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print(f"❌ Failed to extract text from {pdf_path.name}")
        sys.exit(1)

    # 섹션 파싱
    sections = parse_paper_structure(text)

    # 마크다운 생성
    md_content = create_markdown(pdf_path, sections)

    # 저장
    md_path = pdf_path.with_suffix('.md')
    md_path.write_text(md_content, encoding='utf-8')

    print(f"✅ Created: {md_path}")
    print(f"📊 Size: {len(md_content)} characters")

if __name__ == '__main__':
    main()
