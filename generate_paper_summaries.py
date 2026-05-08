#!/usr/bin/env python3
"""
VLM 폴더의 모든 논문에 대해 마크다운 정리 파일 생성
"""

import os
import subprocess
from pathlib import Path
import re

VLM_DIR = Path("/Users/eric/workspace/AI-Papers/vision_language/vlm")

def extract_text_from_pdf(pdf_path: Path, max_pages: int = None) -> str:
    """PDF에서 텍스트 추출"""
    try:
        cmd = ['pdftotext', str(pdf_path), '-']
        if max_pages:
            cmd = ['pdftotext', '-l', str(max_pages), str(pdf_path), '-']

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout
    except Exception as e:
        print(f"Error extracting text from {pdf_path.name}: {e}")
        return ""

def parse_paper_structure(text: str) -> dict:
    """논문 텍스트에서 주요 섹션 추출"""
    sections = {
        'title': '',
        'authors': '',
        'abstract': '',
        'introduction': '',
        'related_work': '',
        'method': '',
        'experiments': '',
        'results': '',
        'conclusion': '',
        'full_text': text
    }

    # 제목 추출 (보통 첫 몇 줄)
    lines = text.split('\n')
    non_empty_lines = [l.strip() for l in lines if l.strip()]
    if non_empty_lines:
        sections['title'] = non_empty_lines[0]

    # Abstract 추출
    abstract_match = re.search(
        r'(?:Abstract|ABSTRACT)\s*\n(.*?)(?:\n\s*\n|(?:\n\s*)?(?:1\s+Introduction|Introduction|1\.|Keywords))',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if abstract_match:
        sections['abstract'] = abstract_match.group(1).strip()

    # Introduction 추출
    intro_match = re.search(
        r'(?:1\s+Introduction|Introduction|1\.)\s*\n(.*?)(?:\n\s*(?:2\s+|2\.|Related Work|Background|Method|Approach))',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if intro_match:
        sections['introduction'] = intro_match.group(1).strip()

    # Related Work 추출
    related_match = re.search(
        r'(?:2\s+Related Work|Related Work|2\.|Background)\s*\n(.*?)(?:\n\s*(?:3\s+|3\.|Method|Approach|Methodology))',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if related_match:
        sections['related_work'] = related_match.group(1).strip()

    # Method 추출
    method_match = re.search(
        r'(?:3\s+Method|Method|Approach|Methodology|3\.)\s*\n(.*?)(?:\n\s*(?:4\s+|4\.|Experiment|Results|Evaluation))',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if method_match:
        sections['method'] = method_match.group(1).strip()

    # Experiments 추출
    exp_match = re.search(
        r'(?:4\s+Experiment|Experiments|Evaluation|4\.)\s*\n(.*?)(?:\n\s*(?:5\s+|5\.|Results|Discussion|Conclusion))',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if exp_match:
        sections['experiments'] = exp_match.group(1).strip()

    # Results 추출
    results_match = re.search(
        r'(?:Results|Discussion)\s*\n(.*?)(?:\n\s*(?:Conclusion|6\s+|6\.|References))',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if results_match:
        sections['results'] = results_match.group(1).strip()

    # Conclusion 추출
    conclusion_match = re.search(
        r'(?:Conclusion|Conclusions|Summary)\s*\n(.*?)(?:\n\s*(?:References|Acknowledgment|Appendix))',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if conclusion_match:
        sections['conclusion'] = conclusion_match.group(1).strip()

    return sections

def create_markdown(pdf_path: Path, sections: dict) -> str:
    """마크다운 문서 생성"""
    pdf_name = pdf_path.stem

    # 섹션 내용 정리 (너무 길면 자르기)
    def truncate(text: str, max_length: int = 3000) -> str:
        if not text or text == '':
            return None
        if len(text) > max_length:
            return text[:max_length] + "\n\n...(계속)"
        return text

    md_content = f"""# {sections['title'] or pdf_name}

## 논문 정보

**파일명**: `{pdf_path.name}`

**위치**: `{pdf_path.relative_to(pdf_path.parents[2])}`

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

def process_pdf(pdf_path: Path, force: bool = False) -> bool:
    """PDF 처리 및 마크다운 생성"""
    # 마크다운 파일 경로
    md_path = pdf_path.with_suffix('.md')

    # 이미 존재하면 스킵
    if md_path.exists() and not force:
        print(f"⏭️  Skip (already exists): {pdf_path.name}")
        return False

    print(f"📄 Processing: {pdf_path.name}")

    # PDF에서 텍스트 추출 (전체)
    text = extract_text_from_pdf(pdf_path)

    if not text:
        print(f"⚠️  No text extracted from {pdf_path.name}")
        return False

    # 섹션 파싱
    sections = parse_paper_structure(text)

    # 마크다운 생성
    md_content = create_markdown(pdf_path, sections)

    # 저장
    md_path.write_text(md_content, encoding='utf-8')
    print(f"✅ Created: {md_path.name}")

    return True

def main():
    """메인 함수"""
    import sys

    # 커맨드라인 인자로 force 옵션 확인
    force = '--force' in sys.argv or '-f' in sys.argv

    # VLM 폴더의 모든 PDF 찾기
    pdf_files = sorted(VLM_DIR.glob('*.pdf'))

    print(f"Found {len(pdf_files)} PDF files in {VLM_DIR}")
    if force:
        print("🔄 Force mode: 기존 파일 덮어쓰기")
    print("=" * 60)

    processed = 0
    skipped = 0
    failed = 0

    for pdf_path in pdf_files:
        try:
            if process_pdf(pdf_path, force=force):
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"✅ Processed: {processed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(pdf_files)}")

if __name__ == '__main__':
    main()
