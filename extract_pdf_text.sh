#!/bin/bash
# PDF에서 텍스트를 빠르게 추출하는 헬퍼 스크립트

if [ $# -eq 0 ]; then
    echo "Usage: ./extract_pdf_text.sh <pdf_file>"
    exit 1
fi

PDF_FILE="$1"

if [ ! -f "$PDF_FILE" ]; then
    echo "Error: File not found: $PDF_FILE"
    exit 1
fi

# 첫 20페이지만 추출 (빠른 확인용)
pdftotext -l 20 "$PDF_FILE" -

echo ""
echo "---"
echo "위 내용은 처음 20페이지입니다."
echo "전체 내용을 보려면: pdftotext \"$PDF_FILE\" -"
