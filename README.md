#<1개 이상의 PDF를 업로드하면 RAG로 답변> Last update. 25.07.17. JUNHWAN

##※주요 특징

-원본 PDF 파일: 임시 파일로 저장 → 처리 완료 후 삭제

-PDF 내용: 텍스트로 변환 → 청크로 분할 → 임베딩 벡터로 변환 → FAISS 벡터 DB에 저장
  
-검색 시: 벡터 DB에서 유사도 검색을 통해 관련 문서 청크를 찾아 답변 생성

https://202507kosolm-sh7fvrljge7ihztar3h7k2.streamlit.app/
