1개 이상의 PDF를 업로드하여 RAG로 답변

※주요 특징
-원본 PDF 파일: 임시 파일로 저장 → 처리 완료 후 삭제

-PDF 내용: 텍스트로 변환 → 청크로 분할 → 임베딩 벡터로 변환 → FAISS 벡터 DB에 저장
  PDF 로드: PyPDFLoader로 PDF 내용을 텍스트로 변환
  텍스트 분할: RecursiveCharacterTextSplitter로 작은 청크(chunk)로 분할
  임베딩 생성: OpenAI의 text-embedding-3-small 모델로 텍스트를 벡터로 변환
  벡터 DB 저장: FAISS 벡터 데이터베이스에 임베딩 벡터들을 저장 (임시)
  
-검색 시: 벡터 DB에서 유사도 검색을 통해 관련 문서 청크를 찾아 답변 생성

