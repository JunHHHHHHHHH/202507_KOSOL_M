# rag_logic.py

import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def extract_issue_number(filename):
    """파일명에서 호수를 추출하는 함수"""
    # 주간농사정보 제XX호 패턴을 찾아서 호수 추출
    pattern = r'제(\d+)호'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return "Unknown"

def initialize_rag_chain(openai_api_key, pdf_paths, file_names=None):
    """OpenAI API 키와 PDF 파일 경로 리스트를 받아 RAG 체인을 초기화합니다."""
    print("--- RAG 파이프라인 초기화 시작 ---")
    
    all_docs = []
    
    # 모든 PDF 파일을 순회하며 문서 로드
    for i, pdf_path in enumerate(pdf_paths):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # 파일명에서 호수 추출
            if file_names and i < len(file_names):
                filename = file_names[i]
                issue_number = extract_issue_number(filename)
            else:
                filename = f"Document_{i+1}"
                issue_number = str(i+1)
            
            # 각 문서에 메타데이터 추가
            for doc_idx, doc in enumerate(docs):
                doc.metadata['file_index'] = i
                doc.metadata['document_id'] = i
                doc.metadata['file_name'] = filename
                doc.metadata['document_name'] = filename
                doc.metadata['issue_number'] = issue_number
                
                # 페이지 번호 정보 추가 (PyPDFLoader는 'page' 키로 페이지 번호 제공)
                original_page = doc.metadata.get('page', doc_idx)
                page_num = original_page + 1  # 0부터 시작하므로 +1
                doc.metadata['page_number'] = page_num
                
                # 정확한 출처 정보 생성
                doc.metadata['source_info'] = f"주간농사정보 제{issue_number}호의 {page_num}p"
                
                print(f"메타데이터 추가: {doc.metadata['source_info']}")
            
            all_docs.extend(docs)
            print(f"✅ 파일 {i+1} 로드 완료 - {len(docs)}페이지")
            
        except Exception as e:
            print(f"❌ 파일 {i+1} 로드 실패: {str(e)}")
            raise e
    
    print(f"✅ [1/5] 전체 문서 로드 완료 - 총 {len(all_docs)}페이지")
    
    # 문서가 비어있는지 확인
    if not all_docs:
        raise ValueError("PDF 문서들이 비어있거나 텍스트를 추출할 수 없습니다.")
    
    # 문서 내용 길이 확인
    total_text = ""
    for doc in all_docs:
        total_text += doc.page_content
    
    print(f"전체 텍스트 길이: {len(total_text)} 문자")
    
    if len(total_text.strip()) < 100:
        raise ValueError("문서 내용이 너무 짧습니다. 스캔된 이미지 PDF일 가능성이 있습니다.")
    
    try:
        # 2. 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        splits = text_splitter.split_documents(all_docs)
        
        # 분할된 청크의 메타데이터 확인 및 보정
        for split in splits:
            if 'source_info' not in split.metadata:
                # 원본 문서에서 메타데이터 복구
                issue_num = split.metadata.get('issue_number', 'Unknown')
                page_num = split.metadata.get('page_number', 'Unknown')
                split.metadata['source_info'] = f"주간농사정보 제{issue_num}호의 {page_num}p"
        
        print(f"✅ [2/5] 문서 분할 완료 - 총 {len(splits)}개 청크")
        
        # 분할 결과 확인
        if not splits:
            raise ValueError("문서 분할 결과가 비어있습니다.")
        
        # 첫 번째 청크 메타데이터 확인
        print(f"첫 번째 청크 출처: {splits[0].metadata.get('source_info', 'Unknown')}")
        
        # 3. OpenAI 임베딩 및 벡터 DB 설정
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("✅ [3/5] FAISS 벡터 DB 생성 완료")
        
        # 4. 검색기 생성 (개선된 파라미터)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 20,  # 더 많은 청크 검색
                "score_threshold": 0.2  # 유사도 임계값을 더 낮춤
            }
        )
        print("✅ [4/5] 검색기 생성 완료")
        
        # 5. OpenAI LLM 설정
        template = """당신은 주어진 문맥(context)의 내용을 바탕으로만 질문에 답하는 AI 어시스턴트입니다.

**절대 준수 사항:**
1. 오직 제공된 CONTEXT 내용만 사용하여 답변하세요
2. 외부 지식이나 일반적인 정보는 절대 사용하지 마세요
3. 답변에는 반드시 (출처: 주간농사정보 제○호의 ○p) 형식으로 출처를 명시하세요
4. CONTEXT에서 질문과 관련된 정보를 찾을 수 없다면 "해당 문서들에는 정보가 포함되어 있지 않습니다"라고 답변하세요
5. 웹사이트 URL이나 외부 링크는 절대 사용하지 마세요

CONTEXT: {context}

QUESTION: {question}

답변:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key,
            max_tokens=800,
            timeout=30
        )
        
        def format_docs(docs):
            """문서들을 출처 정보와 함께 포맷팅"""
            if not docs:
                return "검색된 문서가 없습니다."
            
            formatted = []
            for doc in docs:
                source = doc.metadata.get('source_info', 'Unknown')
                content = doc.page_content
                # 디버깅: 출처 정보 확인
                print(f"포맷팅 중인 문서 출처: {source}")
                formatted.append(f"[출처: {source}]\n{content}")
            
            return "\n\n".join(formatted)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✅ [5/5] RAG 체인 생성 완료")
        
        return rag_chain, retriever
        
    except Exception as e:
        print(f"❌ RAG 초기화 중 오류 발생: {str(e)}")
        raise e

def get_answer(chain, retriever, question):
    """RAG 체인과 검색기를 이용하여 답변을 생성합니다."""
    try:
        docs = retriever.get_relevant_documents(question)
        print(f"검색된 문서 개수: {len(docs)}")
        
        if not docs:
            return "해당 문서들에는 정보가 포함되어 있지 않습니다."
        
        # 업로드된 문서에서만 검색되었는지 확인
        valid_docs = []
        for i, doc in enumerate(docs):
            source_info = doc.metadata.get('source_info', 'Unknown')
            print(f"문서 {i+1} 출처: {source_info}")
            print(f"내용: {doc.page_content[:100]}...")
            
            # 주간농사정보 문서인지 확인
            if '주간농사정보' in source_info:
                valid_docs.append(doc)
        
        if not valid_docs:
            return "해당 문서들에는 정보가 포함되어 있지 않습니다."
            
        print(f"유효한 문서 개수: {len(valid_docs)}")
        
    except Exception as e:
        print(f"검색 오류: {e}")
        return "검색 중 오류가 발생했습니다."
    
    return chain.invoke(question)
