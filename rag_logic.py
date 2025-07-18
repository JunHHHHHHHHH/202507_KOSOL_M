# rag_logic.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def initialize_rag_chain(openai_api_key, pdf_paths):
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
            
            # 각 문서에 파일 인덱스 메타데이터 추가
            for doc in docs:
                doc.metadata['file_index'] = i
                doc.metadata['file_name'] = f"Document_{i+1}"
            
            all_docs.extend(docs)
            print(f"✅ 파일 {i+1} 로드 완료 - {len(docs)}페이지")
            
        except Exception as e:
            print(f"❌ 파일 {i+1} 로드 실패: {str(e)}")
            raise e
    
    print(f"✅ [1/5] 전체 문서 로드 완료 - 총 {len(all_docs)}페이지")
   
    # 🔹메타데이터 추가 코드 삽입
    if file_names:
        current_file_idx = 0
        pages_processed = 0
        
        for i, pdf_path in enumerate(pdf_paths):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # 각 문서에 메타데이터 추가
            for doc in all_docs[pages_processed:pages_processed + len(docs)]:
                doc.metadata['document_id'] = i
                doc.metadata['document_name'] = file_names[i]
                # 간단한 키워드 추출 (복잡한 extract_topic 함수 대신)
                doc.metadata['topic'] = extract_simple_keywords(doc.page_content)
            
            pages_processed += len(docs)
            print(f"📋 문서 {i+1} 메타데이터 추가 완료: {file_names[i]}")
  
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
        print(f"✅ [2/5] 문서 분할 완료 - 총 {len(splits)}개 청크")
        
        # 분할 결과 확인
        if not splits:
            raise ValueError("문서 분할 결과가 비어있습니다.")
        
        # 첫 번째 청크 내용 확인
        print(f"첫 번째 청크 내용: {splits[0].page_content[:200]}...")
        
        # 3. OpenAI 임베딩 및 벡터 DB 설정
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("✅ [3/5] FAISS 벡터 DB 생성 완료")
        
        # 4. 검색기 생성
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,  # 다중 문서이므로 더 많은 청크 검색
                "score_threshold": 0.5  # 유사도 임계값 설정
            }
        )
        print("✅ [4/5] 검색기 생성 완료")
        
        # 5. OpenAI LLM 설정
        template = """당신은 주어진 문맥(context)의 내용을 바탕으로만 질문에 답하는 AI 어시스턴트입니다.

**중요한 규칙:**
1. 질문과 정확히 일치하는 주제에 대한 정보만 답변하세요
2. 다른 주제의 정보를 질문한 주제에 적용하지 마세요  
3. 문맥에서 질문한 주제에 대한 구체적인 정보를 찾을 수 없다면 반드시 "해당 문서들에는 정보가 포함되어 있지 않습니다"라고 답변하세요
4. 여러 문서에서 관련 정보를 찾은 경우, 통합하여 답변하세요
5. 답변 시 해당 정보가 어느 문서에서 나온 것인지 명시하세요

모든 답변은 한국어로 대답해주세요.

CONTEXT: {context}

QUESTION: {question}

"""
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key,
            max_tokens=500,
            timeout=30
        )
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
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
    # 디버깅: 검색 결과 확인
    try:
        docs = retriever.get_relevant_documents(question)
        print(f"검색된 문서 개수: {len(docs)}")
        for i, doc in enumerate(docs):
            file_info = doc.metadata.get('file_name', 'Unknown')
            print(f"문서 {i+1} ({file_info}): {doc.page_content[:200]}...")
    except Exception as e:
        print(f"검색 디버깅 중 오류: {e}")
    
    return chain.invoke(question)
