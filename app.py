# app.py

import streamlit as st
import os
import tempfile
from rag_logic import initialize_rag_chain, get_answer

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("PDF문서 기반 RAG Chatbot_JunH")

# OpenAI API 키 입력
st.sidebar.title("🔑 API 설정")
openai_api_key = st.sidebar.text_input(
    "OpenAI API 키를 입력하세요:",
    type="password",
    placeholder="sk-..."
)

if not openai_api_key:
    st.warning("⚠️ OpenAI API 키를 입력해주세요.")
    st.stop()

# 파일 업로드 기능 (다중 파일 지원)
st.sidebar.title("📄 문서 업로드")
uploaded_files = st.sidebar.file_uploader(
    "PDF 파일들을 업로드하세요:",
    type=['pdf'],
    accept_multiple_files=True,
    help="분석하고 싶은 여러 PDF 문서를 업로드하세요."
)

if not uploaded_files:
    st.warning("⚠️ 분석할 PDF 파일들을 업로드해주세요.")
    st.info("👈 사이드바에서 PDF 파일들을 업로드하면 해당 문서들을 기반으로 질문에 답변해드립니다.")
    st.stop()

# RAG 체인 초기화
file_hashes = [str(hash(file.getvalue())) for file in uploaded_files]
combined_hash = str(hash(tuple(file_hashes)))

if ("rag_chain" not in st.session_state or 
    st.session_state.get("api_key") != openai_api_key or 
    st.session_state.get("file_hash") != combined_hash):
    
    try:
        with st.spinner("문서들을 분석하고 RAG 시스템을 초기화 중..."):
            # 임시 파일들로 저장
            temp_file_paths = []
            file_names = []
            total_file_size = 0
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_paths.append(tmp_file.name)
                    file_names.append(uploaded_file.name)
                    
                    # 파일 크기 확인
                    file_size = os.path.getsize(tmp_file.name)
                    total_file_size += file_size
                    st.write(f"📄 {uploaded_file.name}: {file_size / (1024*1024):.2f} MB")
            
            st.write(f"📊 총 파일 크기: {total_file_size / (1024*1024):.2f} MB")
            
            # RAG 체인 초기화 (여러 파일 경로와 파일명 전달)
            rag_chain, retriever, api_key = initialize_rag_chain(openai_api_key, temp_file_paths, file_names)
            
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever
            st.session_state.api_key = openai_api_key
            st.session_state.openai_api_key = api_key  # API 키 저장
            st.session_state.file_hash = combined_hash
            st.session_state.file_names = file_names
            
            # 임시 파일들 삭제
            for temp_path in temp_file_paths:
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            st.success(f"✅ {len(file_names)}개 문서 분석 완료: {', '.join(file_names)}")
            
    except ValueError as ve:
        st.error(f"❌ 문서 처리 오류: {str(ve)}")
        st.info("💡 다음을 확인해주세요:")
        st.info("• PDF가 텍스트를 포함하고 있는지 확인")
        st.info("• 파일이 손상되지 않았는지 확인")
        st.info("• 파일 크기가 너무 크지 않은지 확인")
        st.stop()
        
    except Exception as e:
        st.error(f"❌ 초기화 오류: {str(e)}")
        st.info("💡 가능한 해결 방법:")
        st.info("• OpenAI API 키가 올바른지 확인")
        st.info("• 네트워크 연결 상태 확인")
        st.info("• 다른 PDF 파일로 시도")
        st.stop()

# 현재 분석 중인 문서들 표시
if "file_names" in st.session_state:
    st.info(f"📖 현재 분석 문서들: **{', '.join(st.session_state.file_names)}**")

# 채팅 기능
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("답변 생성 중..."):
                response = get_answer(
                    st.session_state.rag_chain,
                    st.session_state.retriever,
                    prompt,
                    st.session_state.openai_api_key  # API 키 전달
                )
                
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"❌ 답변 생성 오류: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

