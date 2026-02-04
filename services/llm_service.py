from openai import OpenAI
import streamlit as st
from config.settings import OPENAI_API_KEY, CHAT_MODEL, ANALYSIS_MODEL

# 클라이언트 초기화
if not OPENAI_API_KEY:
    # st.secrets에서 시도 (Streamlit Cloud 배포용)
    if "OPENAI_API_KEY" in st.secrets:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    else:
        client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

import json


# RAG Service 초기화 (한 번만 로드 - 캐싱)
@st.cache_resource
def get_initialized_rag_service():
    try:
        from services.rag_service import RAGService

        return RAGService()
    except Exception as e:
        print(f"RAG Service Load Failed: {e}")
        return None


rag_service = get_initialized_rag_service()


def sanitize_user_input(text):
    """
    프롬프트 인젝션 공격을 방어하기 위해 사용자 입력을 필터링합니다.
    
    Args:
        text: 사용자 입력 텍스트
        
    Returns:
        tuple: (is_safe: bool, cleaned_text: str, warning: str)
    """
    if not text:
        return True, text, ""
    
    # 1. 특수 토큰 패턴 감지
    dangerous_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "[INST]",
        "[/INST]",
        "<<SYS>>",
        "<</SYS>>",
        "<s>",
        "</s>",
    ]
    
    # 2. 시스템 명령어 패턴 감지
    system_keywords = [
        "ignore previous",
        "ignore all previous",
        "disregard previous",
        "forget previous",
        "new instructions",
        "system prompt",
        "you are now",
        "pretend you are",
        "act as",
        "roleplay as",
        "너는 이제부터",
        "시스템 프롬프트",
        "이전 지시를 무시",
        "이전 명령을 무시",
        "위 지시를 무시",
        "프롬프트를 무시",
        "지시사항을 무시",
    ]
    
    # 3. JSON 인젝션 패턴 감지
    json_attack_keywords = [
        "\"request\":",
        "\"system\":",
        "\"instruction\":",
        "\"instructions\":",
        "\"response\":",
        "\"score\":",
        "\"reason\":",
        '"request":',
        '"system":',
        '"instruction":',
        '"response":',
    ]
    
    text_lower = text.lower()
    
    # 특수 토큰 감지
    for token in dangerous_tokens:
        if token.lower() in text_lower:
            return False, "", f"⚠️ 특수 토큰이 감지되었습니다: {token}"
    
    # 시스템 명령어 감지
    for keyword in system_keywords:
        if keyword in text_lower:
            return False, "", f"⚠️ 허용되지 않는 명령어가 감지되었습니다: {keyword}"
    
    # JSON 인젝션 패턴 감지
    for keyword in json_attack_keywords:
        if keyword.lower() in text_lower:
            return False, "", f"⚠️ JSON 인젝션 시도가 감지되었습니다"
    
    # JSON 구조 의심 패턴 (중괄호 과다 사용)
    import re
    brace_count = text.count('{') + text.count('}')
    if brace_count >= 4:  # { } 가 각각 2개 이상
        # JSON 파싱 시도
        try:
            import json
            parsed = json.loads(text)
            # 파싱 성공 + 의심스러운 키가 있으면 차단
            suspicious_keys = ['request', 'system', 'instruction', 'response', 'score', 'reason']
            if any(key in str(parsed).lower() for key in suspicious_keys):
                return False, "", "⚠️ JSON 구조 인젝션이 감지되었습니다"
        except:
            # JSON 파싱 실패는 괜찮음 (일반 중괄호 사용)
            pass
    
    # 4. 과도하게 긴 입력 차단 (일반적인 대화는 500자 이내)
    if len(text) > 1000:
        return False, "", "⚠️ 메시지가 너무 깁니다. (최대 1000자)"
    
    # 5. 연속된 특수문자 제거 (예: <<<, >>>)
    cleaned = re.sub(r'([<>|{}[\]])\1{2,}', r'\1', text)
    
    return True, cleaned, ""


def get_ai_response(messages):
    """
    OpenAI API를 통해 챗봇 응답을 받아옵니다.
    messages: game_view에서 관리하는 대화 내역 리스트 (System Prompt 포함)
    Returns: dict {"response": str, "score": int}
    """
    if not client:
        return {"response": "🚨 API Key가 설정되지 않았습니다.", "score": 0}

    # 프롬프트 인젝션 방어: 마지막 사용자 메시지 검증
    last_user_msg = ""
    last_user_index = -1
    for i, msg in enumerate(reversed(messages)):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            last_user_index = len(messages) - 1 - i
            break
    
    if last_user_msg:
        is_safe, cleaned_msg, warning = sanitize_user_input(last_user_msg)
        if not is_safe:
            # 위험한 입력 감지 시 안전한 응답 반환 (LLM 호출 안함)
            return {
                "response": "죄송하지만 기술적인 공격이네요. 안통한다 애송이!",
                "score": -100,
                "reason": "기술적인 공격"
            }
        
        # 입력이 정제되었다면 메시지 교체
        if cleaned_msg != last_user_msg:
            messages = list(messages)  # 복사
            messages[last_user_index] = {"role": "user", "content": cleaned_msg}

    # [RAG Integration]
    # 원본 messages를 변경하지 않기 위해 복사
    final_messages = list(messages)

    # 마지막 유저 메시지 추출
    last_user_msg = ""
    for msg in reversed(final_messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    # 검색 및 컨텍스트 주입
    if rag_service and last_user_msg:
        context = rag_service.search_context(last_user_msg)
        if context:
            # 시스템 메시지를 찾아서 컨텍스트 추가
            # 보통 messages[0]이 시스템 프롬프트임
            for i, msg in enumerate(final_messages):
                if msg["role"] == "system":
                    new_content = (
                        msg["content"]
                        + f"\n\n[참고 가능한 과거 대화 데이터]\n{context}\n\n위 데이터를 참고하되, 현재 대화 흐름에 맞게 자연스럽게 반응해."
                    )
                    # 해당 메시지만 교체 (딕셔너리 새로 생성)
                    final_messages[i] = {"role": "system", "content": new_content}
                    break

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=final_messages,
            response_format={"type": "json_object"},  # JSON 모드 강제
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"response": f"🚨 오류 발생: {str(e)}", "score": 0}


def analyze_conversation(history):
    """
    대화 기록을 분석하여 사용자의 연애 성향을 파악합니다.
    history: 각 라운드별 대화 기록 리스트 [{"round": 1, "persona": "EMOTIONAL", "messages": [...], "final_score": 70}, ...]
    Returns: dict (my_persona, ideal_preference, summary)
    """
    from config.prompts import get_analysis_prompt

    if not client:
        return {"error": "API Key가 설정되지 않았습니다."}

    # 대화 내용을 텍스트로 정리
    conversation_text = ""
    for entry in history:
        round_num = entry.get("round", "?")
        persona = entry.get("persona", "UNKNOWN")
        score = entry.get("final_score", "N/A")
        messages = entry.get("messages", [])

        conversation_text += (
            f"\n\n### 라운드 {round_num}: {persona} 타입 (최종 호감도: {score})\n"
        )
        for msg in messages:
            if msg["role"] == "user":
                conversation_text += f"[USER]: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation_text += f"[AI]: {msg['content']}\n"

    try:
        response = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": get_analysis_prompt()},
                {
                    "role": "user",
                    "content": f"다음 대화 기록을 분석해줘:\n{conversation_text}",
                },
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": f"분석 실패: {str(e)}"}
