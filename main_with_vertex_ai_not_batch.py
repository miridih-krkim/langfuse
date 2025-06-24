from langfuse import observe, get_client
# from langfuse.decorators import observe
# from langfuse.google import vertexai

PROJECT_ID = "mabq-247521"
LOCATION = "us-central1"
 
import vertexai
from google.oauth2 import service_account
import base64
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

# 서비스 계정 인증 정보 로드 및 vertexai 초기화 (한 번만)
credentials = service_account.Credentials.from_service_account_file(
    '/Users/miridih/Desktop/dp-ae-gcs-api_service_account_key.json')
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Langfuse 클라이언트 초기화 (한 번만)
langfuse = get_client()

# 기존 텍스트 생성 함수 (주석 처리)
"""
@observe(as_type="generation")
def vertex_generate_content(input, model_name = "gemini-2.0-flash-001"):
  # vertexai.init() 호출 제거 (위에서 이미 초기화됨)
  model = GenerativeModel(model_name)
  response = model.generate_content(
      [input],
      generation_config={
        "max_output_tokens": 1000,
        "temperature": 0.5,
        "top_p": 0.95,
      }
  )
 
  # pass model, model input, and usage metrics to Langfuse
  langfuse.update_current_generation(
      input=input,
      model=model_name,
      usage_details={
          "input": response.usage_metadata.prompt_token_count,
          "output": response.usage_metadata.candidates_token_count,
          "total": response.usage_metadata.total_token_count
      }
  )
  return response.candidates[0].content.parts[0].text

@observe()
def assemble_prompt():
  return "please generate a small poem addressing the size of the sun and its importance for humanity"
 
@observe()
def poem():
  langfuse.update_current_trace(user_id="krkim@miridih.com")
  prompt = assemble_prompt()
  return vertex_generate_content(prompt)
 
poem()
## install requirements for this notebook
# %pip install langchain langchain-google-vertexai langfuse anthropic[vertex] google-cloud-aiplatform Pillow
"""

# 새로운 멀티모달 기능 구현
import requests
from io import BytesIO
from PIL import Image

@observe(as_type="generation")
def vertex_multimodal_generate(text_prompt, image_url, model_name="gemini-2.0-flash-001"):
    """
    Vertex AI의 Gemini 모델을 사용하여 텍스트와 이미지를 입력으로 받아 응답을 생성합니다.
    """
    model = GenerativeModel(model_name)
    
    try:
        # URL에서 이미지 다운로드
        response = requests.get(image_url)
        response.raise_for_status()
        
        # 이미지 데이터를 바이트로 처리
        image_bytes = response.content
        
        # Part 객체 생성
        text_part = Part.from_text(text_prompt)
        image_part = Part.from_data(mime_type="image/webp", data=image_bytes)
        
        # 멀티모달 콘텐츠 생성 요청
        response = model.generate_content(
            [text_part, image_part],
            generation_config={
                "max_output_tokens": 1000,
                "temperature": 0.2,
                "top_p": 0.95,
            }
        )
        
        # OpenAI 형식과 유사한 입력 구조 생성 (Langfuse UI에서 이미지 표시를 위함)
        formatted_input = [
            {
                "role": "system",
                "content": "You are a helpful assistant who understands both images and text."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "auto"
                        }
                    }
                ]
            }
        ]
        
        # Langfuse에 사용 정보 기록 (OpenAI 형식으로 구조화)
        langfuse.update_current_generation(
            input=formatted_input,
            output=response.text,
            model=model_name,
            usage_details={
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count
            }
        )
        
        return response.text
        
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {str(e)}")
        return f"오류: {str(e)}"

@observe()
def multimodal_chat():
    """
    멀티모달 채팅 기능을 구현한 함수
    """
    langfuse.update_current_trace(user_id="krkim@miridih.com")
    
    # 이미지 URL과 텍스트 프롬프트 설정
    image_url = "https://file.miricanvas.com/template_thumb/2024/12/11/14/10/kl20y1d0si920djw/thumb.webp"
    text_prompt = "What is shown in this image?"
    
    # 멀티모달 생성 함수 호출
    response = vertex_multimodal_generate(text_prompt, image_url)
    
    # 응답 결과 출력 및 리턴
    print(response)
    return response

# 멀티모달 채팅 함수 실행
multimodal_chat()