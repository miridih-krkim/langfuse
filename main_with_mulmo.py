from langfuse import Langfuse, observe, get_client
# from langfuse.decorators import observe
from langfuse.openai import openai

# Langfuse 인스턴스 초기화 (환경 변수로 설정해도 무방)
# langfuse = Langfuse(
#     public_key="your_public_key",
#     secret_key="your_secret_key",
#     host="https://cloud.langfuse.com"  # 또는 self-hosted URL
# )


langfuse = Langfuse()

@observe()
def multimodal_chat():
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # 멀티모달 입력 지원 모델
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who understands both images and text."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is shown in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://file.miricanvas.com/template_thumb/2024/12/11/14/10/kl20y1d0si920djw/thumb.webp",
                            "detail": "auto"
                        }
                    }
                ]
            }
        ],
        max_tokens=300,
    )

    # 응답 결과 출력 및 리턴
    print(response.choices[0].message.content)
    return response.choices[0].message.content


@observe()
def main():
    langfuse.update_current_trace(user_id="krkim@miridih.com")  # ← 여기가 올바른 위치
    return multimodal_chat()

main()
