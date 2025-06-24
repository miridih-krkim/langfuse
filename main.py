from langfuse import Langfuse, observe, get_client
from langfuse.openai import openai  # OpenAI integration

langfuse = Langfuse()

@observe()
def story():
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a great storyteller."},
            {"role": "user", "content": "Once upon a time in a galaxy far, far away..."}
        ],
        max_tokens=500,
    ).choices[0].message.content

@observe()
def main():
    langfuse.update_current_trace(user_id="krkim@miridih.com")  # ← 여기가 올바른 위치
    return story()

main()
