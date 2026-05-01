import dotenv
from openai import OpenAI

TEST_IMAGE_URL = "https://dummyimage.com/256x256/000/fff.png&text=test"
MAX_TOKENS = 1

credentials = dotenv.dotenv_values("auth.env")
client = OpenAI(api_key=credentials["OPENAI_API_KEY"])

def supports_vision(model_name: str) -> bool:
    """
    Probe whether a model supports image input.
    """
    try:
        client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": TEST_IMAGE_URL,
                                "detail": "low"
                            }
                        }
                    ],
                }
            ],
            max_tokens=MAX_TOKENS,
        )
        return True
    except Exception as e:
        msg = str(e).lower()
        if "image" in msg or "vision" in msg or "cannot handle" in msg:
            return False
        return False


def main():
    print("Fetching model list...\n")
    models = client.models.list().data

    vision_models = []
    text_models = []

    for m in models:
        model_id = m.id
        print(f"Testing model: {model_id}")
        ok = supports_vision(model_id)
        if ok:
            print("  ✅ Vision supported\n")
            vision_models.append(model_id)
        else:
            print("  ❌ Text-only or unsupported\n")
            text_models.append(model_id)

    print("\n================ FINAL RESULT ================\n")
    print("✅ Vision-capable models:")
    for m in vision_models:
        print("  ", m)

    print("\n❌ Text-only models:")
    for m in text_models:
        print("  ", m)


if __name__ == "__main__":
    main()


'''
================ FINAL RESULT ================

✅ Vision-capable models:
   gpt-4o
   gpt-4o-2024-05-13
   gpt-4o-mini-2024-07-18
   gpt-4o-mini
   gpt-4o-2024-08-06
   gpt-4o-2024-11-20
   gpt-4.1-2025-04-14
   gpt-4.1
   gpt-4.1-mini-2025-04-14
   gpt-4.1-mini
   gpt-4.1-nano-2025-04-14
   gpt-4.1-nano
   gpt-5-chat-latest
   gpt-5-search-api
   gpt-5-search-api-2025-10-14
'''