from openai import OpenAI
import getpass, os

api_key=os.getenv("OPENAI_API_KEY")

QUERIES_MODEL_T = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
]
QUERIES_CR = [
    "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?",
    "What mistake did Elise Stefanik make in Congress on January 23, 2026?",
    "What is the purpose of the Main Street Parity Act?",
    "Who in Congress has spoken for and against funding of pregnancy centers?",
]

for question in QUERIES_CR:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    print("=" * 60)
    print("Question:")
    print(question + "\n")
    print(f"GPT-4o-mini Response: \n{response.choices[0].message.content}\n")
    # print(f"Cost: ${response.usage.total_tokens * 0.000000375:.6f}")