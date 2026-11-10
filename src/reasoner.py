
import os
import ollama


MODEL_NAME = "llama3.2:3b"


class ReasoningModel:

    def __init__(self):
        self.model_name = MODEL_NAME

        # Local Ollama by default.
        # For remote hosting, set OLLAMA_HOST.
        host = os.getenv("OLLAMA_HOST")

        if host:
            self.client = ollama.Client(host=host)
        else:
            self.client = ollama.Client()

    def answer(self, question):

        if not question.strip():
            raise ValueError("Please enter a mathematical question.")

        prompt = f"""
You are a mathematical reasoning assistant.

Solve the following mathematical word problem carefully.

Show the calculation briefly.

At the end, write the final answer in this format:

Final Answer: [answer]

Question:
{question}
"""

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        return response["message"]["content"]


if __name__ == "__main__":

    reasoner = ReasoningModel()

    question = (
        "Steve is 5 feet 6 inches tall. "
        "He grows 6 inches. "
        "How tall is he in inches?"
    )

    result = reasoner.answer(question)

    print("\nQUESTION:")
    print(question)

    print("\nMODEL ANSWER:")
    print(result)
