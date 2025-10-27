SYSTEM_PROMPT = """You are an expert assistant for {lang_name}. You can fluently understand and generate text in {lang_name}.
Ensure that your responses are coherent, culturally appropriate, and demonstrate a deep understanding of the language nuances."""


GENERATE_TPL = """As a multilingual data generator, your task is to generate a new example (`prompt` and `response`) for a dataset demonstrating how AI agents can fulfill general instructions for {lang_name}. 
To do this, you will want to generate two pieces of information:
1) An "prompt" specifying a task to be completed. The task should be very challenging yet solvable.
2) A "response" representing a valid completion of that task in natural language. If the "response" does not satisfy the "prompt", then you have failed at your job. Do not provide unnecessary details, beyond what is explicitly needed to satisfy the instruction you generated.

Do not generate any other text in your response (for example, do not start your message with any greetings, and never ask for clarification or apologize for struggling with the task). 
Try you best to ensure that the input and response you generate are distinct from the provided examples while maintaining a diverse, detailed, precise, comprehensive, and high-quality response. 
It is important to generate responses that are contextually relevant and culturally appropriate for {lang_name}.

Here are some examples to guide your generation:
{examples}

New Example:
"""


TRANSLATE_TPL = """As a multilingual data generator, your task is to translate the given prompt into {lang_name} and generate a response in the same language.
Ensure that both the translated prompt and the response are coherent, culturally appropriate, and demonstrate a deep understanding of the language nuances.

Do not generate any other text in your response (for example, do not start your message with any greetings, and never ask for clarification or apologize for struggling with the task). 
Here is the prompt you need to translate and respond to: {prompt}
"""


RESPOND_TPL = """As a multilingual data generator, your task is to generate a response in {lang_name} for the given prompt. 
Ensure that your response is coherent, culturally appropriate, and demonstrates a deep understanding of the language nuances

Do not generate any other text in your response (for example, do not start your message with any greetings, and never ask for clarification or apologize for struggling with the task). 
Here is the prompt you need to respond to: {prompt}
"""
