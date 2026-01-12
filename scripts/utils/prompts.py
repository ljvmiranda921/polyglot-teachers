SYSTEM_PROMPT = """You are an expert assistant for {lang_name}. You can fluently understand and generate text in {lang_name}.
Ensure that your responses are coherent, culturally appropriate, and demonstrate a deep understanding of the language nuances."""


GENERATE_TPL = """As a multilingual data generator, your task is to generate a new example (`prompt` and `response`) for a dataset demonstrating how AI agents can fulfill general instructions for {lang_name}.
To do this, you will want to generate two pieces of information:
1) An "prompt" specifying a task to be completed. The task should be very challenging yet solvable.
2) A "response" representing a valid completion of that task in natural language. If the "response" does not satisfy the "prompt", then you have failed at your job. Do not provide unnecessary details, beyond what is explicitly needed to satisfy the instruction you generated.

Add diversity to your generations by varying the types of tasks you create, the styles and tones of the responses, and the complexity of the language used. This will help ensure a rich and varied dataset.
For example, you might create tasks that involve answering knowledge-based questions, answering math questions, providing explanations, generating creative content, or performing translations.

Please provide a JSON dictionary response that includes the new `prompt` and its corresponding `response`. Use the `prompt` and `response` keys in the dictionary.
Do not generate any other text in your response (for example, do not start your message with any greetings, and never ask for clarification or apologize for struggling with the task).
Try you best to ensure that the input and response you generate are distinct from the provided examples while maintaining a diverse, detailed, precise, comprehensive, and high-quality response.
It is important to generate responses that are contextually relevant and culturally appropriate for {lang_name}.

Here are some examples to guide your generation. The best way to use these examples is to identify the patterns and structures they follow, rather than copying them directly:
{examples}

New Example:
"""


TRANSLATE_TPL = """As a multilingual data generator, your task is to translate the given prompt from English into {lang_name} and generate the appropriate response in the same language.
Important: you must return both the translated prompt (into {lang_name}) and the response. Ensure that both the translated prompt and the response are coherent, culturally appropriate, and demonstrate a deep understanding of the language nuances.

Do not generate any other text in your response (for example, do not start your message with any greetings, and never ask for clarification or apologize for struggling with the task).
Do not return the original English prompt. Remember, you must translate the prompt first and return it.
Here is the prompt you need to translate and respond to: {prompt}
"""


RESPOND_TPL = """As a multilingual data generator, you will be presented a user request or instruction in the {lang_name} language. Your task is to generate an appropriate response for the given request.
Ensure that your response is coherent, culturally appropriate, and demonstrates a deep understanding of the language nuances

Do not generate any other text in your response (for example, do not start your message with any greetings, and never ask for clarification or apologize for struggling with the task).
Here is the prompt you need to respond to: {prompt}
"""

# Based on M-Prometheus rubric prompt (slightly adapted for multilingual use)
M_RUBRIC_PROMPT = """###Task Description: 
An instruction (might include an Input inside it) in {language}, a response to evaluate, and a score rubric representing a evaluation criteria are given. 
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output should contain the score and feedback only.
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{{instruction}}

###Response to evaluate:
{{response}}

###Score Rubrics:
{{rubric}}

###Feedback:"""


def get_rubric_criteria(lang_name: str) -> dict:
    RUBRIC_CRITERIA = {
        "criteria": f"Is the model proficient in language {lang_name}, including its cultural nuance and grammatical usage, and responds in a helpful and harmless manner according to the instruction?",
        "score1_description": "The response contains severe grammatical errors, lacks cultural appropriateness, or is unhelpful/harmful. The language proficiency is very poor.",
        "score2_description": "The response has noticeable grammatical errors and limited cultural awareness. It partially addresses the instruction but with significant gaps in language proficiency or helpfulness.",
        "score3_description": "The response demonstrates adequate language proficiency with some minor grammatical errors. It shows reasonable cultural awareness and addresses the instruction in a helpful manner, though improvements are possible.",
        "score4_description": "The response exhibits strong language proficiency with minimal grammatical errors and good cultural nuance. It addresses the instruction in a helpful and harmless way with only minor room for improvement.",
        "score5_description": "The response demonstrates excellent language proficiency with proper grammar, appropriate cultural nuance, and idiomatic usage. It fully addresses the instruction in a helpful and harmless manner.",
    }
    return RUBRIC_CRITERIA
