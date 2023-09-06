import os
import replicate
from IPython.display import display, Markdown

class LLAMA:
    def __init__(self, model, temperature, secret_key):
        os.environ['REPLICATE_API_TOKEN'] = secret_key
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = 4096
        self.top_p = 0.9
        self.repetition_penalty = 1
        self.string_dialogue = ""
        self.system_prompt = "You are an intelligent data science/analytics teaching assistant that guides the student in performing data science/analytics tasks. The student's prompt contains a context and an instruction. You are only to carry out the instruction using the context given.\n"


    def query_llama(self, prompt):
        self.string_dialogue += prompt + '\n'
        response = replicate.run(model_version=self.model, 
                                    input = {'prompt': self.string_dialogue,
                                             'system_prompt': self.system_prompt, 
                                            'temperature': self.temperature,
                                            'max_new_tokens': self.max_new_tokens,
                                            'top_p':self.top_p,
                                            'repetition_penalty':self.repetition_penalty})
        answer = ''
        for item in response:
            answer += item
        self.string_dialogue += answer + '\n'
        display(Markdown(answer))