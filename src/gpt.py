import openai
from IPython.display import display, Markdown

class GPT:
    def __init__(self, model, temperature, secret_key, log=None):
        openai.api_key = secret_key
        self.model = model
        self.temperature = temperature
        self.completion = openai.ChatCompletion()
        self.log = log

    def query_gpt(self, prompt):
        if self.log is None:
            self.log = [{'role':'system', 'content':'You are an intelligent data science/analytics teaching assistant'}]
        self.log.append({'role':'user', 'content':prompt})
        response = self.completion.create(model=self.model, messages=self.log, temperature=self.temperature)
        answer = response.choices[0]['message']['content']
        self.log.append({'role': 'assistant', 'content': answer})
        display(Markdown(answer))