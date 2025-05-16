from trans.utils.io import load_jsonlines
from dataclasses import dataclass
from typing import Dict

@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    stop_word: str

    def build_prompt(self, system, query):
        system_format = self.system_format
        user_format = self.user_format

        system_text = system_format.format(content=system)
        user_message = user_format.format(content=query)
        prompt = system_text + user_message

        return prompt

template_dict: Dict[str, Template] = dict()

def register_template(template_name, system_format, user_format, assistant_format, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        stop_word=stop_word,
    )

register_template(
    template_name='llama3',
    system_format='<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|end_of_text|>\n',
    stop_word='<|end_of_text|>'
)