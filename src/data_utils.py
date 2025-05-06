import re
import pickle
import os
from typing import Dict, List, Any


class Dataloader:
    def __init__(self, dataset: str):
        self.dataset = dataset

    def combine_text_before_position(self, text_dict: Dict[str, List[str]], position: int) -> str:
        """
        Combine sentences before a specific position in the 'origin_text' list.
        Removes tokens like [number] using regex.
        """
        if 'origin_text' not in text_dict:
            raise KeyError("Missing 'origin_text' in input dictionary.")

        text_list = text_dict['origin_text']
        if position < 0 or position > len(text_list):
            raise ValueError("Position out of range.")

        # Clean and combine text up to the given position
        clean_texts = [re.sub(r'\[\d+\]\s*', '', text) for text in text_list[:position]]
        return ' '.join(clean_texts)

    def load_pickle(self, filepath: str) -> Any:
        """
        Load a pickle file with error handling for file not found.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_data(self, size: int) -> List[Dict[str, Any]]:
        """
        Load and format data from the specified dataset.
        Supports 'DetectiveQA' and 'NovelQA'.
        """
        if self.dataset == 'DetectiveQA':
            human_anno = self.load_pickle('../datasets/DetectiveQA/human_anno.pkl')
            novel_data = self.load_pickle('../datasets/DetectiveQA/novel_data.pkl')
            data = []

            for key, value in human_anno.items():
                for item in value['questions']:
                    data.append({
                        'question': item['question'],
                        'options': item['options'],
                        'answer': item['answer'],
                        'context': self.combine_text_before_position(novel_data[key], item['answer_position'])
                    })

        elif self.dataset == 'NovelQA':
            ori_data = self.load_pickle('../datasets/NovelQA/data.pkl')
            data = [
                {
                    'question': item['question'],
                    'options': item['options'],
                    'answer': item['answer'],
                    'context': item['context']
                }
                for item in ori_data if item.get('complexity') == 'mh'
            ]
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        return data[:size] if size < len(data) else data
