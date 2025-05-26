import os
import copy
import json
import time
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from data_utils import Dataloader
from agent import DeepSeek, LocalLlama
from utils import (
    p_data, parse_json, extract_numbers,
    get_permutations, most_frequent_items, break_tie,
    split_text_into_token_chunks, truncate_text_by_tokens
)


class CustomFilter(logging.Filter):
    def filter(self, record):
        # Filter out noisy HTTP logs
        return "HTTP Request" not in record.getMessage()


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-agent model reasoning.")
    parser.add_argument('--model', type=str, default='llama', help="Agent model name (e.g., llama, deepseek)")
    parser.add_argument('--dataset', type=str, default='DetectiveQA', help="Dataset to use")
    parser.add_argument('--sample_num', type=int, default=1, help="Number of samples to process")
    parser.add_argument('--agent_num', type=int, default=5, help="Number of agents")
    parser.add_argument('--repetition_num', type=int, default=2, help="Repeat the full experiment N times")
    parser.add_argument('--api_key', type=str, default=os.getenv("API_KEY"), help="API key for model access")
    parser.add_argument('--base_url', type=str, default=os.getenv("BASE_URL"), help="Base URL for API endpoint")
    return parser.parse_args()


def build_agents(model_name, agent_num, api_key=None, base_url=None):
    agent_map = {
        'llama': LocalLlama,
        'deepseek': DeepSeek
    }
    agent_class = agent_map.get(model_name.lower())
    if not agent_class:
        raise ValueError(f"Unsupported model name: {model_name}")

    return [agent_class(id=i, model=model_name, api_key=api_key, base_url=base_url) for i in range(agent_num)]


def assign_docs(agent_list, data):
    chunks, total_tokens = split_text_into_token_chunks(data, agent_list)
    logging.info(f"Assigned for {len(agent_list)} agents, total tokens: {total_tokens}.")
    return chunks


def initial_topic_confirm(item, agent_list, doc_list):
    logging.info("-------- Initial Topic Confirm --------")
    for index, agent in enumerate(agent_list):
        agent.claim = item['question']
        agent.options = item['options']
        chunk = truncate_text_by_tokens(doc_list[index], 32768)
        messages = [
            {'role': 'system', 'content': p_data('round1')},
            {'role': 'user',
             'content': f"question:{agent.claim}\noptions:{item['options']}\ndocument chunk:{chunk}\n{p_data('round1')}"}
        ]
        for retry in range(6):
            try:
                response = agent.generate_response(messages)
                result = parse_json(response)
                agent.fact = result['evidence']
                agent.conclusion = result['answer']
                logging.warning(f"Agent {agent.id}, evidence: {result['evidence']}")
                logging.warning(f"Agent {agent.id}, answer: {result['answer']}")
                break
            except Exception as e:
                logging.warning(f"Agent {agent.id} response error, retry {retry}/6: {e}")
        agent.sequence.append(index)
        agent.opinions[tuple(agent.sequence)] = (agent.fact, agent.conclusion)
    logging.info("-------- Initial Topic Confirm Done --------")


def exchange_fact(agent_list, item):
    logging.info("-------- Exchange Fact --------")
    for agent in agent_list:
        others = [(a.id, a.fact, a.conclusion) for a in agent_list if a.id != agent.id]
        user_input = f"question:{agent.claim}\noptions:{item['options']}\n"
        user_input += f"Your evidence: {agent.fact}\nYour answer: {agent.conclusion}\n"
        for sid, fact, conclusion in others:
            user_input += f"\n{'#' * 20}\nAgent {sid}:\nevidence:{fact}\nanswer:{conclusion}\n{'#' * 20}\n"
        messages = [
            {'role': 'system',
             'content': p_data('round2', agent_list=[i for i in range(len(agent_list)) if i != agent.id])},
            {'role': 'user', 'content': user_input}
        ]
        for retry in range(6):
            try:
                response = agent.generate_response(messages)
                result = parse_json(response)
                agent.inspired = extract_numbers(result['id'])
                agent.explanation = result['explanation']
                break
            except Exception as e:
                logging.warning(f"Agent {agent.id} exchange error, retry {retry}/6: {e}")
    logging.info("-------- Exchange Fact Done --------")


def refine_topic(agent_list, doc_list):
    logging.info("-------- Refine Topic --------")
    for agent in agent_list:
        if not agent.inspired:
            continue
        all_sequences = get_permutations(agent.inspired)
        ori_sequence = copy.deepcopy(agent.sequence)
        for seq in all_sequences:
            agent.sequence = ori_sequence.copy()
            for index in seq:
                current_seq = tuple(agent.sequence + [index])
                if current_seq in agent.opinions or current_seq in agent.useless_sequence:
                    continue
                message = [
                    {'role': 'system', 'content': p_data('round3', agent_id=index)},
                    {'role': 'user', 'content': f"question:{agent.claim}\noption:{agent.options}\n"
                                                f"prev evidence:{agent.fact}\nprev answer:{agent.conclusion}\n"
                                                f"new document chunk:{truncate_text_by_tokens(doc_list[index], 32768)}"}
                ]
                for attempt in range(12):
                    try:
                        response = agent.generate_response(message)
                        result = parse_json(response)
                        agent.helpful[index] = result['utility']
                        if result['utility'] == 'useless':
                            agent.useless_sequence.append(current_seq)
                            break
                        agent.fact = result['fact']
                        agent.conclusion = result['conclusion']
                        agent.sequence.append(index)
                        agent.opinions[tuple(agent.sequence)] = (agent.fact, agent.conclusion)
                        break
                    except Exception as e:
                        logging.warning(f"Agent {agent.id} refine error, attempt {attempt}/12: {e}")
    logging.info("-------- Refine Topic Done --------")


def final_decision(agent_list, item):
    logging.info("-------- Final Decision --------")
    decisions = []
    content = f"Question: {item['question']}, Options: {item['options']}\n"
    for agent in agent_list:
        max_len = max(len(k) for k in agent.opinions if isinstance(k, tuple))
        opinions = [v for k, v in agent.opinions.items() if isinstance(k, tuple) and len(k) == max_len]
        points = ''.join(
            f"<opinion{i}>Chunks {k}:{v}</opinion{i}>" for i, (k, v) in enumerate(agent.opinions.items()) if
            len(k) == max_len
        )
        message = [
            {'role': 'system', 'content': p_data('final_round')},
            {'role': 'user', 'content': f"question:{agent.claim}\noptions:{agent.options}\nyour opinions:{points}"}
        ]
        for retry in range(6):
            try:
                response = agent.generate_response(message)
                result = parse_json(response)
                agent.final_decision = result['result']
                break
            except Exception as e:
                logging.warning(f"Final decision error (Agent {agent.id}), retry {retry}/6: {e}")
        decisions.append(agent.final_decision)
    filtered = [d for d in decisions if d != "None"]
    if not filtered:
        return "None"
    result = most_frequent_items(filtered)
    if len(result) > 1:
        result = break_tie(agent_list, result, content)
    else:
        result = result[0]
    logging.info(f"Majority Vote: {result}")
    return result


def main():
    args = parse_args()
    for run_idx in range(args.repetition_num):
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        result_dir = "../results"
        os.makedirs(result_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{args.model}_{args.dataset}_{datetime.now().strftime('%m-%d-%H-%M')}.log")
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().addFilter(CustomFilter())

        dataloader = Dataloader(args.dataset)
        data = dataloader.get_data(args.sample_num)
        agent_list = build_agents(args.model, args.agent_num, api_key=args.api_key, base_url=args.base_url)
        result_path = os.path.join(result_dir, f"{args.model}_{args.dataset}_{run_idx}.json")

        results, count = [], 0
        for sample in tqdm(data, desc="Processing Questions"):
            docs_list = assign_docs(agent_list, sample['context'])
            initial_topic_confirm(sample, agent_list, docs_list)
            exchange_fact(agent_list, sample)
            refine_topic(agent_list, docs_list)
            result = final_decision(agent_list, sample)
            print(f"[Q{count}] Final Decision: {result}")
            results.append({
                "question": count,
                "final_decision": result,
                "label": sample['answer']
            })
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=4)
            for agent in agent_list:
                agent.reset()
            count += 1


if __name__ == '__main__':
    main()
