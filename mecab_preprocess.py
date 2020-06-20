import argparse
import os
import json
import math
import pickle as pkl
from tqdm import tqdm
from collections import namedtuple, Counter
from bert_small.tokenizer import BertTokenizer
from konlpy.tag import Mecab

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument("--save_path", type=str, default="data/")
parser.add_argument("--bert_path", type=str, default="bert_small/")

parser.add_argument("--split_train_data", action="store_true") # colab에서 RAM메모리가 부족한 경우 학습데이터 split 하기 위한 옵션

parser.add_argument("--max_question_length", type=int, default=80)
parser.add_argument("--max_context_length", type=int, default=1000)
parser.add_argument("--max_seq_length", type=int, default=384)
parser.add_argument("--context_stride", type=int, default=128)
args = parser.parse_args()

class Example(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 context_tokens,
                 context_spaces,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_tokens = context_tokens
        self.context_spaces = context_spaces
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

class Feature(object):
    def __init__(self,
                 unique_id,
                 example_index,
                 context_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 p_mask,
                 passage_idx,
                 ques_idx,
                 question_length,
                 start_position,
                 end_position,
                 term_freq):
        self.unique_id = unique_id
        self.example_index = example_index
        self.context_span_index = context_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.p_mask = p_mask
        self.passage_idx = passage_idx
        self.ques_idx = ques_idx
        self.question_length = question_length
        self.start_position = start_position
        self.end_position = end_position
        self.term_freq = term_freq

def get_morph_text(text, analyzer, check=False):
    text = text.strip()
    morph_text, morph_space = [], []
    morph_eojeol_tokens = []
    clean_eojeol_tokens = []

    orig_eojeol_tokens = []
    for token in text.split(" "):
        if len(token) > 0:
            orig_eojeol_tokens.append(token)
    idx = 0
    for i, [word, postag] in enumerate(analyzer.pos(text)):
        postag = postag.split("+")[0]
        morphtag_word = word + "/" + postag
        if len(word) == 0:
            continue
        morph_text.append(morphtag_word)

        if i == 0:
            morph_eojeol_tokens.append(morphtag_word)
            clean_eojeol_tokens.append(word)
            morph_space.append("B")
            continue

        if "".join(clean_eojeol_tokens[-1].split(" + ")) == orig_eojeol_tokens[idx]:
            morph_eojeol_tokens.append(morphtag_word)
            clean_eojeol_tokens.append(word)
            morph_space.append("B")
            idx += 1
        else:
            morph_eojeol_tokens[-1] += " + " + morphtag_word
            clean_eojeol_tokens[-1] += " + " + word
            morph_space.append("I")

    convert_text = []
    for token in clean_eojeol_tokens:
        convert_text.append("".join(token.split(" + ")))
    convert_text = " ".join(convert_text)
    if check and (convert_text != " ".join(orig_eojeol_tokens)):
        print(convert_text)
        print(text)
        print(morph_text)
        exit()
    return " ".join(morph_text), morph_eojeol_tokens, morph_space


def remove_postag(morph_tokens):
    tokens = []
    for token in morph_tokens:
        idx = token.rfind("/")
        if idx != -1:
            word = token[:idx]
        else:
            word = token
        tokens.append(word)
    return tokens

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def improve_morph_answer_span(tokens, spaces, input_start, input_end, orig_answer_text):
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = []
            for token, space in zip(tokens[new_start:(new_end+1)], spaces[new_start:(new_end+1)]):
                if len(text_span) == 0 or space == "B":
                    text_span.append(token)
                elif space == "I":
                    text_span[-1] += token
            text_span = " ".join(text_span)

            if text_span == orig_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def improve_answer_span(tokens, input_start, input_end, tokenizer, orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def check_is_max_context(spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, span) in enumerate(spans):
        end = span.start + span.length - 1
        if position < span.start:
            continue
        if position > end:
            continue
        num_left_context = position - span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def read_examples(file_name, data_type, morph_analyzer):

    global total_word_count
    examples = []
    total = 0.
    error = 0.
    skiped = 0.
    with open(file_name, "r") as fp:
        json_data = json.load(fp)["data"]

    for data in tqdm(json_data, total=len(json_data)):
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            morph_context, morph_eojeol_context_tokens, morph_context_space = get_morph_text(context, morph_analyzer, True)

            context_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in context:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        context_tokens.append(c)
                    else:
                        context_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(context_tokens) - 1)

            morph_context_tokens = []
            orig_to_tok_index = []
            tok_to_orig_index = []
            for (i, token) in enumerate(context_tokens):
                orig_to_tok_index.append(len(morph_context_tokens))
                sub_tokens = morph_eojeol_context_tokens[i].split(" + ")
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    morph_context_tokens.append(sub_token)

            clean_morph_context_tokens = remove_postag(morph_context_tokens)

            all_context_tokens = tokenizer.tokenize(" ".join(clean_morph_context_tokens))

            if data_type.lower() == "train" and args.max_context_length != -1 and len(all_context_tokens) > args.max_context_length:
                skiped += 1.
                continue

            for token in all_context_tokens:
                word_counter[token] += 1.
                total_word_count += 1.

            for qas in paragraph["qas"]:
                qas_id = qas["id"]
                question = qas["question"]

                for answer in qas["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer["answer_start"]
                    answer_length = len(answer_text)

                    start_position = char_to_word_offset[answer_start]
                    end_position = char_to_word_offset[answer_start + answer_length - 1]

                    morph_start_position = orig_to_tok_index[start_position]

                    if end_position < len(context_tokens) - 1:
                        morph_end_position = orig_to_tok_index[end_position + 1] - 1
                    else:
                        morph_end_position = len(morph_context_tokens) - 1

                    morph_start_position, morph_end_position = improve_morph_answer_span(clean_morph_context_tokens, morph_context_space, morph_start_position,
                                                                                         morph_end_position, answer_text)

                    if data_type.lower() == "train":
                        orig_text = " ".join(context_tokens[start_position:(end_position+1)])
                        morph_answer_tokens = clean_morph_context_tokens[morph_start_position:(morph_end_position+1)]
                        morph_answer_space = morph_context_space[morph_start_position:(morph_end_position+1)]

                        morph_text = []
                        for token, space in zip(morph_answer_tokens, morph_answer_space):
                            if len(morph_text) == 0 or space == "B":
                                morph_text.append(token)
                            elif space == "I":
                                morph_text[-1] += token
                        morph_text = " ".join(morph_text)

                        if orig_text.find(morph_text) == -1:
                            print("Could not find answer: '%s' vs. '%s' -> '%s'" % (orig_text, morph_text, answer_text))

                        total += 1.
                        if morph_text != answer_text:
                            error += 1.
                            continue

                    else :
                        morph_start_position = None
                        morph_end_position = None

                    morph_question, _, _ = get_morph_text(question, morph_analyzer)
                    morph_question = " ".join(remove_postag(morph_question.split(" ")))

                    question_tokens = tokenizer.tokenize(morph_question)
                    for token in question_tokens:
                        word_counter[token] += 1.
                        total_word_count += 1.

                    examples.append(Example(qas_id=qas_id,
                                            question_text=morph_question,
                                            context_tokens=clean_morph_context_tokens,
                                            context_spaces=morph_context_space,
                                            orig_answer_text=answer_text,
                                            start_position=morph_start_position,
                                            end_position=morph_end_position))

    if data_type.lower() == "train":
        print("[{}] Error: {} / {} ({:.2f}%)".format(data_type, error, total, error / total * 100.))
        print("[{}] Skiped: {} / {} ({:.2f}%)".format(data_type, skiped, total, skiped / total * 100.))

    with open(os.path.join(args.save_path, data_type + "_examples.pkl"), "wb") as fp:
        pkl.dump(examples, fp)

    return examples

def convert_examples_to_features(examples, data_type, tokenizer):
    unique_id = 1000000000
    orig_num_features = 0

    max_question_length = args.max_question_length
    max_seq_length = args.max_seq_length
    context_stride = args.context_stride
    features = []

    for (example_index, example) in enumerate(examples):
        question_tokens = tokenizer.tokenize(example.question_text)

        if len(question_tokens) > max_question_length:
            question_tokens = question_tokens[:max_question_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_context_tokens = []

        for (i, token) in enumerate(example.context_tokens):
            orig_to_tok_index.append(len(all_context_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_context_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        if data_type.lower() == "train":
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.context_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_context_tokens) - 1

            tok_start_position, tok_end_position = improve_answer_span(all_context_tokens, tok_start_position,
                                                                       tok_end_position, tokenizer, example.orig_answer_text)

        max_tokens_for_context = max_seq_length - len(question_tokens) - 3

        ContextSpan = namedtuple("ContextSpan", ["start", "length"])
        context_spans = []

        start_offset = 0
        while start_offset < len(all_context_tokens):
            length = len(all_context_tokens) - start_offset
            if length > max_tokens_for_context:
                length = max_tokens_for_context
            context_spans.append(ContextSpan(start=start_offset, length=length))
            if start_offset + length == len(all_context_tokens):
                break
            start_offset += min(length, context_stride)

        for (context_span_index, context_span) in enumerate(context_spans):
            tokens = []
            token_to_orig_map = dict()
            token_is_max_context = dict()
            segment_ids = []

            p_mask = []

            term_freq = []
            passage_idx = []
            ques_idx = []

            passage_idx.append(len(tokens))
            tokens.append("[CLS]")
            segment_ids.append(0)
            p_mask.append(0)
            term_freq.append(0.)

            for token in question_tokens:
                ques_idx.append(len(tokens))
                tokens.append(token)
                segment_ids.append(0)
                p_mask.append(1)
                term_freq.append(word_counter[token] / total_word_count)

            tokens.append("[SEP]")
            segment_ids.append(0)
            p_mask.append(1)
            term_freq.append(0.)

            for i in range(context_span.length):
                split_token_index = context_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = check_is_max_context(context_spans, context_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context

                passage_idx.append(len(tokens))
                tokens.append(all_context_tokens[split_token_index])
                segment_ids.append(1)
                p_mask.append(0)

                term_freq.append(word_counter[all_context_tokens[split_token_index]] / total_word_count)

            tokens.append("[SEP]")
            segment_ids.append(1)
            p_mask.append(1)
            term_freq.append(0.)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 for _ in range(len(input_ids))]

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                p_mask.append(1)
                term_freq.append(0.)

            while len(passage_idx) < max_seq_length:
                passage_idx.append(-1)
            while len(ques_idx) < max_question_length:
                ques_idx.append(-1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(term_freq) == max_seq_length
            assert len(passage_idx) == max_seq_length
            assert len(ques_idx) == max_question_length

            start_position = None
            end_position = None
            out_of_span = True

            if data_type.lower() == "train":
                context_start = context_span.start
                context_end = context_span.start + context_span.length - 1
                out_of_span = False
                if not (tok_start_position >= context_start and
                    tok_end_position <= context_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    context_offset = len(question_tokens) + 2
                    start_position = tok_start_position - context_start + context_offset
                    end_position = tok_end_position - context_start + context_offset

            if example_index < 20:
                print("unique_id: %s" % (unique_id))
                print("example_index: %s" % (example_index))
                print("tokens: %s" % " ".join(tokens))
                print("token_is_max_context: %s" % " ".join(["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                print("term_freq: %s" % " ".join([str(x) for x in term_freq]))

                if data_type.lower() == "train":
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    print("start_position: %d" % (start_position))
                    print("end_position: %d" % (end_position))
                    print("answer: %s, orig_answer: %s" % (answer_text, example.orig_answer_text))

                    if start_position > 0:
                        orig_start_position = token_to_orig_map[start_position]
                        orig_end_position = token_to_orig_map[end_position]
                        print("orig_mapping: %s" % example.context_tokens[orig_start_position:(orig_end_position+1)])

            features.append(
                Feature(unique_id=unique_id,
                        example_index=example_index,
                        context_span_index=context_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        p_mask=p_mask,
                        passage_idx=passage_idx,
                        ques_idx=ques_idx,
                        question_length=len(question_tokens),
                        start_position=start_position,
                        end_position=end_position,
                        term_freq=term_freq))
            unique_id += 1

        orig_num_features += 1

    num_features = len(features)

    if data_type.lower() == "train" and args.split_train_data:
        split = 60000
        offset = 0
        for i in range(1, math.ceil(num_features / split) + 1, 1):
            with open(os.path.join(args.save_path, data_type + "_features_{}.pkl".format(i)), "wb") as fp:
                pkl.dump(features[offset:offset+split], fp)
            offset += split
    else:
        with open(os.path.join(args.save_path, data_type + "_features.pkl"), "wb") as fp:
            pkl.dump(features, fp)
    print("[{}] {} -> {} Features".format(data_type, orig_num_features, num_features))

    return features

if __name__ == "__main__":

    morph_analyzer = Mecab()

    word_counter = Counter()
    total_word_count = 0.
    tokenizer = BertTokenizer(os.path.join(args.bert_path, "ko_vocab_32k.txt"), do_basic_tokenize=True)

    train_examples = read_examples(os.path.join(args.data_path, "KorQuAD_v1.0_train.json"), "train", morph_analyzer)
    dev_examples = read_examples(os.path.join(args.data_path, "KorQuAD_v1.0_dev.json"), "dev", morph_analyzer)

    with open(os.path.join(args.save_path, "word_counter.txt"), "w", encoding="utf-8") as fp:
        for word, cnt in word_counter.most_common():
            fp.write("{}\t{}\n".format(word, cnt))

    train_features = convert_examples_to_features(train_examples, "train", tokenizer)
    dev_features = convert_examples_to_features(dev_examples, "dev", tokenizer)