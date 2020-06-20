import os
import argparse
import torch
import json
import math
import pickle as pkl
from model import Model
from utils.optimization import AdamW, get_linear_schedule_with_warmup
from utils.dataset import Dataset
from tqdm import tqdm
from collections import namedtuple, defaultdict, OrderedDict
from utils.evaluate import evaluate

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument("--save_path", type=str, default="data/")
parser.add_argument("--bert_path", type=str, default="bert_small/")
parser.add_argument("--model_path", type=str, default="models/")
parser.add_argument("--result_path", type=str, default="result/")
parser.add_argument("--model_name", type=str, default="tf+rnn_model")

parser.add_argument("--use_cuda", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--split_train_data", action="store_true")

parser.add_argument("--decay_period", type=int, default=2)
parser.add_argument("--seed", type=int, default=1023)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_best_size", type=int, default=20)
parser.add_argument("--max_answer_length", type=int, default=20)

parser.add_argument("--decay", type=float, default=0.5)
parser.add_argument("--warmup_proportion", type=float, default=0.06)
parser.add_argument("--lrate", type=float, default=0.002)
parser.add_argument("--bert_lrate", type=float, default=5e-5)

args = parser.parse_args()
torch.manual_seed(args.seed)

os.makedirs(args.model_path, exist_ok=True)
os.makedirs(args.result_path, exist_ok=True)

RawResult = namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

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

def get_best_indexes(logits, n_best_size):
    index_and_score = sorted(enumerate(logits), key=lambda x : x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def compute_softmax(scores):
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x
    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, output_prediction_file, output_nbest_file):

    example_index_to_features = defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = dict()
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    PrelimPrediction = namedtuple("PrelimPrediction", ["feature_index", "start_index", "end_index",
                                                       "start_logit", "end_logit"])

    all_predictions = OrderedDict()
    all_nbest_json = OrderedDict()

    for (example_index, example) in tqdm(enumerate(all_examples), total=len(all_examples), desc="write_predictions"):
        features = example_index_to_features[example_index]

        prelim_predictions = []

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = get_best_indexes(result.start_logits, n_best_size)
            end_indexes = get_best_indexes(result.end_logits, n_best_size)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    #if not feature.token_is_max_context.get(start_index, False):
                    #    continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(PrelimPrediction(feature_index=feature_index,
                                                               start_index=start_index,
                                                               end_index=end_index,
                                                               start_logit=result.start_logits[start_index],
                                                               end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(prelim_predictions, key=lambda x : (x.start_logit + x.end_logit), reverse=True)
        NbestPrediction = namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = dict()
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                orig_start_index = feature.token_to_orig_map[pred.start_index]
                orig_end_index = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.context_tokens[orig_start_index:(orig_end_index + 1)]
                token_spaces = example.context_spaces[orig_start_index:(orig_end_index + 1)]

                text = []
                for i, (token, space) in enumerate(zip(orig_tokens, token_spaces)):
                    if i == 0:
                        text.append(token)
                        continue
                    if space == "B":
                        text.append(token)
                    elif space == "I":
                        text[-1] += token

                final_text = " ".join(text)
            else:
                final_text = ""

            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True

            nbest.append(NbestPrediction(text=final_text,
                                         start_logit=pred.start_logit,
                                         end_logit=pred.end_logit))

        if not nbest:
            nbest.append(NbestPrediction(text="", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")
    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

def train():
    options = vars(args)
    for opt, val in options.items():
        print("[{}] = {}".format(opt, val))
    dataset = Dataset(options)

    device = "cuda" if args.use_cuda else "cpu"
    model = Model(options)
    model.to(device)
    dev_batches = dataset.get_batches(dataset.dev_data, args.batch_size, training=False)

    if args.eval:
        load_model_name = os.path.join(args.model_path, args.model_name)
        state_dict = torch.load(load_model_name, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        all_results = model.get_results(dev_batches, dataset.dev_features)
        model_name = args.model_name
        output_prediction_file = os.path.join(args.result_path, "{}_predictions.json".format(model_name))
        output_nbest_file = os.path.join(args.result_path, "{}_nbest_predictions.json".format(model_name))

        write_predictions(dataset.dev_examples, dataset.dev_features, all_results, args.n_best_size,
                          args.max_answer_length, output_prediction_file, output_nbest_file)
        with open(os.path.join(args.data_path, "KorQuAD_v1.0_dev.json"), "r") as fp:
            eval_dataset = json.load(fp)["data"]

        with open(output_prediction_file) as prediction_file:
            predictions = json.load(prediction_file)

        eval_info = evaluate(eval_dataset, predictions)
        print("[Dev] EM = {:.2f}%, F1 = {:.2f}%".format(eval_info["exact_match"], eval_info["f1"]))
        exit()

    if not args.split_train_data:
        train_batches = dataset.get_batches(dataset.train_data, args.batch_size)
        total_train_size = len(train_batches)
    else:
        total_train_size = 0
        for i in range(1, 3):
            with open(os.path.join(args.data_path, "train_features_{}.pkl".format(i)), "rb") as fp:
                train_features = pkl.load(fp)
                total_train_size += len(train_features)
                del train_features
        total_train_size = math.ceil(total_train_size / args.batch_size)

    bert_parameters = []
    parameters = []
    for param in model.named_parameters():
        if param[1].requires_grad:
            if "bert" in param[0]:
                bert_parameters.append(param)
            else:
                parameters.append(param[1])

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    bert_lrate = args.bert_lrate
    warmup_proportion = args.warmup_proportion
    num_train_steps = total_train_size * args.epochs
    bert_optimizer = AdamW(optimizer_grouped_parameters, lr=bert_lrate)

    lrate = args.lrate
    warmup_step = math.ceil(warmup_proportion * num_train_steps)

    optimizer = torch.optim.Adamax(parameters, lr=lrate, betas=(0.9, 0.999))
    bert_scheduler = get_linear_schedule_with_warmup(bert_optimizer, warmup_step, num_train_steps)

    best_f1 = 0.0
    dev_batches = dataset.get_batches(dataset.dev_data, args.batch_size, training=False)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.split_train_data:
            pbar = tqdm(train_batches, total=total_train_size)
            for i, batch_data in enumerate(pbar):
                loss = model(batch_data)
                loss.backward()

                optimizer.step()

                bert_optimizer.step()
                bert_scheduler.step()

                optimizer.zero_grad()
                bert_optimizer.zero_grad()
                model.zero_grad()

                if i % 100 == 0:
                    pbar.set_description("[Epoch {}] Step = {} / {}, Loss = {:.5f}".format(epoch, i, total_train_size, loss))

        else:
            for num_feat in range(1, 3):
                with open(os.path.join(args.data_path, "train_features_{}.pkl".format(num_feat)), "rb") as fp:
                    train_features = pkl.load(fp)
                    train_data = dataset.prepare_data(train_features)
                    train_batches = dataset.get_batches(train_data, args.batch_size)
                    del train_features, train_data

                train_size = len(train_batches)
                pbar = tqdm(train_batches, total=train_size)
                for i, batch_data in enumerate(pbar):
                    loss = model(batch_data)
                    loss.backward()

                    optimizer.step()

                    bert_optimizer.step()
                    bert_scheduler.step()

                    optimizer.zero_grad()
                    bert_optimizer.zero_grad()
                    model.zero_grad()
                    if i % 100 == 0:
                        pbar.set_description("[Epoch {}] Step = {} / {}, Loss = {:.5f}".format(epoch, i, train_size, loss))

        model.eval()
        all_results = model.get_results(dev_batches, dataset.dev_features)
        model_name = args.model_name
        output_prediction_file = os.path.join(args.result_path, "{}_predictions.json".format(model_name))
        output_nbest_file = os.path.join(args.result_path, "{}_nbest_predictions.json".format(model_name))

        write_predictions(dataset.dev_examples, dataset.dev_features, all_results, args.n_best_size,
                          args.max_answer_length, output_prediction_file, output_nbest_file)

        with open(os.path.join(args.data_path, "KorQuAD_v1.0_dev.json"), "r") as fp:
            eval_dataset = json.load(fp)["data"]

        with open(output_prediction_file) as prediction_file:
            predictions = json.load(prediction_file)

        eval_info = evaluate(eval_dataset, predictions)
        print("[Epoch {}] EM = {:.2f}%, F1 = {:.2f}%".format(epoch, eval_info["exact_match"], eval_info["f1"]))

        if best_f1 < eval_info["f1"]:
            best_f1 = eval_info["f1"]
            save_model_name = os.path.join(args.model_path, args.model_name)
            state_dict = model.state_dict()
            torch.save(state_dict, save_model_name)

        if epoch % args.decay_period == 0:
            lrate = lrate * args.decay
            for group in optimizer.param_groups:
                group['lr'] = lrate

if __name__ == "__main__":
    train()