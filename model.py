import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from collections import namedtuple
from torch import nn
from bert_small.modeling import BertModel, Config
from utils.layers import Linear, Summ, MnemonicPointerNet

RawResult = namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

class Model(nn.Module):
    def __init__(self, options):
        super(Model, self).__init__()
        self.options = options
        self.device = "cuda" if options["use_cuda"] else "cpu"
        self.build_model()

    def build_model(self):
        options = self.options
        bert_path = options["bert_path"]
        bert_state_dict = torch.load(os.path.join(bert_path, "bert_small_ckpt.bin"), map_location=self.device)
        config = Config.from_json_file(os.path.join(bert_path, "bert_small.json"))
        print(config)
        self.bert = BertModel(config)
        self.bert.load_state_dict(bert_state_dict)

        bert_dim = config.hidden_size

        self.rnn = nn.LSTM(input_size=bert_dim + 1,
                           hidden_size=bert_dim // 2,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True,
                           dropout=0.0)

        self.summ_layer = Summ(input_size=bert_dim,
                               dropout=config.dropout_prob)
        self.pointer_net = MnemonicPointerNet(input_size=bert_dim,
                                              dropout=config.dropout_prob)

    def prepare_data(self, batch_data):
        if self.training:
            input_ids, input_mask, segment_ids, term_freq, p_mask, passage_idx, ques_idx, ques_length, start_position, end_position = batch_data
        else:
            input_ids, input_mask, segment_ids, term_freq, p_mask, passage_idx, ques_idx, ques_length, _ = batch_data
            start_position, end_position = None, None

        input_lengths = input_ids.ne(0).sum(1)
        max_length = torch.max(input_lengths, 0)[0]

        input_ids = input_ids[:, :max_length]
        input_mask = input_mask[:, :max_length]
        segment_ids = segment_ids[:, :max_length]
        term_freq = term_freq[:, :max_length]
        p_mask = p_mask[:, :max_length]

        ques_lengths = ques_idx.ne(-1).sum(1)
        max_ques_length = torch.max(ques_lengths, dim=0)[0]
        ques_idx = ques_idx[:, :max_ques_length]

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)
        term_freq = term_freq.to(self.device)
        p_mask = p_mask.to(self.device)
        ques_idx = ques_idx.to(self.device)

        if start_position is not None:
            start_position = start_position.to(self.device)
            end_position = end_position.to(self.device)

        dict_data = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "term_freq": term_freq,
            "p_mask": p_mask,
            "passage_idx": passage_idx,
            "ques_idx": ques_idx,
            "start_position": start_position,
            "end_position": end_position,
        }

        return dict_data

    def encoding(self, dict_data):
        input_ids = dict_data["input_ids"]
        segment_ids = dict_data["segment_ids"]
        input_mask = dict_data["input_mask"]
        term_freq = dict_data["term_freq"]
        p_mask = dict_data["p_mask"]
        ques_idx = dict_data["ques_idx"]

        hidden_states, _ = self.bert(input_ids, segment_ids, input_mask)

        rnn_inp = torch.cat([hidden_states, term_freq.unsqueeze(2)], dim=2)
        hidden_states, _ = self.rnn(rnn_inp)

        batch_size, seqlen, hidden_size = hidden_states.size()
        ques_length = ques_idx.size(1)

        exp_ques_idx = ques_idx.unsqueeze(2).expand(batch_size, ques_length, hidden_size) + 1
        padding = hidden_states.data.new(hidden_states.size(0), 1, hidden_states.size(2)).zero_()
        ques_states = torch.gather(torch.cat([padding, hidden_states], dim=1), 1, exp_ques_idx)
        ques_mask = ques_idx.eq(-1).float() * -1e10

        return hidden_states, p_mask, ques_states, ques_mask

    def decoding(self, hidden_states, p_mask, ques_states, ques_mask):
        mask = p_mask * -1e10

        ques_summ = self.summ_layer(ques_states, ques_mask)
        start_logits, end_logits = self.pointer_net(hidden_states, mask, ques_summ)

        return start_logits, end_logits

    def compute_loss(self, start_logits, end_logits, start_position, end_position):
        ignored_index = start_logits.size(1)
        start_position.clamp_(0, ignored_index)
        end_position.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_position)
        end_loss = loss_fct(end_logits, end_position)
        total_loss = (start_loss + end_loss) / 2.

        return total_loss

    def get_results(self, batches, features):
        pbar = tqdm(batches, total=len(batches), desc="get_results")
        all_results = []
        for batch_data in pbar:
            with torch.no_grad():
                start_logits, end_logits = self.forward(batch_data, predict=True)
                example_indices = batch_data[-1]

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits[i].detach().cpu().tolist(),
                                             end_logits=end_logits[i].detach().cpu().tolist()))

        return all_results

    def forward(self, batch_data, predict=False):
        dict_data = self.prepare_data(batch_data)
        hidden_states, p_mask, ques_states, ques_mask = self.encoding(dict_data)
        start_logits, end_logits = self.decoding(hidden_states, p_mask, ques_states, ques_mask)
        if predict:
            return start_logits, end_logits
        loss = self.compute_loss(start_logits, end_logits, dict_data["start_position"], dict_data["end_position"])

        return loss
