import torch
import os
import pickle as pkl
from torch import LongTensor, FloatTensor

class Dataset(object):
    def __init__(self, options):
        self.options = options
        data_path = options["data_path"]
        save_path = options["save_path"]
        bert_path = options["bert_path"]
        split_train_data = options["split_train_data"]

        if not split_train_data:
            with open(os.path.join(save_path, "train_features.pkl"), "rb") as fp:
                self.train_features = pkl.load(fp)
            self.train_data = self.prepare_data(self.train_features)

        with open(os.path.join(save_path, "dev_examples.pkl"), "rb") as fp:
            self.dev_examples = pkl.load(fp)

        with open(os.path.join(save_path, "dev_features.pkl"), "rb") as fp:
            self.dev_features = pkl.load(fp)

        self.dev_data = self.prepare_data(self.dev_features, training=False)

    def prepare_data(self, features, training=True):
        if training:
            all_input_ids = LongTensor([f.input_ids for f in features])
            all_input_mask = LongTensor([f.input_mask for f in features])
            all_segment_ids = LongTensor([f.segment_ids for f in features])
            all_term_freq = FloatTensor([f.term_freq for f in features])
            all_p_mask = LongTensor([f.p_mask for f in features])
            all_passage_idx = LongTensor([f.passage_idx for f in features])
            all_ques_idx = LongTensor([f.ques_idx for f in features])
            all_ques_length = LongTensor([f.question_length for f in features])
            all_start_position = LongTensor([f.start_position for f in features])
            all_end_position = LongTensor([f.end_position for f in features])
            return (all_input_ids, all_input_mask, all_segment_ids, all_term_freq, all_p_mask, all_passage_idx, all_ques_idx, all_ques_length, all_start_position, all_end_position)
        else:
            all_input_ids = LongTensor([f.input_ids for f in features])
            all_input_mask = LongTensor([f.input_mask for f in features])
            all_segment_ids = LongTensor([f.segment_ids for f in features])
            all_term_freq = FloatTensor([f.term_freq for f in features])
            all_p_mask = LongTensor([f.p_mask for f in features])
            all_passage_idx = LongTensor([f.passage_idx for f in features])
            all_ques_idx = LongTensor([f.ques_idx for f in features])
            all_ques_length = LongTensor([f.question_length for f in features])
            all_example_indices = torch.arange(0, len(all_input_ids))
            return (all_input_ids, all_input_mask, all_segment_ids, all_term_freq, all_p_mask, all_passage_idx, all_ques_idx, all_ques_length, all_example_indices)

    def get_batches(self, dataset, batch_size, training=True):
        batches = []
        if training:
            all_input_ids, all_input_mask, all_segment_ids, all_term_freq, all_p_mask, all_passage_idx, all_ques_idx, all_ques_length, all_start_position, all_end_position = dataset
        else:
            all_input_ids, all_input_mask, all_segment_ids, all_term_freq, all_p_mask, all_passage_idx, all_ques_idx, all_ques_length, all_example_index = dataset
        for i in range(0, len(all_input_ids), batch_size):
            if training:
                batches.append((all_input_ids[i:i+batch_size],
                                all_input_mask[i:i+batch_size],
                                all_segment_ids[i:i + batch_size],
                                all_term_freq[i:i + batch_size],
                                all_p_mask[i:i + batch_size],
                                all_passage_idx[i:i + batch_size],
                                all_ques_idx[i:i + batch_size],
                                all_ques_length[i:i + batch_size],
                                all_start_position[i:i + batch_size],
                                all_end_position[i:i + batch_size]))
            else:
                batches.append((all_input_ids[i:i + batch_size],
                                all_input_mask[i:i + batch_size],
                                all_segment_ids[i:i + batch_size],
                                all_term_freq[i:i + batch_size],
                                all_p_mask[i:i + batch_size],
                                all_passage_idx[i:i + batch_size],
                                all_ques_idx[i:i + batch_size],
                                all_ques_length[i:i + batch_size],
                                all_example_index[i:i + batch_size]))

        return batches