import os
import json
import numpy as np

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:
            if split == "train":    # the official splits
                filepath = os.path.join(anno_dir, 'Train', '%s_%s_enc.json' % (dataset.upper(), split))

                with open(filepath) as f:
                    new_data = json.load(f)
            else:
                env_name = split.split(':')[0]
                prefix1 = env_name.split('_')[0]
                prefix2 = env_name.split('_')[1]
                prefix3 = env_name.split('_')[2]
                scan = split.split(':')[1]
                if prefix3 == 'User':
                    character = env_name.split('_')[3]
                    filepath = os.path.join(anno_dir, prefix1, prefix2, prefix3, f"{character}.json")   
                else:
                    filepath = os.path.join(anno_dir, prefix1, prefix2, prefix3)
                    file_name = os.listdir(filepath)[0]
                    filepath = os.path.join(filepath, file_name)
                with open(filepath) as f:
                    new_data = json.load(f)
                new_data = [item for item in new_data if item['scan'] == scan]
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)    
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data

def get_scans(anno_dir, splits):
    env2scan = {}
    for split in splits:
        prefix1 = split.split('_')[0]
        prefix2 = split.split('_')[1]
        prefix3 = split.split('_')[2]
        if prefix3 == 'User':
            character = split.split('_')[3]
            filepath = os.path.join(anno_dir, prefix1, prefix2, prefix3, f"{character}.json")   
        else:
            filepath = os.path.join(anno_dir, prefix1, prefix2, prefix3)
            file_name = os.listdir(filepath)[0]
            filepath = os.path.join(filepath, file_name)
        with open(filepath) as f:
            data = json.load(f)
        scans = list(set([item['scan'] for item in data]))
        env2scan[split] = scans
    return env2scan
