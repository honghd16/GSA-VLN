import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.append('.')
sys.path.append('..')

import torch
from tensorboardX import SummaryWriter

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB
from r2r.data_utils import construct_instrs, get_scans
from r2r.env import R2RNavBatch
from r2r.parser import parse_args

from r2r.agent import GMapNavAgent

def build_dataset(args, rank=0, is_test=False):
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    dataset_class = R2RNavBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, 
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
        )
    else:
        aug_env = None

    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], 
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        is_test=is_test
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir,
        batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', 
    )

    # val_env_names = ['Validation_Residential_Basic', 'Validation_Non-residential_Basic', 'Validation_Non-residential_Scene']
    val_env_names = ['Test_Residential_Basic', 'Test_Non-residential_Basic', 'Test_Non-residential_Scene']
    
    env2scans = get_scans(args.anno_dir, val_env_names)
    val_env_names = []
    for env_name, scans in env2scans.items():
        for scan in scans:
            val_env_names.append(f"{env_name}:{scan}")

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=1,
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
        )   # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env, env2scans

def train(args, train_env, val_envs, aug_env=None, rank=-1, env2scans=None):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapNavAgent
    listner = agent_class(args, train_env, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )

    # first evaluation
    if args.eval_first:
        srs = defaultdict(list)
        loss_str = "validation before training"
        for env_name, env in tqdm(val_envs.items()):
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                srs[env_name.split(":")[0]].append(score_summary['sr'])
        if default_gpu:
            for k, v in srs.items():
                mean_sr = np.mean(v)
                loss_str += ", %s " % k
                loss_str += ', %s: %.2f' % ('sr', mean_sr)
            write_to_record_file(loss_str, record_file)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {k.split(":")[0]: {"spl": 0., "sr": 0., "state":""} for k in env2scans.keys()}
    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                listner.train(1, feedback=args.feedback)

                # Train with Augmented data
                listner.env = aug_env
                listner.train(1, feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss),
                record_file
            )

        # Run validation
        loss_str = "iter {}".format(iter)
        srs = defaultdict(list)		
        for env_name, env in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                srs[env_name.split(":")[0]].append(score_summary['sr'])

        for k, v in srs.items():
            mean_sr = np.mean(v)
            loss_str += ", %s " % k
            loss_str += ', %s: %.2f' % ('sr', mean_sr)
            writer.add_scalar('%s/%s' % ('sr', k), mean_sr, idx)
            # select model by spl
            if k in best_val:
                if mean_sr >= best_val[k]['sr']:
                    has_aug = True if aug_env is not None else False
                    init_way = 'memory' if 'full_graph' in args.bert_ckpt_file else 'original'
                    best_val[k]['spl'] = score_summary['spl']
                    best_val[k]['sr'] = score_summary['sr']
                    best_val[k]['state'] = 'Iter %d %s' % (iter, loss_str)
                    listner.save(idx, os.path.join(args.ckpt_dir, f"best_{k}_memory_train_{args.max_traj_num}_{has_aug}_{init_way}"))

        if default_gpu:
            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )

def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    srs = defaultdict(list)
    spls = defaultdict(list)
    tls = defaultdict(list)
    nes = defaultdict(list)
    ndtws = defaultdict(list)
    for env_name, env in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            score_summary, _ = env.eval_metrics(preds)
            srs[env_name.split(":")[0]].append(score_summary['sr'])
            spls[env_name.split(":")[0]].append(score_summary['spl'])
            tls[env_name.split(":")[0]].append(score_summary['lengths'])
            nes[env_name.split(":")[0]].append(score_summary['nav_error'])
            ndtws[env_name.split(":")[0]].append(score_summary['nDTW'])
            
    results = {}
    for k, v in srs.items():
        mean_sr = np.mean(v)
        mean_spl = np.mean(spls[k])
        mean_tl = np.mean(tls[k])
        mean_ne = np.mean(nes[k])
        mean_ndtw = np.mean(ndtws[k])
        results[k] = {"sr": mean_sr, "spl": mean_spl,  "tl": mean_tl,  "ne": mean_ne,  "ndtw": mean_ndtw}
    
    with open(os.path.join(args.log_dir, f'best_valid_{args.resume_file.split("/")[-1]}.json'), 'w') as f:
        json.dump(results, f, indent=4)

def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env, env2scans = build_dataset(args, rank=rank, is_test=args.test)

    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank, env2scans=env2scans)
    else:
        valid(args, train_env, val_envs, rank=rank)
            

if __name__ == '__main__':
    main()
