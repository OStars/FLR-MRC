import sys
sys.path.insert(0, 'utils/')

import os
import shutil
from bertviz import head_view
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from model import FLRMRC
import torch
import torch.nn.functional as F
from transformers import AdamW, BertTokenizerFast
from torch.optim import lr_scheduler
from utils.lr_scheduler import get_linear_schedule_with_warmup
from utils.common import seed_everything, init_logger, logger, load_model
import json
import time
from src.data_loader import NerDataProcessor
from utils.finetuning_args import get_argparse, print_arguments
from utils.evaluate import MetricsCalculator4Ner
import multiprocessing
import pickle
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random_label_emb = None


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def compute_loss(gold, infer, padding_mask, loss_mask=None, is_logit=True, weight=-1):
    if loss_mask is not None:
        loss_mask = loss_mask.view(-1, 1, 1)
        infer = infer * loss_mask
        gold = gold * loss_mask
    label_num = infer.shape[-1]
    active_pos = padding_mask.contiguous().view(-1) == 1
    masked_infer = infer.contiguous().view(-1, label_num)[active_pos]
    masked_gold = gold.contiguous().view(-1, label_num)[active_pos]
    loss_ = F.binary_cross_entropy_with_logits(masked_infer, masked_gold, reduction='none') if is_logit else F.binary_cross_entropy(
        masked_infer, masked_gold, reduction='none')
    loss_ = torch.sum(loss_, 1)
    loss = torch.mean(loss_)

    return loss


def load_and_cache_examples(args, processor, input_file, data_type='train'):
    if os.path.exists(args.model_name_or_path):
        pretrain_model_name = str(args.model_name_or_path).split('/')[-1]
    else:
        pretrain_model_name = str(args.model_name_or_path)
    data_prefix = "".join(input_file.split("/")[-1].split(".")[:-1])
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(pretrain_model_name, args.data_name, data_prefix, 
            ('uncased' if args.do_lower_case else 'cased'), str(args.max_seq_length if data_type == 'train' else args.max_seq_length)))
    if args.data_tag != "":
        cached_features_file += "_{}".format(args.data_tag)
    if args.sliding_len != -1:
        cached_features_file += "_slided{}".format(args.sliding_len)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        with open(cached_features_file, 'rb') as fr:
            results = pickle.load(fr)
            # results = {'features': features}
            logger.info("total records: {}, {}".format(len(results['features']), results['stat_info']))
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        # id2label, label2id = processor.get_labels(args.second_label_file)
        results = processor.convert_examples_to_feature(input_file, data_type)
        logger.info("Saving features into cached file {}, total_records: {}, {}".format(
                    cached_features_file, len(results['features']), results['stat_info']))
        # torch.save(features, cached_features_file)
        with open(cached_features_file, 'wb') as fw:
            pickle.dump(results, fw)
    return results['features']


def evaluate(args, model, processor, input_file, output=False, output_eval_info=False, data_type='dev'):
    dev_dataset = load_and_cache_examples(args, processor, input_file, data_type=data_type)
    dev_dataloader = DataLoaderX(dataset=dev_dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=multiprocessing.cpu_count()//4,
                                 collate_fn=processor.generate_batch_data())

    metrics = MetricsCalculator4Ner(args, processor)
    batch_label_token_ids, batch_label_token_type_ids, batch_label_input_mask = processor.get_label_data(args.device)

    global random_label_emb
    if args.use_random_label_emb and random_label_emb is None:
        random_label_emb = torch.rand(size=batch_label_token_ids.shape + (1024,))
    model.eval()

    start_time = time.time()
    dev_bar = tqdm(dev_dataloader, desc="Evaluation")
    for data in dev_bar:
        # for step, data in enumerate(dev_dataloader):
        with torch.no_grad():
            if args.use_random_label_emb:
                data['random_label_emb'] = random_label_emb
            for key in data.keys():
                if key not in ['golden_label', 'ids', 'seq_len']:
                    data[key] = data[key].to(args.device)

            results = model(data, batch_label_token_ids,
                            batch_label_token_type_ids, batch_label_input_mask, return_score=args.dump_result, mode='inference', return_bert_attention=args.visualizate_bert)
            infer_starts, infer_ends = torch.sigmoid(results[0]), torch.sigmoid(results[1])
            if args.visualizate_bert:
                head_view(results[-1], processor.get_tokenizer().decode(data['token_ids'][0]).split(' '))

            metrics.update(infer_starts.cpu(), infer_ends.cpu(), data['golden_label'], data['seq_len'].cpu(),
                           match_label_ids=None, is_logits=False, tokens=(data['token_ids'] if output else None))

    end_time = time.time()
    logger.info("Evaluation costs {} seconds.".format(end_time - start_time))

    if output:
        # check the result dir
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        result_list = metrics.get_results()
        path = os.path.join(args.result_dir, "{}_result_ner.json".format(data_type))
        with open(path, 'w', encoding='utf-8') as fw:
            for line in result_list:
                fw.write(json.dumps(line, indent=4, ensure_ascii=False) + '\n')

    result_dict = metrics.get_metrics()['general']
    if output_eval_info:
        data_prefix = input_file.split('/')[-1].split('.')[0]
        logger.info("***** Eval results: {} *****".format(data_prefix))
        logger.info(
            "f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".format(result_dict["f1"], result_dict["precision"],
                                                                       result_dict["correct_num"], result_dict["infer_num"], result_dict[
                "recall"], result_dict["correct_num"], result_dict["golden_num"]))

    return result_dict

def train(args, model, tokenizer, processor):

    if args.do_ema:
        ema = ModelEma(model, 0.9997)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples(args, data_type="train", processor=processor, input_file=args.train_set)
    train_dataloader = DataLoaderX(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   drop_last=args.drop_last,
                                   num_workers=4,
                                   collate_fn=processor.generate_batch_data())

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.val_step = max(1, len(train_dataloader) // args.eval_per_epoch)
    if args.val_skip_epoch > 0:
        args.val_skip_step = max(
            1, len(train_dataloader)) * args.val_skip_epoch

    # define the optimizer
    bert_parameters = model.bert.named_parameters()
    first_start_params = model.entity_start_classifier.named_parameters()
    first_end_params = model.entity_end_classifier.named_parameters()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_parameters if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate},

        {"params": [p for n, p in first_start_params if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
        {"params": [p for n, p in first_start_params if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr},
        {"params": [p for n, p in first_end_params if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
        {"params": [p for n, p in first_end_params if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr},
    ]

    label_fused_params = model.label_fusing_layer.named_parameters()
    optimizer_grouped_parameters += [
        {"params": [p for n, p in label_fused_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
        {"params": [p for n, p in label_fused_params if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr},
    ]

    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    if args.do_ema:
        steps_per_epoch = len(train_dataloader)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=steps_per_epoch,
                                            epochs=int(args.num_train_epochs), pct_start=0.2)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)


    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # check the output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logger.info('\n')
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    # logger.info("  Total warmup steps = %d", warmup_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  lr of encoder = {}, lr of task_layer = {}".format(
        args.learning_rate, args.learning_rate * args.task_layer_lr))
    logger.info('\n')

    model.zero_grad()
    seed_everything(args.seed)

    batch_label_token_ids, batch_label_token_type_ids, batch_label_input_mask = processor.get_label_data(args.device)
    global random_label_emb
    if args.use_random_label_emb and random_label_emb is None:
        random_label_emb = torch.rand(size=batch_label_token_ids.shape + (1024,))

    if args.do_train:
        model.train()
        global_step = 0

        best_result = {'f1': 0.0}
        init_time = time.time()

        for epoch in range(int(args.num_train_epochs)):
            train_bar = tqdm(train_dataloader, ncols=100, desc="Training")
            for cur_step, data in enumerate(train_bar):
                global_step += 1

                if args.use_random_label_emb:
                    data['random_label_emb'] = random_label_emb
                for key in data.keys():
                    if key not in ['golden_label', 'ids', 'seq_len']:
                        data[key] = data[key].to(args.device)

                results = model(data, batch_label_token_ids, batch_label_token_type_ids, batch_label_input_mask)

                start_logits, end_logits = results[:2]
                start_loss = compute_loss(data['first_starts'], start_logits, data['input_mask'])
                end_loss = compute_loss(data['first_ends'], end_logits, data['input_mask'])
                total_loss = (start_loss + end_loss)
                writer.add_scalars(
                    'loss/train', {
                        'total_loss': total_loss.item(),
                        'start_loss': start_loss.item(),
                        'end_loss': end_loss.item()
                    },
                    global_step,
                )

                if args.n_gpu > 1:
                    total_loss = total_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / args.gradient_accumulation_steps

                total_loss.backward()
                train_bar.set_description(
                    "Training {}/{} step:{}, loss:{:.6}".format(epoch + 1, int(args.num_train_epochs), global_step, total_loss.item()))

                # add_label_info = False

                if global_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    if args.do_ema:
                        ema.update(model)
                    # add_label_info = True
                    # print(global_step)

                if args.do_eval and global_step > args.val_skip_step and global_step % args.val_step == 0:
                    model.eval()
                    test_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                    # call the test function

                    eval_result = evaluate(args, test_model, processor=processor, input_file=args.dev_set, data_type='dev', output_eval_info=True)
                    writer.add_scalar(
                        "f1/dev", eval_result["f1"], global_step)
                    logger.info("[dev], f1: {}\n".format(eval_result['f1']))
                    if eval_result["f1"] > best_result["f1"]:
                        best_result.update(eval_result)
                        best_result["step"] = global_step
                        best_result['epoch'] = epoch + 1
                        # save the best model
                        output_dir = args.output_dir
                        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.bin"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

                    if eval_result["f1"] > 0:
                        logger.info(
                            "best model: epoch {}, step {}, -- f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".format(
                                best_result["epoch"], best_result["step"],
                                best_result["f1"], best_result["precision"],
                                best_result["correct_num"], best_result["infer_num"],
                                best_result["recall"], best_result["correct_num"], best_result["golden_num"]))
                    if args.do_ema:
                        ema_results = evaluate(args, ema.module, processor=processor, input_file=args.dev_set,
                                               data_type='dev', output_eval_info=False)
                        logger.info("ema result [dev]: f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                                    format(ema_results["f1"], ema_results["precision"], ema_results["correct_num"], ema_results["infer_num"], ema_results["recall"], ema_results["correct_num"], ema_results["golden_num"], time.time() - init_time))
                        writer.add_scalar("ema_f1/dev", ema_results["f1"], global_step)
                    if args.eval_test:
                        eval_result = evaluate(args, test_model, processor=processor, input_file=args.test_set, data_type='test')
                        logger.info("[test], f1: {}\n".format(eval_result['f1']))
                        writer.add_scalar("f1/test", eval_result["f1"], global_step)

                        if args.do_ema:
                            ema_results = evaluate(args, ema.module, processor=processor, input_file=args.test_set,
                                                   data_type='test', output_eval_info=False)
                            logger.info("ema result [test]: f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                                        format(ema_results["f1"], ema_results["precision"], ema_results["correct_num"], ema_results["infer_num"], ema_results["recall"], ema_results["correct_num"], ema_results["golden_num"], time.time() - init_time))
                            writer.add_scalar("ema_f1/test", ema_results["f1"], global_step)
                    model.train()

        logger.info("** finish training **\n")
        logger.info("best model: step {}, -- f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                    format(best_result["step"], best_result["f1"], best_result["precision"], best_result["correct_num"], best_result["infer_num"], best_result["recall"], best_result["correct_num"], best_result["golden_num"], time.time() - init_time))
        if args.do_ema:
            ema_results = evaluate(args, ema.module, processor=processor, input_file=args.dev_set, data_type='dev', output_eval_info=True)
            logger.info("ema result [dev]: f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                        format(ema_results["f1"], ema_results["precision"], ema_results["correct_num"], ema_results["infer_num"], ema_results["recall"], ema_results["correct_num"], ema_results["golden_num"], time.time() - init_time))
            if ema_results["f1"] > best_result["f1"]:
                # save the best model
                output_dir = args.output_dir
                model_to_save = ema.module
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.bin"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

            ema_results = evaluate(args, ema.module, processor=processor, input_file=args.test_set, data_type='dev', output_eval_info=True)
            logger.info("ema result [test]: f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                        format(ema_results["f1"], ema_results["precision"], ema_results["correct_num"], ema_results["infer_num"], ema_results["recall"], ema_results["correct_num"], ema_results["golden_num"], time.time() - init_time))


def main(args):
    print("-"*20,"start","-"*20)
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and args.overwrite_output_dir:
        shutil.rmtree(os.path.join(args.output_dir, "tensorboard"))
        os.mkdir(os.path.join(args.output_dir, 'tensorboard'))
        # shutil.rmtree(os.path.join(args.output_dir, "logs"))
    if args.model_name_or_path.endswith('-uncased') and (not args.do_lower_case):
        raise ValueError(
            "use uncased model, 'do_lower_case' must be True")
    if args.model_name_or_path.endswith('-cased') and args.do_lower_case:
        raise ValueError(
            "use cased model, 'do_lower_case' must be False")

    if args.do_train and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.mkdir(os.path.join(args.output_dir, 'logs'))
        os.mkdir(os.path.join(args.output_dir, 'tensorboard'))
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file="{}/logs/{}-{}.log".format(args.output_dir, ('train' if args.do_train else 'eval'), time_))

    args.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    processor = NerDataProcessor(args, tokenizer)
    args.first_label_num = processor.get_class_num()
    # Set seed
    seed_everything(args.seed)

    logger.info("Training/evaluation parameters %s", args)
    print_arguments(args, logger)

    # init_model
    model = FLRMRC(args)
    model.to(args.device)

    if args.do_train:
        train(args, model, tokenizer, processor)
    if args.do_eval:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        load_model(model, args.output_dir, 'model.bin')

        if args.dev_set is not None:
            if os.path.isdir(args.dev_set):
                for dev_file in os.listdir(args.dev_set):
                    evaluate(args, model, processor, output=args.dump_result, output_eval_info=True,
                             data_type='dev', input_file=os.path.join(args.dev_set, dev_file))
            else:
                evaluate(args, model, processor, output=args.dump_result, output_eval_info=True, data_type='dev', input_file=args.dev_set)
        if args.test_set is not None:
            if os.path.isdir(args.test_set):
                for test_file in os.listdir(args.test_set):
                    evaluate(args, model, processor,
                             output=args.dump_result, output_eval_info=True, data_type='test', input_file=os.path.join(args.test_set, test_file))
            else:
                evaluate(args, model, processor, output=args.dump_result, output_eval_info=True, data_type='test', input_file=args.test_set)


if __name__ == "__main__":
    args = get_argparse().parse_args()
    main(args)