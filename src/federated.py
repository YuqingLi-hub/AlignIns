import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging
import argparse
import os
import warnings
from watermarks.modi_qim import QIM
from attacks import attack
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description="pass in a parameter")

    parser.add_argument(
        "--data", type=str, default="cifar10", help="dataset we want to train on"
    )
    parser.add_argument("--num_agents", type=int, default=20, help="number of agents:K")
    parser.add_argument(
        "--agent_frac", type=float, default=1.0, help="fraction of agents per round:C"
    )
    parser.add_argument(
        "--num_corrupt", type=int, default=0, help="number of corrupt agents"
    )
    parser.add_argument(
        "--rounds", type=int, default=150, help="number of communication rounds:R"
    )
    parser.add_argument(
        "--local_ep", type=int, default=2, help="number of local epochs:E"
    )
    parser.add_argument("--bs", type=int, default=64, help="local batch size: B")
    parser.add_argument(
        "--client_lr", type=float, default=0.1, help="clients learning rate"
    )
    parser.add_argument(
        "--server_lr", type=float, default=1, help="servers learning rate"
    )
    parser.add_argument(
        "--target_class", type=int, default=7, help="target class for backdoor attack"
    )
    parser.add_argument(
        "--poison_frac",
        type=float,
        default=0.5,
        help="fraction of dataset to corrupt for backdoor attack",
    )
    parser.add_argument(
        "--pattern_type", type=str, default="plus", help="shape of bd pattern"
    )
    parser.add_argument(
        "--theta", type=int, default=8, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--theta_ld", type=int, default=10, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--snap", type=int, default=1, help="do inference in every num of snap rounds"
    )
    parser.add_argument(
        "--device",
        default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="To use cuda, set to a specific GPU ID.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="num of workers for multithreading"
    )
    parser.add_argument(
        "--dense_ratio",
        type=float,
        default=0.25,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--anneal_factor",
        type=float,
        default=0.0001,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--se_threshold",
        type=float,
        default=1e-4,
        help="num of workers for multithreading",
    )
    parser.add_argument("--non_iid", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--attack",
        type=str,
        default="badnet",
        choices=["badnet", "DBA", "neurotoxin", "pgd"],
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default="avg",
        choices=[
            "avg",
            "alignins",
            "rlr",
            "mkrum",
            "mmetric",
            "lockdown",
            "foolsgold",
            "signguard",
            "rfa",
            "flgmm",
        ],
        help="aggregation function to aggregate agents' local weights",
    )
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--mask_init", type=str, default="ERK")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--same_mask", type=int, default=1)
    parser.add_argument("--cease_poison", type=float, default=100000)
    parser.add_argument("--exp_name_extra", type=str, help="defence name", default="")
    parser.add_argument("--super_power", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=1.0)
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=0)
    parser.add_argument("--job", type=str,default="ALTestDefault")
    parser.add_argument("--backdoor", action="store_true")
    parser.add_argument("--byz", action="store_true")
    parser.add_argument("--byz_attack", default="min_max", choices=["random", "sign_flip", "zero", "noise", "nan", "label_flip", "lie", "byzMean", "min_max", "min_sum", "adaptive_std", "adaptive_sign", "adaptive_uv", "non"], help="the attack method of byzantine agents")
    parser.add_argument("--num_byz", type=int, default=2, help="the number of byzantine agents")
    parser.add_argument("--use_g0", action="store_true")
    args = parser.parse_args()

    if args.clean:
        args.num_corrupt = 0
        args.byz = False
        args.num_byz = 0
        args.exp_name_extra = "clean"
    args.logging = logging
    args.logging = logging
    if args.super_power:
        args.exp_name_extra = "sp"
    if args.watermark:
        logging.info("Watermarking")
        args.rqim = QIM(args.delta)
    if args.watermark:
        logging.info("Watermarking")
        args.rqim = QIM(args.delta)
    per_data_dict = {
        "rounds": {"fmnist": 50, "cifar10": 100, "cifar100": 100, "tinyimagenet": 50},
        "num_target": {"fmnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,},
    }

    args.rounds = per_data_dict["rounds"][args.data]
    args.num_target = per_data_dict["num_target"][args.data]

    args.log_dir = utils.setup_logging(args)

    train_dataset, val_dataset = utils.get_datasets(args.data)
    backdoor_train_dataset = None
    Attack  = attack(args.byz_attack)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.non_iid:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = utils.distribute_data(
            train_dataset, args, n_classes=args.num_target
        )

    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()

    if args.data != "tinyimagenet":
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = utils.DatasetSplit(
            copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args
        )

    poisoned_val_loader = DataLoader(
        poisoned_val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.data != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(
            poisoned_val_set_only_x.dataset,
            args,
            idxs,
            poison_all=True,
            modify_label=False,
        )
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(
            copy.deepcopy(val_dataset),
            idxs,
            runtime_poison=True,
            args=args,
            modify_label=False,
        )

    poisoned_val_only_x_loader = DataLoader(
        poisoned_val_set_only_x,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # initialize a model, and the agents
    global_model = models.get_model(args.data, args).to(args.device)

    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(
        parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        )
    )
    params = {
        name: copy.deepcopy(global_model.state_dict()[name])
        for name in global_model.state_dict()
    }

    if args.aggr == "lockdown":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)

    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.aggr == "lockdown":
            if args.same_mask == 0:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=utils.init_masks(params, sparsity),
                    backdoor_train_dataset=backdoor_train_dataset,
                )
            else:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=mask,
                    backdoor_train_dataset=backdoor_train_dataset,
                )
        else:
            agent = Agent(
                _id,
                args,
                train_dataset,
                user_groups[_id],
                backdoor_train_dataset=backdoor_train_dataset,
            )
        if args.byz:
            agent.is_malicious = 1 if _id < args.num_corrupt or _id >= args.num_agents-args.num_byz else 0
        else:
            agent.is_malicious = 1 if _id < args.num_corrupt else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info(
            "build client:{} mal:{} data_num:{}".format(
                _id, agent.is_malicious, agent.n_data
            )
        )

    aggregator = Aggregation(agent_data_sizes, n_model_params, args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_dict = {}

    best_acc = -1
    pre_rnd_params = None
    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        rnd_global_params = parameters_to_vector(
            [
                copy.deepcopy(global_model.state_dict()[name])
                for name in global_model.state_dict()
            ]
        )
        mask = None
        if args.watermark:
            num_param = len(rnd_global_params)
            num_spars = 1000
            idx = torch.randint(0, (num_param - num_spars),size=(1,)).item()
            grads_unwater = copy.deepcopy(rnd_global_params)
            message = args.rqim.random_msg(num_spars)
            mask = [idx,idx+num_spars]
            print(f"Global model params: {rnd_global_params[mask[0]:mask[0]+5]}")
            grads_water = utils.embedding_watermark_on_position(masks=mask,whole_grads=grads_unwater,Watermark=args.rqim,message=message,alpha=args.alpha,k=args.k)
            utils.vector_to_model(copy.deepcopy(grads_water), global_model)
            if torch.allclose(parameters_to_vector(global_model.parameters()),grads_water):
                logging.info("Successfully update Watermarked parameter!")
            else:
                logging.warning("Update Failure")
            rnd_global_params = grads_water
        client_alpha = args.alpha
        client_k = args.k
        client_delta = args.delta
        mask = None
        if args.watermark:
            num_param = len(rnd_global_params)
            num_spars = 1000
            idx = torch.randint(0, (num_param - num_spars),size=(1,)).item()
            grads_unwater = copy.deepcopy(rnd_global_params)
            message = args.rqim.random_msg(num_spars)
            mask = [idx,idx+num_spars]
            print(f"Global model params: {rnd_global_params[mask[0]:mask[0]+5]}")
            grads_water = utils.embedding_watermark_on_position(masks=mask,whole_grads=grads_unwater,Watermark=args.rqim,message=message,alpha=args.alpha,k=args.k)
            utils.vector_to_model(copy.deepcopy(grads_water), global_model)
            if torch.allclose(parameters_to_vector(global_model.parameters()),grads_water):
                logging.info("Successfully update Watermarked parameter!")
            else:
                logging.warning("Update Failure")
            rnd_global_params = grads_water
        client_alpha = args.alpha
        client_k = args.k
        client_delta = args.delta
        agent_updates_dict = {}
        chosen = np.random.choice(
            args.num_agents,
            math.floor(args.num_agents * args.agent_frac),
            replace=False,
        )
        chosen = sorted(chosen)
        if args.aggr == "lockdown":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]
        byz_params = []
        benign_params = []
        for agent_id in chosen:
            if agents[agent_id].is_malicious and args.super_power:
                continue
            global_model = global_model.to(args.device)

            if args.aggr == "lockdown":
                update = agents[agent_id].local_train(
                    global_model,
                    criterion,
                    rnd,
                    global_mask=global_mask,
                    neurotoxin_mask=neurotoxin_mask,
                    updates_dict=updates_dict,
                )
            else:
                update = agents[agent_id].local_train(
                    global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask,masks=mask,delta=client_delta,alpha=client_alpha, k=client_k
                )
            
            
            # check = parameters_to_vector(
            #         [
            #             copy.deepcopy(global_model.state_dict()[name])
            #             for name in global_model.state_dict()
            #         ]
            #     )
            # if mask is not None:
            #     print(f"after agent {agent_id} model parameter:{check[mask[0]:mask[0]+5]}")
            #     print(f"agent {agent_id} update:{update[mask[0]:mask[0]+5]}")
            # else:
            #     print(f"after agent {agent_id} model parameter:{check[:10]}")
            #     print(f"agent {agent_id} update:{update[:10]}")
            
            agent_updates_dict[agent_id],m = utils.detect_recover_on_position(masks=mask,whole_grads=update,alpha=client_alpha,k=client_k,Watermark=args.rqim) if args.watermark else (update,None)
            if agents[agent_id].is_malicious and agent_id > args.num_corrupt:
                byz_params.append(agent_updates_dict[agent_id])
            else: benign_params.append(agent_updates_dict[agent_id])
            # if mask is not None:
            #     print(f"after recover model parameter:{agent_updates_dict[agent_id][mask[0]:mask[0]+5]}")
            # else:
            #     print(f"after recover model parameter:{agent_updates_dict[agent_id][:10]}")
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)
        if args.byz:
            byz_params = Attack(byz_params, benign_params)
            for agent_id in range(args.num_corrupt,args.num_agents-args.byz_num+1):
                if agents[agent_id].is_malicious:
                    agent_updates_dict[agent_id] = byz_params[agent_id]
                    logging.info(f"Byzantine agent {agent_id} attack with {args.byz_attack} method")
        
        # aggregate params obtained by agents and update the global params
        if args.aggr == "flgmm":
            updates_dict = aggregator.aggregate_updates(
            global_model, agent_updates_dict, epoch=rnd, g0=pre_rnd_params
            )
        else:
            updates_dict = aggregator.aggregate_updates(
                global_model, agent_updates_dict
            )
        check = parameters_to_vector(
            [
                copy.deepcopy(global_model.state_dict()[name])
                for name in global_model.state_dict()
            ]
        )
        pre_rnd_params = copy.deepcopy(check)
        # print(f"current model parameter:{check[:10]}")
        # # print(f"updates:{updates_dict[:10]}")
        # print(f"")
        # inference in every args.snap rounds
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            if args.aggr != "lockdown":
                val_acc = utils.get_loss_n_accuracy(
                    global_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    num_classes=args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
            else:
                test_model = copy.deepcopy(global_model)

                # CF
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device)
                    param.data = torch.where(
                        mask.to(args.device) >= args.theta_ld,
                        param,
                        torch.zeros_like(param),
                    )
                val_acc = utils.get_loss_n_accuracy(
                    test_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                del test_model

            logging.info("Clean ACC:              %.4f" % val_acc)
            logging.info("Attack Success Ratio:   %.4f" % asr)
            logging.info("Backdoor ACC:           %.4f" % poison_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_asr = asr
                best_bcdr_acc = poison_acc

        logging.info("------------------------------".format(rnd))

    logging.info("Best results:")
    logging.info("Clean ACC:              %.4f" % best_acc)
    logging.info("Attack Success Ratio:   %.4f" % best_asr)
    logging.info("Backdoor ACC:           %.4f" % best_bcdr_acc)
    logging.info("Training has finished!")
