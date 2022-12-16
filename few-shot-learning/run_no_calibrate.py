import argparse
from collections import defaultdict

from data_utils import load_dataset
from utils import *
import sqlparse
import sqlvalidator
import time
import random

def main(models, datasets, all_shots, num_seeds, subsample_test_set, api_num_log_prob, bs, use_saved_results):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'subsample_test_set': subsample_test_set,
        'api_num_log_prob': api_num_log_prob,
        'bs': bs
    }

    all_params = []
    for model in models:
        for dataset in datasets:
            for num_shots in all_shots:
                for seed in range(num_seeds):
                    p = deepcopy(default_params)
                    p['model'] = model
                    p['dataset'] = dataset
                    p['seed'] = seed
                    p['num_shots'] = num_shots
                    p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                    all_params.append(p)


    # query the model and save the responses
    if use_saved_results:
        load_results(all_params)
    else:
        save_results(all_params)


def save_results(params_list, freeze_test_set=True):
    """
    Save all model's responses and the rest of configs into a pickle file
    """
    result_tree = dict()
    
    for param_index, params in enumerate(params_list):
        print("\nExperiment name:", params['expr_name'])

        ### load data
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset(params)


        va_a_list = []
        ex_a_list = []
        em_a_list = []
        max_length = 100 if len(all_test_sentences) > 100 else len(all_test_sentences)
        r_list = random.sample(range(len(all_test_sentences)), 11)
        print(f'random list {r_list}')
        
        for idx in r_list:
            gt_labels = []
            result_sql = []
            for i in range(1):
                test_sentences = [all_test_sentences[idx]]
                test_labels = [all_test_labels[idx]]
                gt_labels.append(all_test_labels[idx])
                # ### sample test set
                # if params['subsample_test_set'] is None:
                #     test_sentences, test_labels = all_test_sentences, all_test_labels
                #     print(f"selecting full test set ({len(all_test_labels)} examples)")
                # else:
                #     if freeze_test_set:
                #         np.random.seed(0) # always use seed 0 result if freeze
                #     else:
                #         np.random.seed(params['seed'])
                #     test_sentences, test_labels = random_sampling(all_test_sentences, all_test_labels, params['subsample_test_set'])
                #     print(f"selecting {len(test_labels)} subsample of test set")

                ### sample few-shot training examples
                np.random.seed(params['seed'])
                # train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])
                train_sentences, train_labels = [], []

                ### Get model's original answers
                all_orig_ans, all_prompts_orig = get_model_response_text(params, all_train_sentences, all_train_labels, test_sentences,
                                                                return_all_prompts=True, num_tokens_to_predict_override=1024)
                # print(f'step1 answers: {all_orig_ans}')
                ### Get contextual-calibrated answer (first token)
                # ask model for candidate first token, for each of the test sentence

                ### Get accuracy
                all_orig_ans = [ans.strip() for ans in all_orig_ans]
                result_sql.append(all_orig_ans[0])

                # # add to result_tree
                # keys = [params['dataset'], params['model'], params['num_shots']]
                # node = result_tree # root
                # for k in keys:
                #     if not (k in node.keys()):
                #         node[k] = dict()
                #     node = node[k]
                # node[params['seed']] = accuracies


                # ### savings
                # result_to_save = dict()
                # params_to_save = deepcopy(params)
                # result_to_save['params'] = params_to_save
                # result_to_save['train_sentences'] = train_sentences
                # result_to_save['train_labels'] = train_labels
                # result_to_save['test_sentences'] = test_sentences
                # result_to_save['test_labels'] = test_labels
                # result_to_save['all_prompts_orig'] = all_prompts_orig
                # result_to_save['all_orig_ans'] = all_orig_ans
                # result_to_save['accuracies'] = accuracies
                # if 'prompt_func' in result_to_save['params'].keys():
                #     params_to_save['prompt_func'] = None
                # if 'execute_func' in result_to_save['params'].keys():
                #     params_to_save['execute_func'] = None
                # save_pickle(params, result_to_save)
                if not 't5' in params['model']: 
                    time.sleep(30)
            orig_accuracy = em_accuracy_helper(result_sql, gt_labels)
            accuracies = [orig_accuracy]
            em_a_list.append(orig_accuracy)
            print(f'index {idx}')
            print(f"accuracies {accuracies}")

            # org_va_accuracy = valid_sql_accuracy_helper(all_orig_ans, test_labels)
            # reweighted_va_accuracy = valid_sql_accuracy_helper(all_reweighted_ans, test_labels)
            # va_accuracies = [org_va_accuracy, reweighted_va_accuracy]
            # print(f"va_accuracies {va_accuracies}")

            org_ex_accuracy, org_va_accuracy = execution_accuracy_helper(params, result_sql, gt_labels)

            va_accuracies = [org_va_accuracy]
            va_a_list.append(org_va_accuracy)
            print(f"va_accuracies {va_accuracies}")

            ex_accuracies = [org_ex_accuracy]
            ex_a_list.append(org_ex_accuracy)
            print(f"ex_accuracies {ex_accuracies}")
        
        print(f'EM: {em_a_list} avg: {np.mean(em_a_list)}')
        print(f'EX: {ex_a_list} avg: {np.mean(ex_a_list)}')
        print(f'VA: {va_a_list} avg: {np.mean(va_a_list)}')

def em_accuracy_helper(prediction, label):
    correctness_list = []
    for pred, l in zip(prediction, label):
        pred = pred.split('\n')[0]
        if pred == l:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)

def valid_sql_accuracy_helper(prediction, label):
    correctness_list = []
    for sql in prediction:
        sql_query = sqlvalidator.parse(sql)
        if sql_query.validated:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)

def execution_accuracy_helper(params, prediction, label):
    correctness_list = []
    va_list = []
    for pred, l in zip(prediction, label):
        # print(f'prediction: {pred} gold: {l}')
        try:
            p_result = params['execute_func'](pred)
            gt_result = params['execute_func'](l)
            va_list.append(1)
            if p_result == gt_result:
                correctness_list.append(1)
            else:
                correctness_list.append(0)
        except Exception as e:
            va_list.append(0)
            correctness_list.append(0)
    return np.mean(correctness_list), np.mean(va_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True, default=False,
                        help='whether to load the results from pickle files and not run the model')

    args = parser.parse_args()
    args = vars(args)

    # simple processing
    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)

    main(**args)
