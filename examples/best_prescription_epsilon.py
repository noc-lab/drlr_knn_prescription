import pickle
import json
import argparse
import numpy as np
from drlr_knn_prescription.util import build_validation_set_prescription, get_impute_outcome, return_prediction_and_std, \
    get_boltzman_policy, eval_prescription_probability, find_prescription_threshold
from drlr_knn_prescription.load_table import load_diabetes_final_table_for_prescription, \
    load_hypertension_final_table_for_prescription

parser = argparse.ArgumentParser()

parser.add_argument("--trial", type=int, default=0)
parser.add_argument("--test_ratio", type=float, default=0.2)
parser.add_argument("--save_dir", type=str, default='../ckpt/hypertension/')
parser.add_argument("--diabetes", type=bool, default=False)

args = parser.parse_args()


def find_best_random_epsilon(use_previous, T, y_pred, imputed_outcome):
    """
    find the best soften factor for the random policy
    :param use_previous: the outcome for keeping using the previous policy
    :param T: probablity to use randomized prescription
    :param y_pred: outcome prediction for different type of prescription
    :param imputed_outcome: imputed outcome using test data for different type of prescription
    :return: best epsilon founded
    """
    best_epsilon = None
    best_drop = 10000
    for epsilon in np.logspace(-2, 2, 20):
        probability = get_boltzman_policy(y_pred, epsilon)
        random_outcome = eval_prescription_probability(probability, imputed_outcome)

        final_outcome = random_outcome * T + use_previous * (1 - T)

        score = np.mean(final_outcome)
        if score < best_drop:
            best_drop = score
            best_epsilon = epsilon

    return best_epsilon


def main():
    model_save_dir = args.save_dir
    trial_number = args.trial
    test_ratio = args.test_ratio
    use_diabetes = args.diabetes

    all_result_dict = {}

    if use_diabetes:
        train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u = \
            load_diabetes_final_table_for_prescription(trial_number, test_ratio=test_ratio)
    else:
        train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u = \
            load_hypertension_final_table_for_prescription(trial_number, test_ratio=test_ratio)

    data = build_validation_set_prescription(train_all_x, train_all_y, train_all_u)

    valid_u = np.concatenate(data['valid_u'], axis=0)

    # OLS imputation model
    ols_knn_impute = pickle.load(open(model_save_dir + 'ols_knn_impute_trial_' + str(trial_number) + '.pkl', 'rb'))
    valid_x, imputed_outcome_ols = get_impute_outcome(data['valid_x'], data['valid_y'], ols_knn_impute)
    use_previous_ols = [imputed_outcome_ols[i, valid_u[i]] for i in range(len(imputed_outcome_ols))]

    # DRLR imputation model
    drlr_knn_impute = pickle.load(open(model_save_dir + 'drlr_knn_impute_trial_' + str(trial_number) + '.pkl', 'rb'))
    valid_x, imputed_outcome_drlr = get_impute_outcome(data['valid_x'], data['valid_y'], drlr_knn_impute)
    use_previous_drlr = [imputed_outcome_drlr[i, valid_u[i]] for i in range(len(imputed_outcome_drlr))]

    # get prescription model evaluated
    # 1. OLS-kNN
    ols_model = pickle.load(open(model_save_dir + 'ols_knn_trial_' + str(trial_number) + '.pkl', 'rb'))
    ols_y, ols_y_std = return_prediction_and_std(valid_x, ols_model)
    T = find_prescription_threshold(ols_y, ols_y_std, valid_x[:, 0])

    ols_epsilon = find_best_random_epsilon(use_previous_ols, T, ols_y, imputed_outcome_ols)
    all_result_dict['ols_knn_use_ols_knn'] = ols_epsilon

    drlr_epsilon = find_best_random_epsilon(use_previous_drlr, T, ols_y, imputed_outcome_drlr)
    all_result_dict['ols_knn_use_drlr_knn'] = drlr_epsilon

    # 2. drlr-kNN
    drlr_model = pickle.load(open(model_save_dir + 'drlr_knn_trial_' + str(trial_number) + '.pkl', 'rb'))
    drlr_y, drlr_y_std = return_prediction_and_std(valid_x, drlr_model)
    T = find_prescription_threshold(drlr_y, drlr_y_std, valid_x[:, 0])

    ols_epsilon = find_best_random_epsilon(use_previous_ols, T, drlr_y, imputed_outcome_ols)
    all_result_dict['drlr_knn_use_ols_knn'] = ols_epsilon

    drlr_epsilon = find_best_random_epsilon(use_previous_drlr, T, drlr_y, imputed_outcome_drlr)
    all_result_dict['drlr_knn_use_drlr_knn'] = drlr_epsilon

    # 3. LASSO
    lasso_model = pickle.load(open(model_save_dir + 'lasso_trial_' + str(trial_number) + '.pkl', 'rb'))
    lasso_y, lasso_y_std = return_prediction_and_std(valid_x, lasso_model)
    T = find_prescription_threshold(lasso_y, lasso_y_std, valid_x[:, 0])

    ols_epsilon = find_best_random_epsilon(use_previous_ols, T, lasso_y, imputed_outcome_ols)
    all_result_dict['lasso_use_ols_knn'] = ols_epsilon

    drlr_epsilon = find_best_random_epsilon(use_previous_drlr, T, lasso_y, imputed_outcome_drlr)
    all_result_dict['lasso_use_drlr_knn'] = drlr_epsilon

    # 4. CART
    cart_model = pickle.load(open(model_save_dir + 'cart_trial_' + str(trial_number) + '.pkl', 'rb'))
    cart_y, cart_y_std = return_prediction_and_std(valid_x, cart_model)
    T = find_prescription_threshold(cart_y, cart_y_std, valid_x[:, 0])

    ols_epsilon = find_best_random_epsilon(use_previous_ols, T, cart_y, imputed_outcome_ols)
    all_result_dict['cart_use_ols_knn'] = ols_epsilon

    drlr_epsilon = find_best_random_epsilon(use_previous_drlr, T, cart_y, imputed_outcome_drlr)
    all_result_dict['cart_use_drlr_knn'] = drlr_epsilon

    # save results
    json.dump(all_result_dict, open(model_save_dir + 'best_random_trial_' + str(trial_number) + '.json', 'w'),
              indent=True)


if __name__ == '__main__':
    main()
