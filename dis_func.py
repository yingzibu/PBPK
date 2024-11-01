import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from scipy.stats import norm
import warnings
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)



def cal_EDNE(ref_df, test_df, alpha_=0.05, delta_square=None):
    if delta_square == None: delta_square = 99 * len(ref_df.columns)

    ref_mean = ref_df.mean(axis=0)
    test_mean = test_df.mean(axis=0)
    diff = ref_mean - test_mean

    ref_s = ref_df.cov()
    test_s = test_df.cov()

    m = len(ref_df); n = len(test_df)
    assert m == n

    s_pool = ref_s / m + test_s / n
    u_ = norm.ppf(alpha_)

    t_up = diff.T@ diff - delta_square
    t_down = np.sqrt(4 * diff.T @ s_pool @ diff)
    t_EDNE = t_up / t_down
    if t_EDNE < u_: relation_ = '<'; conc = 'similar'
    else: relation_ = '>='; conc = 'dissimilar'

    temp_ = {}
    temp_['EDNE Test'] = [f'| T_EDNE = {t_EDNE:.3f}',
                         relation_,
                         f'[u_{alpha_}] = {u_:.3f}', conc]
    temp_df = pd.DataFrame.from_dict(temp_, orient='index',
            columns=['Test statistic', 'vs', 'critical value', 'conclusion'])
    print(temp_df)
    return t_EDNE


def cal_T_square(ref_df, test_df, alpha_=0.05, delta_square=None):
    if delta_square == None: delta_square = 0.74 ** 2

    ref_mean = ref_df.mean(axis=0)
    test_mean = test_df.mean(axis=0)
    diff = ref_mean - test_mean

    ref_s = ref_df.cov()
    test_s = test_df.cov()

    m = len(ref_df); n = len(test_df)
    assert m == n
    N = m + n
    P = len(ref_df.columns)


    s = (ref_s + test_s) / 2
    s_inv = np.linalg.inv(s)
    t_square = m * n / N * diff.T @ s_inv @ diff

    # Fcrit = scipy.stats.f.ppf(dfn=P, q=alpha_, dfd=N-P-1) previous
    # yet here we have noncentrality parameter, should use ncf function and param nc
    nc = delta_square * m * n / N
    Fcrit = scipy.stats.ncf.ppf(q=alpha_, dfn=P, dfd=N-P-1, nc=nc, loc=0, scale=1)
    adjusted_t_square = (N - 1 - P) / (N - 2) / P * t_square
    # C_ = (N-2) * P / (N - 1 - P) * Fcrit
    if adjusted_t_square < Fcrit: relation_ = '<'; conc = 'similar'
    else: relation_ = '>='; conc = 'dissimilar'

    temp_ = {}
    temp_['T^2-test'] = [f'| param*T2 = {adjusted_t_square:.3f}',
                         relation_,
                         f'[F_{alpha_}] = {Fcrit:.3f}', conc]
    temp_df = pd.DataFrame.from_dict(temp_, orient='index',
            columns=['Test statistic', 'vs', 'critical value', 'conclusion'])
    print(temp_df)
    # print(f'test statistic    critical value    conclusion')
    # print(f'(N-p-1)/[p(N-2)] T^2 =  {adjusted_t_square:.3f}     {relation_}[F_{alpha_}] = {Fcrit:.3f}   {conc}')
    return adjusted_t_square


def cal_SE(ref_df, test_df, alpha_=0.05, delta_square=None):
    if delta_square == None: delta_square = 0.74 ** 2

    ref_mean = ref_df.mean(axis=0)
    test_mean = test_df.mean(axis=0)
    diff = (ref_mean - test_mean)

    ref_s = ref_df.cov()
    test_s = test_df.cov()

    m = len(ref_df); n = len(test_df)
    assert m == n
    N = m + n
    P = len(ref_df.columns)

    s_list = [ref_s.iloc[i,i] + test_s.iloc[i,i] for i in range(P)]
    hat_delta = diff.div(pd.Series(s_list, index=ref_df.columns), axis=0)
    hat_epsilon = np.multiply(hat_delta, hat_delta)

    t_up = 2 * hat_delta.T @ diff - delta_square
    s_pool = (ref_s/m + test_s/n)

    t_down = 16 * hat_delta.T @ s_pool @ hat_delta
    hat_pi_1 = 2 * np.multiply(ref_s, ref_s)
    hat_pi_2 = 2 * np.multiply(test_s, test_s)
    t_down += 4 * hat_epsilon.T @ (hat_pi_1 / m + hat_pi_2 / n) @ hat_epsilon
    t_se = t_up / np.sqrt(t_down)

    u_ = norm.ppf(alpha_)
    if t_se < u_: relation_ = '<'; conc = 'similar'
    else: relation_ = '>='; conc = 'dissimilar'

    temp_ = {}
    temp_['SE Test'] = [f'| T_SE = {t_se:.3f}',
                         relation_,
                         f'[u_{alpha_}] = {u_:.3f}', conc]
    temp_df = pd.DataFrame.from_dict(temp_, orient='index',
            columns=['Test statistic', 'vs', 'critical value', 'conclusion'])
    print(temp_df)
    return t_se

def cal_GMD(ref_df, test_df, alpha_=0.05, delta_square=None):
    if delta_square == None: delta_square = 0.74 ** 2

    ref_mean = ref_df.mean(axis=0)
    test_mean = test_df.mean(axis=0)
    diff = (ref_mean - test_mean)

    ref_s = ref_df.cov()
    test_s = test_df.cov()

    m = len(ref_df); n = len(test_df)
    assert m == n
    N = m + n
    P = len(ref_df.columns)
    s_inv = np.linalg.inv(ref_s + test_s)
    hat_gamma = s_inv @ diff

    t_up = np.sqrt(N) * (2 * diff.T @ hat_gamma - delta_square)

    hat_v = 16 * hat_gamma @ (N/m * ref_s + N/n * test_s) @ hat_gamma
    total_sum = 0
    for i in range(P):
        for j in range(P):
            for k in range(P):
                for l in range(P):
                    total_sum += 4 * N / m * (
    ref_s.iloc[i,k] * ref_s.iloc[j,l] + ref_s.iloc[i,l] * ref_s.iloc[j,k]
                ) * hat_gamma[i] * hat_gamma[j] * hat_gamma[k] * hat_gamma[l]

                    total_sum += 4 * N / n * (
    test_s.iloc[i,k] * test_s.iloc[j,l] + test_s.iloc[i,l] * test_s.iloc[j,k]
            ) * hat_gamma[i] * hat_gamma[j] * hat_gamma[k] * hat_gamma[l]
    hat_v += total_sum
    t_gmd = t_up / np.sqrt(hat_v)
    t_gmd
    u_ = norm.ppf(alpha_)
    if t_gmd < u_: relation_ = '<'; conc = 'similar'
    else: relation_ = '>='; conc = 'dissimilar'

    temp_ = {}
    temp_['GMD Test'] = [f'| T_GMD = {t_gmd:.3f}',
                        relation_,
                        f'[u_{alpha_}] = {u_:.3f}', conc]
    temp_df = pd.DataFrame.from_dict(temp_, orient='index',
            columns=['Test statistic', 'vs', 'critical value', 'conclusion'])
    print(temp_df)
    return t_gmd

def cal_f2(ref_df, test_df, bc=False, vc=False, ver=False): # implement bias corrected
    # input is df, could do bc vc
    if type(ref_df) == pd.DataFrame: ref_mean = list(ref_df.mean(axis=0))
    # input is a list of mean, cannot do vc, bias corrected
    elif type(ref_df) == list:       ref_mean = ref_df; bc, vc=False, False
    else: print('unrecognized for ref data, type={type(ref_df)}'); return

    if type(test_df) == pd.DataFrame: test_mean = list(test_df.mean(axis=0))
    elif type(test_df) == list:       test_mean = test_df; bc, vc=False, False
    else: print('unrecognized for ref data, type={type(test_df)}'); return

    P = len(ref_mean) # number of time points
    assert len(ref_mean) == len(test_mean)
    sum_diff_square = 0
    for i, j in zip(ref_mean, test_mean):
        sum_diff_square += (i-j) ** 2

    sum_variance = 0
    if vc or bc: # will apply var or bias corrected only inf ref and test are dataframes
        try: assert len(ref_df) == len(test_df)
        except: print('Different unit number between ref and test'); return

        ref_S = [i**2 for i in np.std(ref_df, ddof=1).tolist()]
        test_S= [i**2 for i in np.std(test_df, ddof=1).tolist()]
        if bc and vc == False:
            sum_variance =  np.sum(ref_S) + np.sum(test_S)
        if vc: # apply variance-corrected f2
            bc = True
            sum_variance = 0
            for rs, ts in zip(ref_S, test_S):
                sum_s = rs + ts
                w_t = 0.5 + ts / sum_s
                w_r = 0.5 + rs / sum_s
                sum_variance += w_t * ts + w_r * rs
        n = len(ref_df)   # number of units
        sum_variance /= n

    if sum_variance > sum_diff_square: # definitely applied bc or vc
        if vc: param_name = 'vc'; vc = False
        else: param_name = 'bc' ; bc = False

        print(f'var    >   sum(|t-r|^2), cannot apply {param_name}')
        print(f'{sum_variance:.3f} > {sum_diff_square:.3f}')
        return None
    # else: # 2 conditions, sum_variance=0, sum_variance \in (0, sum_diff_square)

        # reset bc = False, vc = Fal

    D = sum_diff_square - sum_variance

    f2 = 100 - 25 * np.log10(1+D/P)
    if ver: print(f'F2 value R & T: {f2:.3f} | bc: {bc} | vc: {vc}')
    # if ver: return f2, sum_variance, sum_diff_square
    return f2


def cal_MSD(ref_df, test_df, tolerance_list=[10,11,13,15]):
    try:
        assert list(test_df.columns) == list(ref_df.columns)
    except:
        print(f'time diff: test{list(test_df.columns)} ref {list(ref_df.columns)}')
        return
    time_points = list(test_df.columns)

    P = len(time_points)
    n = len(ref_df)

    try: assert n == len(test_df)
    except:
        print(f'ref units {n} are different from test units {len(test_df)}')
        print('Check data before cal MSD'); return

    S1 = ref_df.cov()
    S2 = test_df.cov()
    S_pooled = (S1 + S2) / 2
    ref_mean = list(ref_df.mean(axis=0))
    test_mean = list(test_df.mean(axis=0))
    x2_x1 = [i-j for i, j in zip(test_mean, ref_mean)]
    a = np.array(x2_x1).reshape(len(time_points), 1)
    K = n**2/(2*n)* (2*n - P - 1) / ((2*n - 2) * P)
    Fcrit = scipy.stats.f.ppf(q=1-.1, dfn=P, dfd=2*n-P-1)
    spinv = np.linalg.inv(S_pooled.loc[time_points, time_points])
    D_M = np.sqrt(a.T @ spinv @ a)[0][0]
    print('Mahalanobis distance (T & R):', D_M)

    bound1 = a @ (1 + np.sqrt(Fcrit/(K * a.T @ spinv @ a)))
    bound2 = a @ (1 - np.sqrt(Fcrit/(K * a.T @ spinv @ a)))
    # 90% CI of Mahalanobis distance:
    DM_1 = np.sqrt(bound1.T @ spinv @ bound1)[0][0]
    DM_2 = np.sqrt(bound2.T @ spinv @ bound2)[0][0]
    DM_upper = max(DM_1, DM_2)
    DM_lower = min(DM_1, DM_2)

    print('lower bound of DM:', DM_lower)
    print('upper bound of DM:', DM_upper)


    print('DM_upper | tolerance limit | conclusion')
    for tolerance in tolerance_list:

        D_g = np.array([tolerance] * len(time_points)).reshape(len(time_points), 1)
        RD = np.sqrt(D_g.T @ spinv @ D_g)[0][0]

        if DM_upper <= RD:
            print(f'{DM_upper:.3f} \t <=  {RD:.3f}[{tolerance}%]     Similar')
        else:
            print(f'{DM_upper:.3f} \t >   {RD:.3f} [{tolerance}%]    Dissimilar')

def jackknife_statistic(ref_df, test_df, type_jk='nt=nr', 
                        bc=False, vc=False, ver=False):
    nt = len(test_df)
    nr = len(ref_df)
    jk_list = []
    if type_jk == 'nt=nr':
        assert nt == nr
        for i in range(nt):
            t = test_df.drop(i)
            r = ref_df.drop(i)
            f2 = cal_f2(t, r, bc=bc, vc=vc, ver=ver)
            jk_list.append(f2)
    elif type_jk == 'nt+nr':
        for i in range(nt):
            t = test_df.drop(i)
            f2 = cal_f2(t, ref_df, bc=bc, vc=vc, ver=ver)
            jk_list.append(f2)

        for i in range(nr):
            r = ref_df.drop(i)
            f2 = cal_f2(test_df, r, bc=bc, vc=vc, ver=ver)
            jk_list.append(f2)
    elif type_jk == 'nt*nr':
        for i in range(nt):
            t = test_df.drop(i)
            for j in range(nr):
                r = ref_df.drop(j)
                f2 = cal_f2(t, r, bc=bc, vc=vc, ver=ver)
                jk_list.append(f2)

    else:
        print("""type_jk should be one of['nt+nr', 'nt*nr', 'nt=nr']""")
        return
    return jk_list


def bootstrap_f2_list(test_df, ref_df, B=10000, bc=False, vc=False, ver=False):
    n = len(ref_df)
    f2_orig = cal_f2(test_df, ref_df, bc=bc, vc=vc, ver=ver)
    f2_estimates = []
    for i in tqdm(range(B), total=B, desc=f'bootstrap {B} samples'):
        r = ref_df.sample(n=n, replace=True) # resample with replacement
        t = test_df.sample(n=n, replace=True)
        f2 = cal_f2(t,r, bc=bc, vc=vc, ver=ver)
        f2_estimates.append(f2)
    f2_estimates.sort()
    assert len(f2_estimates) == B
    return f2_estimates, f2_orig

def BCa_jk(jk_list, f2_estimates, f2_orig, alpha_=0.05):
    m = np.mean(jk_list) # mean of the jackknife statistics
    u, d = 0, 0
    for i in jk_list:
        diff = m - i
        u += diff**3
        d += diff**2
    a_hat = u / (d**1.5)
    a_hat /= 6
    f2_num = sum(i < f2_orig for i in f2_estimates)
    z0_hat = norm.ppf(f2_num/len(f2_estimates))
    z_alpha = norm.ppf(alpha_)
    z_1_alpha = norm.ppf(1-alpha_)

    def cal_alpha(z_, z0_hat=z0_hat, a_hat=a_hat):
        temp = z0_hat + z_
        temp1 = temp / (1-a_hat*temp)
        temp1 += z0_hat
        return norm.cdf(temp1)

    alpha_1 = cal_alpha(z_alpha)
    alpha_2 = cal_alpha(z_1_alpha)
    f2_L = np.percentile(np.array(f2_estimates), 100*alpha_1)
    f2_U = np.percentile(np.array(f2_estimates), 100*alpha_2)
    return m, f2_L, f2_U

def cal_bootf2(ref_df, test_df, B=1000, bc=False, vc=False, alpha_=0.05,
               type_jk='nt=nr', ver=False):
    jk_list = jackknife_statistic(ref_df, test_df, type_jk=type_jk, bc=bc, vc=vc, ver=ver)
    result_here = {}
    f2_estimates, f2_orig = bootstrap_f2_list(ref_df, test_df, B=B, bc=bc, vc=vc, ver=ver)
    bootstrap_mean = np.mean(f2_estimates)
    m, f2_L, f2_U = BCa_jk(jk_list, f2_estimates, f2_orig, alpha_=alpha_)
    f2_L_percent = np.percentile(np.array(f2_estimates), 100*alpha_)
    f2_U_percent = np.percentile(np.array(f2_estimates), 100*(1-alpha_))
    # print(m, f2_L, f2_U)
    result_here[f'Test batch'] = [f2_orig, bootstrap_mean,
                                  f2_L_percent, f2_U_percent]

    result_here[f'Test batch BCa'] = [m, bootstrap_mean, f2_L, f2_U]
    cols_here = ['sample mean', f'{B} bootstraps mean', 'CI_L', 'CI_U']
    result_df = pd.DataFrame.from_dict(result_here, orient='index', columns=cols_here)
    print('\n')
    print(result_df)

