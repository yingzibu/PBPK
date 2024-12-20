from dis_func import *

class dis_data:
    def __init__(self, df, time_points=None, col_name='Time', ref_test=['Ref', 'Test']):
        if type(df) == pd.DataFrame: # input is the whole df directly
            data_df = df.copy()
            ref_df = data_df[data_df[col_name]==ref_test[0]]
            test_df = data_df[data_df[col_name]==ref_test[1]]
        elif type(df) == list: # [ref_df, test_df]
            ref_df = df[0]; test_df = df[1]
        if time_points == None:
            print('did not specify time points, will automatically select')
            time_points = []
            time_points_float = []
            for i in ref_df.columns:
                try:
                    time_num = float(i)
                    time_points.append(i)
                    time_points_float.append(time_num)
                except: pass
        print('selected time points: ', time_points)
        self.time_points = time_points
        self.time_points_float = time_points_float

        self.ref_df = ref_df[self.time_points].reset_index(drop=True)
        self.test_df = test_df[self.time_points].reset_index(drop=True)
        self.ref_mean = list(self.ref_df.mean(axis=0))
        self.test_mean = list(self.test_df.mean(axis=0))

        # when they calculate standard deviation, they use ddof=1,
        # divided by N-1 instead of N
        # calculate standard deviation for ref_df and test_df
        self.ref_sd = list(self.ref_df.std(ddof=1))
        self.test_sd = list(self.test_df.std(ddof=1))

        # calculate cv for ref and test data, cv = sd / mean at each time point
        self.ref_cv = [i/j for i, j in zip(self.ref_sd, self.ref_mean)]
        self.test_cv = [i/j for i, j in zip(self.test_sd, self.test_mean)]
        self.ref_cov = self.ref_df.cov()
        self.test_cov = self.test_df.cov()

        dict_here = {}
        dict_here['time_point'] = self.time_points
        dict_here['ref_mean']  = self.ref_mean
        dict_here['test_mean'] = self.test_mean
        dict_here['ref_cv']  = self.ref_cv
        dict_here['test_cv'] = self.test_cv
        self.stats = pd.DataFrame.from_dict(dict_here)
        self.idx_list = [i for i in range(len(self.time_points))]
        self.idx_list_rule_85 = []
        self.rule_85 = True


    def view_data(self):
        print('*'*50)
    
        print(' reference data extracted:\n', self.ref_df)
        print('\n test data extracted:\n', self.test_df)
        print('\n ---> Mean and CV calculated:\n', self.stats)
        print('\n Ref covariance: \n', self.ref_cov)
        print('\n Test covariance:\n', self.test_cov)
        print('*'*50)

    def plot_data(self, xlabel='time in minutes', ylabel='% dissolved',
                  rlabel='ref batch', tlabel='test batch',
                  title='Mean distribution of the test and reference'):
        x_axis = self.time_points_float
        test_axis = self.test_mean
        ref_axis = self.ref_mean
        plt.plot(x_axis, test_axis, '.-', label=tlabel)
        plt.plot(x_axis, ref_axis, '.-', label=rlabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()
        plt.close()

    def apply_85_rule(self, rule_85=True):
        idx_list = []
        if rule_85: # will evaluate 85% rule,
            if len(self.idx_list_rule_85) == 0: # select from scratch
                print('---> Include 1 time point of average dissolution >85% only')
                first_85_appended = False
                for idx, (i, r, t) in enumerate(
                    zip(self.time_points, self.ref_mean, self.test_mean)):
                    if r > 85 and t > 85:
                        print(f't = {i}, mean of ref {r:.3f} & test {t:.3f} > 85%',
                              end=", ")
                        if first_85_appended == False:
                            idx_list.append(idx) # will include one time point exceeding 85%,
                            print(f'preserve time {i}')
                            first_85_appended = True
                        else: print(f'delete time {i}')
                    else: idx_list.append(idx)
                self.idx_list_rule_85 = idx_list # update rule_85 index list
                print('85% rule index list updated')

            # else: #  len(self.idx_list_rule_85) > 0:
            # self.idx_list_rule_85 is not empty, do not need to cal again
            return self.idx_list_rule_85

        else: #  rule_85 == False
            # print('Did not apply 85% rule, will calculate on all data')
            # idx_list = [i for i in range(len(self.time_points))]
            return self.idx_list

    def data_apply_85(self, rule_85=True, ver_idx=False):
        idx_list = self.apply_85_rule(rule_85=rule_85)
        time_points = list(np.take(self.time_points, idx_list))
        ref_df = self.ref_df[time_points]
        test_df = self.test_df[time_points]
        if ver_idx: return time_points, ref_df, test_df, idx_list
        return time_points, ref_df, test_df


    def cal_f2(self, bc=False, vc=False, ver=True):
        """
        Use of the f2 metric is allowed provided that following constraints are met:
         -    >= 3 time-points
         -    85% rule (<= 1 time point should be included with average dissolution >85%)
         -    CV <= 20% for 1st point, <=10% for other points
        """
        print('\n', '*'*25, 'F2 calculation', '*'*25)
        time_here, ref_df, test_df, idx_list = self.data_apply_85(
                                                    rule_85=self.rule_85,
                                                    ver_idx=True)

        print('\nEvaluate whether f2 is suitable :')

        if len(time_here) < 3:
            print('* WARNING: time points < 3, f2 may not be suitable')
        else:
            print(f'* Satisfy criteria for f2: {len(time_here)} time points, larger than 3')

        # evaluate CV<0.2 for first time point, CV < 0.1 for other time points rule
        test_cv_here = list(np.take(self.test_cv, idx_list))
        ref_cv_here  = list(np.take(self.ref_cv,  idx_list))

        cv_cond = True
        for idx, (i, t, r) in enumerate(zip(time_here, test_cv_here, ref_cv_here)):
            if idx == 0: # check whether CV <20%
                if t > 0.2 or r > 0.2:
                    print('CV at first time point exceeds 20%, f2 may not be suitable')
                    cv_cond = False
            else:
                if t > 0.1 or r > 0.1:
                    print(f'At time {i}, CV exceeds 10%, f2 may not be suitable')
                    cv_cond = False
        if cv_cond: print('* Satisfy CV criteria for f2 calculation')
        print()
        print('Cal time points: ', time_here)
        return cal_f2(ref_df, test_df, bc=bc, vc=vc, ver=ver)

    def cal_bootf2(self, B=10000, bc=False, vc=False, alpha_=0.05,
                    type_jk='nt=nr', ver=False):
        print('\n', '*'*20, 'f2 bootstrap calculation', '*'*20)
        time_points, ref_df, test_df = self.data_apply_85(rule_85=self.rule_85)
        print('Cal time points: ', time_points)
        cal_bootf2(ref_df, test_df, B=B, bc=bc, vc=vc, alpha_=alpha_,
                   type_jk=type_jk, ver=ver)


    def cal_MSD(self, tolerance_list=[10,11,13,15]):
        print('\n', '*'*25, 'MSD calculation', '*'*25)
        time_points, ref_df, test_df = self.data_apply_85(rule_85=self.rule_85)
        print('Cal time points: ', time_points)
        cal_MSD(ref_df, test_df, tolerance_list=tolerance_list)

    def cal_EDNE(self, alpha_=0.05, delta_square=None):
        print('\n', '*'*25, 'EDNE calculation', '*'*25)
        time_points, ref_df, test_df = self.data_apply_85(rule_85=self.rule_85)
        print('Cal time points: ', time_points)
        return cal_EDNE(ref_df, test_df,
                        alpha_=alpha_, delta_square=delta_square)

    def cal_T_square(self, alpha_=0.05, delta_square=None):
        print('\n', '*'*25, 'T square calculation', '*'*25)
        time_points, ref_df, test_df = self.data_apply_85(rule_85=self.rule_85)
        print('Cal time points: ', time_points)
        return cal_T_square(ref_df, test_df,
                        alpha_=alpha_, delta_square=delta_square)

    def cal_SE(self, alpha_=0.05, delta_square=None):
        print('\n', '*'*25, 'SE calculation', '*'*25)
        time_points, ref_df, test_df = self.data_apply_85(rule_85=self.rule_85)
        print('Cal time points: ', time_points)
        return cal_SE(ref_df, test_df,
                        alpha_=alpha_, delta_square=delta_square)


    def cal_GMD(self, alpha_=0.05, delta_square=None):
        print('\n', '*'*25, 'GMD calculation', '*'*25)
        time_points, ref_df, test_df = self.data_apply_85(rule_85=self.rule_85)
        print('Cal time points: ', time_points)

        return cal_GMD(ref_df, test_df,
                        alpha_=alpha_, delta_square=delta_square)

