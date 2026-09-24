"""
Long COVID Secondary Analysis for Paxlovid Study
Step 2: Comprehensive analysis including:
1. All-cause hospitalization or ED visits within 28 days
2. recovery by day 14 (first and sustained)
3. WHO long COVID questions (Day 90 and Week 36)
4. Total symptom burden scores (Day 90 and Week 36)
5. Demographic covariates analysis
6. Linear regression associations

Based on CanTreatCOVID RCC data, paxlovid arm
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
import warnings
warnings.filterwarnings('ignore')

from longcovid_repo_utils import first_change_to_1, sustain_change_to_1
from paxlovid_baseline import load_and_prepare_data, replace_empty_with_none, get_randomization_data, prepare_baseline_data
from longcovid_symptoms import get_longcovid_symptom_variables




def calculate_hosp_ed_visits_28days(df, pax_random):
    """
    Calculate all-cause hospitalization and ED visits within 28 days.
    Returns binary outcome: 1 if any hospitalization/ED visit, 0 otherwise.
    """
    
    # Relevant events: Day 1, Day 4, Day 21, Day 28, and any events containing 'Daily' or 'Flu Pro'
    followup_events = ['Day 1', 'Day 4 (Paxlovid only)', 'Day 21', 'Day 28']
    df_followup = df[
        df['redcap_event_name'].isin(followup_events) | 
        df['redcap_event_name'].str.contains('Daily|Flu Pro', na=False)
    ].copy()
    
    # Variables for hospitalization and ED visits
    hosp_vars = ['fup_hospital_cause', 'fup4_hospital_cause', 'fup_cov_hosp',
                'fup28_cov_hosp', 'pdd_hospital', 'fpp_hospital']
    ed_vars = ['fup_er', 'fup4_er', 'fup_cov_er_visit','fup28_cov_er_visit',
               'pdd_er', 'fpp_er']
    
    # df_followup_hosp_ed = df_followup[hosp_vars + ed_vars + ['participant_id']].copy()
    results = []
    
    for pid in pax_random['participant_id'].unique():
        person_data = df_followup[df_followup['participant_id'] == pid]
        
        # Check for any hospitalization
        hosp = 0
        for hvar in hosp_vars:
            if hvar in person_data.columns:
                if person_data[hvar].notna().any():
                    # Check if any non-zero, non-empty value
                    if (person_data[hvar] == 1).any() or (person_data[hvar] == 'True').any():
                        hosp = 1
                        break
        
        # Check for ED visit
        ed = 0
        for evar in ed_vars:
            if evar in person_data.columns:
                if (person_data[evar] == 1).any() or (person_data[evar] == 'True').any():
                    ed = 1
                    break
        
        # Combined outcome: any hospitalization OR ED visit
        hosp_or_ed = 1 if (hosp == 1 or ed == 1) else 0
        
        results.append({
            'participant_id': pid,
            'hosp_28days': hosp,
            'ed_28days': ed,
            'hosp_or_ed_28days': hosp_or_ed
        })
    
    outcome_df = pd.DataFrame(results)
    
    return outcome_df


def calculate_time_to_recovery(df, pax_random, recovery_col='recover'):

    """
    Calculate both first and sustained time to recovery for a specified recovery column (e.g., 'pdd_recover' or 'fpp_recover').
    User can specify which recovery column to use. Combines pdd_ and fpp_ columns if both exist.
    Returns a DataFrame with participant_id, first_recovery_day, sustained_recovery_day, and recovered (1/0 if ever recovered).
    """

    # Daily diary events
    diary_events = [e for e in df['redcap_event_name'].unique() 
                   if 'Flu Pro Diaries' in str(e) or 'Daily e-Diary' in str(e)]
    df_diary = df[df['redcap_event_name'].isin(diary_events)].copy()

    def calculate_for_column(pid, col):
        person_data = df_diary[df_diary['participant_id'] == pid].copy()
        # Extract day number from event name
        person_data['day'] = person_data['redcap_event_name'].str.extract(r'(?:Day |Day)(\d+)')[0].astype(float)
        # Sort by day
        person_data = person_data.sort_values('day')
        # Combine pdd_ and fpp_ columns if both exist
        pdd_col = f'pdd_{col}'
        fpp_col = f'fpp_{col}'
        if pdd_col in person_data.columns and fpp_col in person_data.columns:
            series = person_data[pdd_col].combine_first(person_data[fpp_col])
        elif pdd_col in person_data.columns:
            series = person_data[pdd_col]
        elif fpp_col in person_data.columns:
            series = person_data[fpp_col]
        else:
            return None, None, 0
        series = pd.to_numeric(series, errors='coerce')
        first = first_change_to_1(series)
        sustained = sustain_change_to_1(series)
        if sustained is not None:
            sus_recovered_binary = 1 if sustained <= 14 else 0
        else:
            sus_recovered_binary = None
        if first is not None and first <= 14:
            first_recovered_binary = 1
        else:            
            first_recovered_binary = 0
        return first, sustained, first_recovered_binary, sus_recovered_binary

    # # User specifies which recovery variable to use (e.g., 'recover', 'return_health', etc.)
    # # For backward compatibility, default to 'recover' if not specified
    # import inspect
    # frame = inspect.currentframe().f_back
    # recovery_col = frame.f_locals.get('recovery_col', 'recover')

    results = []
    for pid in pax_random['participant_id'].unique():
        first, sustained, first_recovered_binary, sus_recovered_binary = calculate_for_column(pid, recovery_col)
        results.append({
            'participant_id': pid,
            f'{recovery_col}_first_recovery_day': first,
            f'{recovery_col}_sustained_recovery_day': sustained,
            f'{recovery_col}_day14_first_binary': first_recovered_binary,
            f'{recovery_col}_day14_sustained_binary': sus_recovered_binary
        })

    outcome_df = pd.DataFrame(results)
    return outcome_df


def extract_who_longcovid(df, pax_random):
    """
    Extract WHO long COVID binary outcomes for Day 90 and Week 36.
    """
    
    # Day 90
    df_day90 = df[df['redcap_event_name'] == 'Day 90'].copy()
    # Week 36
    df_week36 = df[df['redcap_event_name'] == 'Week 36'].copy()
    
    results = []
    
    for pid in pax_random['participant_id'].unique():
        # Day 90 WHO long COVID
        day90_data = df_day90[df_day90['participant_id'] == pid]
        who_day90 = np.nan
        if 'long covid' in day90_data.columns:
            val = day90_data['long covid'].iloc[0] if len(day90_data) > 0 else np.nan
            who_day90 = 1 if val == 1 else (0 if pd.notna(val) else np.nan)
        
        # Week 36 WHO long COVID
        week36_data = df_week36[df_week36['participant_id'] == pid]
        who_week36 = np.nan
        if 'long36 covid' in week36_data.columns:
            val = week36_data['long36 covid'].iloc[0] if len(week36_data) > 0 else np.nan
            who_week36 = 1 if val == 1 else (0 if pd.notna(val) else np.nan)
        
        results.append({
            'participant_id': pid,
            'who_longcovid_day90': who_day90,
            'who_longcovid_week36': who_week36
        })
    
    outcome_df = pd.DataFrame(results)
    
    return outcome_df


def calculate_symptom_burden_score(df, pax_random):
    """
    Calculate total symptom burden score by summing all symptom ratings.
    Symptom variables use 0-3 scale.
    """
    
    # All symptom variables from the codebook
    symptom_vars = get_longcovid_symptom_variables()['all_symptoms']
    
    results_day90 = []
    results_week36 = []
    
    # Day 90
    df_day90 = df[df['redcap_event_name'] == 'Day 90'].copy()
    
    for pid in pax_random['participant_id'].unique():
        person_data = df_day90[df_day90['participant_id'] == pid]
        
        if len(person_data) == 0:
            score = np.nan
            n_symptoms = 0
        else:
            # Calculate sum of all symptoms
            symptom_values = []
            for svar in symptom_vars:
                if svar in person_data.columns:
                    val = pd.to_numeric(person_data[svar].iloc[0], errors='coerce')
                    if pd.notna(val):
                        symptom_values.append(val)
            
            score = np.sum(symptom_values) if len(symptom_values) > 0 else np.nan
            n_symptoms = len(symptom_values)
        
        results_day90.append({
            'participant_id': pid,
            'symptom_burden_day90': score,
            'n_symptoms_day90': n_symptoms
        })
    
    # Week 36
    df_week36 = df[df['redcap_event_name'] == 'Week 36'].copy()
    
    for pid in pax_random['participant_id'].unique():
        person_data = df_week36[df_week36['participant_id'] == pid]
        
        if len(person_data) == 0:
            score = np.nan
            n_symptoms = 0
        else:
            symptom_values = []
            for svar in symptom_vars:
                if svar in person_data.columns:
                    val = pd.to_numeric(person_data[svar].iloc[0], errors='coerce')
                    if pd.notna(val):
                        symptom_values.append(val)
            
            score = np.sum(symptom_values) if len(symptom_values) > 0 else np.nan
            n_symptoms = len(symptom_values)
        
        results_week36.append({
            'participant_id': pid,
            'symptom_burden_week36': score,
            'n_symptoms_week36': n_symptoms
        })
    
    # Merge results
    df_day90_score = pd.DataFrame(results_day90)
    df_week36_score = pd.DataFrame(results_week36)
    
    outcome_df = df_day90_score.merge(df_week36_score, on='participant_id', how='outer')

    return outcome_df



def calculate_food_security_score(row):
    """
    Calculate food security score (0-6) based on 6 questions.
    Affirmative responses = 1, non-affirmative = 0.
    
    USDA categories:
    - 0-1: High/Marginal food security
    - 2-4: Low food security  
    - 5-6: Very low food security
    
    Note: Using available food security variables from the dataset.
    """
    score = 0
    # house_food: worried food would run out (1=Often true, 2=Sometimes true)
    val = row.get('house_food')
    if val is not None and not pd.isna(val) and int(val) <= 2:
        score += 1
    
    # house__food_last: food didn't last (1=Often true, 2=Sometimes true)
    val = row.get('house__food_last')
    if val is not None and not pd.isna(val) and int(val) <= 2:
        score += 1
    
    val = row.get('house_meals')
    if val is not None and not pd.isna(val) and int(val) <= 2:
        score += 1
        
    val = row.get('house_meal_size')
    if val is not None and not pd.isna(val) and int(val) == 1:
        score += 1
        
    # val = row.get('house_meals_cut_often')
    # if val is not None and not pd.isna(val) and int(val) <= 2:
    #     score += 1
        
    val = row.get('house_eat_less')
    if val is not None and not pd.isna(val) and int(val) == 1:
        score += 1
    
    val = row.get('house_hungry')
    if val is not None and not pd.isna(val) and int(val) == 1:
        score += 1
    
    return score  




def extract_demographics(df, pax_random):
    """
    Extract demographic variables for each participant:
    - age, sex, gender, race
    - education level, income level
    - employment, housing status
    - food security score
    """
    
    # Get baseline/randomization data
    df_baseline = prepare_baseline_data(df)
    
    results = []
    # pid = '1-348'
    for pid in pax_random['participant_id'].unique():
        person_data = df_baseline[df_baseline['participant_id'] == pid]
        
        if len(person_data) == 0:
            # Create empty row
            results.append({
                'participant_id': pid,
                'age': np.nan,
                'sex': np.nan,
                'bmi': np.nan,
                # 'gender': np.nan,
                'race': np.nan,
                'education': np.nan,
                'income': np.nan,
                'employment': np.nan,
                'housing': np.nan,
                'food_security_score': np.nan,
                'any_chronic_cond': np.nan,
                'multimorbidity_ge2': np.nan,
                'multimorbidity_ge3': np.nan,
                'vaccination': np.nan
            })
            continue
        
        # Aggregate across baseline events (take first non-null)
        age = person_data['dem_age_calc'].dropna().iloc[0] if 'dem_age_calc' in person_data.columns and person_data['dem_age_calc'].notna().any() else np.nan
        ## sex categorical
        sex = person_data['dem_sex'].dropna().iloc[0] if 'dem_sex' in person_data.columns and person_data['dem_sex'].notna().any() else np.nan
        bmi = person_data['bmi'].dropna().iloc[0] if 'bmi' in person_data.columns and person_data['bmi'].notna().any() else np.nan
        ## race categorical
        race = person_data['dem_race'].dropna().iloc[0] if 'dem_race' in person_data.columns and person_data['dem_race'].notna().any() else np.nan
        
        education = person_data['dem_education'].dropna().iloc[0] if 'dem_education' in person_data.columns and person_data['dem_education'].notna().any() else np.nan
        income = person_data['dem_house_income'].dropna().iloc[0] if 'dem_house_income' in person_data.columns and person_data['dem_house_income'].notna().any() else np.nan
        ## employment categorical
        employment = person_data['dem_empl_status'].dropna().iloc[0] if 'dem_empl_status' in person_data.columns and person_data['dem_empl_status'].notna().any() else np.nan
        ## housing categorical
        housing = person_data['dem_housing'].dropna().iloc[0] if 'dem_housing' in person_data.columns and person_data['dem_housing'].notna().any() else np.nan
        any_chronic_cond = person_data['any_disease'].dropna().iloc[0] if 'any_disease' in person_data.columns and person_data['any_disease'].notna().any() else np.nan
        multimorbidity_ge2 = person_data['multimorbidity_ge2'].dropna().iloc[0] if 'multimorbidity_ge2' in person_data.columns and person_data['multimorbidity_ge2'].notna().any() else np.nan
        multimorbidity_ge3 = person_data['multimorbidity_ge3'].dropna().iloc[0] if 'multimorbidity_ge3' in person_data.columns and person_data['multimorbidity_ge3'].notna().any() else np.nan
        vaccination = person_data['dem_vaccination_status'].dropna().iloc[0] if 'dem_vaccination_status' in person_data.columns and person_data['dem_vaccination_status'].notna().any() else np.nan

        # Calculate food security score
        fs_score = np.nan
        fs_row = person_data.iloc[0].to_dict()
        fs_score = calculate_food_security_score(fs_row)
        
        # fs_row.get('house__food_last')
        
        results.append({
            'participant_id': pid,
            'age': age,
            'sex': sex,
            'bmi': bmi,
            # 'gender': gender,
            'race': race,
            'education': education,
            'income': income,
            'employment': employment,
            'housing': housing,
            'food_security_score': fs_score,
            'any_chronic_cond': any_chronic_cond,
            'multimorbidity_ge2': multimorbidity_ge2,
            'multimorbidity_ge3': multimorbidity_ge3,
            'vaccination': vaccination
        })
    
    demo_df = pd.DataFrame(results)
    return demo_df



def run_regression_analysis(combined_df, outcome_var, outcome_type='continuous', include_randomization=True):
    """
    Run regression analysis for associations between demographics and outcomes.
    Standardizes continuous predictors (age, education, income, food_security) to mean=0, std=1.
    
    Parameters:
    - combined_df: DataFrame with all variables
    - outcome_var: name of outcome variable
    - outcome_type: 'continuous' for OLS, 'binary' for logistic
    """
    
    # Prepare data
    analysis_df = combined_df.copy()
    
    ## include randomization data to adjust for treatment arm
    if include_randomization:
        analysis_df['rand_group_01'] = pd.to_numeric(analysis_df['rand_group'].map({'Paxlovid': 1,  'Usual Care': 0}), errors='coerce')
    # Convert numeric predictors
    analysis_df['age_num'] = pd.to_numeric(analysis_df['age'], errors='coerce')
    analysis_df['education_num'] = pd.to_numeric(analysis_df['education'], errors='coerce')
    analysis_df['income_num'] = pd.to_numeric(analysis_df['income'], errors='coerce')
    analysis_df['food_security_num'] = pd.to_numeric(analysis_df['food_security_score'], errors='coerce')
    
    # Create dummy variables for categorical predictors (sex, race, employment, housing)
    # Sex: 1=Male, 2=Female -> dummy: sex_female (1 if female, 0 if male)
    analysis_df['sex_female'] = (pd.to_numeric(analysis_df['sex'], errors='coerce') == 2).astype(float)
    analysis_df['sex_intersex'] = (pd.to_numeric(analysis_df['sex'], errors='coerce') == 3).astype(float) 
    
    # Race: 1=White, 2=Black, 3=Latino, 4=Asian, 5=Indigenous, 9=Mixed, 99=Other
    # Create dummies with White as reference
    race_num = pd.to_numeric(analysis_df['race'], errors='coerce')
    analysis_df['race_mixed_other'] = (race_num.isin([2, 3, 4, 5, 9, 99])).astype(float)
    
    # Employment: 
    # 1: Unemployed/seeking employment, 
    # 2: Employed (full time including self-employed) 
    # 3: Employed (part time, including self-employed), 
    # 4: On leave from work (maternity, parental, sick leave)
    # 5: Volunteer work, unpaid
    # 6: Retired
    # 7: Student
    # 8: Housewife/husband
    # 9: Unable to work/disabled
    # 10: Other
    employment_num = pd.to_numeric(analysis_df['employment'], errors='coerce')
    analysis_df['employed'] = (employment_num.isin([2, 3, 4])).astype(float)
    
    # Housing: create own_home dummy (1=own home, 0=rent/other)
    # Codebook: 1=Renting, 2=Own home, 3=Assisted living, 4=Boarding/Group home,
    # 5=Shelter/hostel, 6=Lives with family/friend, 7=Homeless, 99=Other
    housing_num = pd.to_numeric(analysis_df['housing'], errors='coerce')
    analysis_df['own_home'] = (housing_num == 2).astype(float)
    analysis_df['rent_home'] = (housing_num == 1).astype(float)

    # Define predictors (all numeric dummy variables)
    predictors = ['rand_group_01'] if include_randomization else []
    predictors += ['age_num', 'education_num', 'income_num', 'food_security_num',
                  'sex_female', 'sex_intersex', 'race_mixed_other',
                  'employed', 'own_home', 'rent_home']
    
    # Prepare outcome
    analysis_df['outcome'] = pd.to_numeric(analysis_df[outcome_var], errors='coerce')

    # Drop rows with missing outcome
    analysis_df = analysis_df.dropna(subset=['outcome'])
    if len(analysis_df) < 10:
        print(f"Insufficient data for {outcome_var} regression (n={len(analysis_df)})")
        return None

    # Prepare X matrix
    X = analysis_df[predictors].copy()
    
    # Impute missing predictor values using MICE (sklearn's IterativeImputer)
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(random_state=42, max_iter=10)
        X_values = X.values.astype(np.float64)
        X_imputed = imputer.fit_transform(X_values)
        X = pd.DataFrame(X_imputed, columns=predictors, index=analysis_df.index)
    except Exception as e:
        print(f"MICE imputation failed ({e}), dropping rows with missing predictors.")
        X = X.dropna()
        analysis_df = analysis_df.loc[X.index]
    
    # Remove predictors with very low variance (e.g., sex_intersex with <5 positive cases)
    predictors_to_drop = []
    for pred in X.columns:
        if X[pred].std() < 0.01 or X[pred].sum() < 5:  # Very low variance or <5 positive cases
            predictors_to_drop.append(pred)
    
    if len(predictors_to_drop) > 0:
        print(f"  Removing low-variance predictors for {outcome_var}: {predictors_to_drop}")
        X = X.drop(columns=predictors_to_drop)
    
    # Standardize continuous predictors (age, education, income, food_security)
    continuous_predictors = ['age_num', 'education_num', 'income_num', 'food_security_num']
    for pred in continuous_predictors:
        if pred in X.columns:
            mean_val = X[pred].mean()
            std_val = X[pred].std()
            if std_val > 0:  # Avoid division by zero
                X[pred] = (X[pred] - mean_val) / std_val
    
    # Ensure all predictors are float64
    X = X.astype(np.float64)
    # Align y with X
    y = analysis_df.loc[X.index, 'outcome'].values.astype(np.float64)
    # Add constant
    X_with_const = sm.add_constant(X.values.astype(np.float64))
    
    if len(y) < 10:
        print(f"Insufficient data for {outcome_var} regression after cleaning (n={len(y)})")
        return None
    
    try:
        if outcome_type == 'continuous':
            model = sm.OLS(y, X_with_const).fit()
        else:  # binary
            model = sm.Logit(y, X_with_const).fit(disp=0, maxiter=100)
        
        # Extract results (const is at index 0, predictors start at index 1)
        results = []
        param_names = ['const'] + list(X.columns)
        for i, var in enumerate(param_names):
            results.append({
                'outcome': outcome_var,
                'predictor': var,
                'coefficient': float(model.params[i]),
                'se': float(model.bse[i]),
                'pvalue': float(model.pvalues[i]),
                'ci_lower': float(model.conf_int()[i, 0]),
                'ci_upper': float(model.conf_int()[i, 1]),
                'n': len(y)
            })
        
        # Adjusted ARD via recycled predictions (G-computation) – binary outcomes only
        if outcome_type == 'binary' and 'rand_group_01' in list(X.columns):
            try:
                X_np   = X.values.astype(np.float64)
                col_idx = list(X.columns).index('rand_group_01')
                X1_arr  = X_np.copy();  X1_arr[:, col_idx] = 1.0
                X0_arr  = X_np.copy();  X0_arr[:, col_idx] = 0.0
                p1_arr  = model.predict(sm.add_constant(X1_arr, has_constant='add'))
                p0_arr  = model.predict(sm.add_constant(X0_arr, has_constant='add'))
                ard_adj = float(p1_arr.mean() - p0_arr.mean())
                # Bootstrap SE (500 iterations) – refit on each bootstrap sample so
                # CIs capture model estimation uncertainty, not just averaging noise.
                # Previous code kept model fixed → artificially narrow CIs (revision 10 fix).
                np.random.seed(42)
                n_obs = X_np.shape[0]
                ards_boot = []
                for _b in range(500):
                    idx_b    = np.random.choice(n_obs, n_obs, replace=True)
                    y_b      = y[idx_b]
                    X_b_c    = sm.add_constant(X_np[idx_b], has_constant='add')
                    try:
                        model_b = sm.Logit(y_b, X_b_c).fit(disp=0, maxiter=100)
                    except Exception:
                        continue
                    # G-computation on the original (full) data using re-estimated model
                    p1_b = model_b.predict(sm.add_constant(X1_arr, has_constant='add'))
                    p0_b = model_b.predict(sm.add_constant(X0_arr, has_constant='add'))
                    ards_boot.append(float(p1_b.mean() - p0_b.mean()))
                se_boot   = float(np.std(ards_boot))
                ci_lo_adj = ard_adj - 1.96 * se_boot
                ci_hi_adj = ard_adj + 1.96 * se_boot
                z_adj     = ard_adj / se_boot if se_boot > 0 else np.nan
                pval_adj  = float(2 * (1 - stats.norm.cdf(abs(z_adj)))) if not np.isnan(z_adj) else np.nan
                results.append({
                    'outcome':     outcome_var,
                    'predictor':   '[adjusted_ard]',
                    'coefficient': round(ard_adj * 100, 3),
                    'se':          round(se_boot * 100, 3),
                    'pvalue':      round(pval_adj, 3),
                    'ci_lower':    round(ci_lo_adj * 100, 3),
                    'ci_upper':    round(ci_hi_adj * 100, 3),
                    'n':           len(y)
                })
            except Exception as e_ard:
                print(f"  Adjusted ARD computation failed: {e_ard}")

        return pd.DataFrame(results)

    except Exception as e:
        print(f"Regression failed for {outcome_var} ({e})")
        return None



def format_regression_results_table(regression_df_path: str, 
                                    outcomes=['who_longcovid_day90', 'symptom_burden_day90']):
    """
    Format regression results into a table showing adjusted OR (binary) or
    adjusted mean difference (continuous) with 95% CI.
    
    Output: DataFrame with predictors as rows and outcomes as columns.
    """
    try:
        reg_df = pd.read_csv(regression_df_path)
    except Exception as e:
        print(f"Error reading regression results: {e}")
        return None
    predictors = reg_df['predictor'].unique()
    
    rows = []
    for pred in predictors:
        row = {'Variable': pred}
        for outcome in outcomes:
            matches = reg_df[(reg_df['outcome'] == outcome) & (reg_df['predictor'] == pred)]
            
            if len(matches) == 0:
                row[f'{outcome}_est'] = 'N/A'
                continue
            
            m = matches.iloc[0]
            ci_lo = m['ci_lower']
            ci_hi = m['ci_upper']
            coef = m['coefficient']
            
            if 'longcovid' in outcome.lower():  # Binary → OR
                try:
                    or_est = np.exp(coef)
                    or_lo = np.exp(ci_lo)
                    or_hi = np.exp(ci_hi)
                    row[f'{outcome}_est'] = f"{or_est:.3f} ({or_lo:.3f}–{or_hi:.3f})"
                except:
                    row[f'{outcome}_est'] = 'N/A'
            else:  # Continuous → Mean difference
                row[f'{outcome}_est'] = f"{coef:.3f} ({ci_lo:.3f}–{ci_hi:.3f})"
            
        rows.append(row)
            
    result_df = pd.DataFrame(rows)
    ## change column names 
    for outcome in outcomes:
        if 'longcovid' in outcome.lower():
            result_df.rename(columns={f'{outcome}_est': 'Long COVID Day 90 adjusted OR (95% CI)'}, inplace=True)
        else:
            result_df.rename(columns={f'{outcome}_est': 'SBQ-LC Day 90 adjusted mean difference (95% CI)'}, inplace=True)
    return result_df

    
    
def summarize_outcomes_by_group(combined_df, outcomes_dict):
    """
    Create summary tables for each outcome by randomization group.
    """
    
    # Filter to Paxlovid and Usual Care only
    df_filtered = combined_df[combined_df['rand_group'].isin(['Paxlovid', 'Usual Care'])].copy()
    
    results = []
    for outcome_name, outcome_type in outcomes_dict.items():
        
        if outcome_name not in df_filtered.columns:
            continue
        
        row = {'outcome': outcome_name, 'type': outcome_type}
        
        if outcome_type == 'binary':
            # Binary outcome
            for group in ['Paxlovid', 'Usual Care', 'Overall']:
                if group == 'Overall':
                    group_df = df_filtered
                else:
                    group_df = df_filtered[df_filtered['rand_group'] == group]
                
                outcome_data = pd.to_numeric(group_df[outcome_name], errors='coerce')
                n = outcome_data.notna().sum()
                n_yes = (outcome_data == 1).sum()
                pct = 100 * n_yes / n if n > 0 else 0
                
                row[f'{group}_n_yes'] = f"{n_yes}/{n} ({pct:.3f}%)"
            
            # Chi-square test
            try:
                pax_data = pd.to_numeric(df_filtered[df_filtered['rand_group'] == 'Paxlovid'][outcome_name], errors='coerce')
                uc_data = pd.to_numeric(df_filtered[df_filtered['rand_group'] == 'Usual Care'][outcome_name], errors='coerce')
                
                contingency = pd.crosstab(
                    df_filtered['rand_group'],
                    pd.to_numeric(df_filtered[outcome_name], errors='coerce')
                )
                chi2, pval, dof, expected = chi2_contingency(contingency)
                row['p_value'] = round(pval, 3)
            except:
                row['p_value'] = np.nan

            # Unadjusted Absolute Risk Difference (Paxlovid - Usual Care) with 95% CI
            try:
                pax_out = pd.to_numeric(df_filtered[df_filtered['rand_group'] == 'Paxlovid'][outcome_name], errors='coerce').dropna()
                uc_out  = pd.to_numeric(df_filtered[df_filtered['rand_group'] == 'Usual Care'][outcome_name], errors='coerce').dropna()
                n_pax_ard, n_uc_ard = len(pax_out), len(uc_out)
                r_pax = pax_out.mean() if n_pax_ard > 0 else np.nan
                r_uc  = uc_out.mean()  if n_uc_ard  > 0 else np.nan
                ard   = r_pax - r_uc
                se_ard = np.sqrt(r_pax * (1 - r_pax) / n_pax_ard + r_uc * (1 - r_uc) / n_uc_ard)
                ci_lo_ard = ard - 1.96 * se_ard
                ci_hi_ard = ard + 1.96 * se_ard
                z_ard = ard / se_ard if se_ard > 0 else np.nan
                p_ard = float(2 * (1 - stats.norm.cdf(abs(z_ard)))) if not np.isnan(z_ard) else np.nan
                row['ard_unadj']        = f"{ard*100:.3f}% ({ci_lo_ard*100:.3f}% to {ci_hi_ard*100:.3f}%)"
                row['ard_unadj_pvalue'] = round(p_ard, 3)
            except Exception:
                row['ard_unadj']        = np.nan
                row['ard_unadj_pvalue'] = np.nan

        else:
            # Continuous outcome
            for group in ['Paxlovid', 'Usual Care', 'Overall']:
                if group == 'Overall':
                    group_df = df_filtered
                else:
                    group_df = df_filtered[df_filtered['rand_group'] == group]
                
                outcome_data = pd.to_numeric(group_df[outcome_name], errors='coerce')
                n = outcome_data.notna().sum()
                mean = outcome_data.mean()
                sd = outcome_data.std()
                median = outcome_data.median()
                q25 = outcome_data.quantile(0.25)
                q75 = outcome_data.quantile(0.75)
                
                row[f'{group}_n'] = n
                row[f'{group}_mean_sd'] = f"{mean:.3f} ({sd:.3f})"
                row[f'{group}_median_iqr'] = f"{median:.3f} ({q25:.3f}-{q75:.3f})"
            
            # T-test
            try:
                pax_data = pd.to_numeric(df_filtered[df_filtered['rand_group'] == 'Paxlovid'][outcome_name], errors='coerce').dropna()
                uc_data = pd.to_numeric(df_filtered[df_filtered['rand_group'] == 'Usual Care'][outcome_name], errors='coerce').dropna()
                
                if len(pax_data) > 0 and len(uc_data) > 0:
                    t_stat, pval = ttest_ind(pax_data, uc_data, equal_var=False)
                    row['p_value'] = pval
                else:
                    row['p_value'] = np.nan
            except:
                row['p_value'] = np.nan
        
        results.append(row)

    return pd.DataFrame(results)




def run_bayesian_analysis(combined_df, output_dir):
    """
    Bayesian mixed-effects regression for long COVID outcomes (revision 9).

    Both binary (who_longcovid) and continuous (symptom_burden) outcomes are
    fitted jointly across Day 90 and Week 36 using patient-level random
    intercepts.  Each model yields treatment effects at both time points:
      - Day 90  : gamma  (log-OR) or theta  (mean difference)
      - Week 36 : gamma + zeta  or  theta + xi

    Stan files used:
      binary    : Stan Code Mixed Logistic LongCovid.stan
      continuous: Stan Code Mixed Normal LongCovid.stan

    Covariates: age, vaccination status, comorbidity.
    Falls back to frequentist mixed-model (statsmodels MixedLM) if Stan fails.
    Returns a DataFrame formatted for reporting (4 rows, one per outcome/timepoint).
    """
    import os as _os
    STAN_BIN = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        'Stan Code Mixed Logistic LongCovid.stan')
    STAN_CONT = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        'Stan Code Mixed Normal LongCovid.stan')

    df = combined_df[combined_df['rand_group'].isin(['Paxlovid', 'Usual Care'])].copy()
    df['treatment'] = (df['rand_group'] == 'Paxlovid').astype(int)

    # ── Patient integer index (needed for Stan random effects) ────────────────
    pids_unique = df['participant_id'].unique()
    pid_map = {p: i + 1 for i, p in enumerate(pids_unique)}   # 1-based

    # ── Covariates (built at participant level, replicated to long format) ────
    def _get_covariate(col_names, default):
        for c in col_names:
            if c in df.columns:
                return pd.to_numeric(df[c], errors='coerce')
        return pd.Series(default, index=df.index)

    age_raw   = _get_covariate(['age', 'dem_age_calc'], np.nan)
    vax_raw   = _get_covariate(['vaccination', 'dem_vaccination_status'], np.nan)
    comorb_raw = _get_covariate(['any_chronic_cond', 'any_disease'], 0.0)

    age_raw    = age_raw.fillna(age_raw.mean())
    vax_raw    = vax_raw.fillna(vax_raw.median() if vax_raw.notna().any() else 2.0)
    comorb_raw = comorb_raw.fillna(0)

    def _std(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else s - s.mean()

    # ── Stan availability ─────────────────────────────────────────────────────
    try:
        from cmdstanpy import CmdStanModel
        stan_ok = True
    except ImportError:
        stan_ok = False
        print('  [Bayesian] cmdstanpy not available – using frequentist fallback')

    results_rows = []

    # ── Helper: build long-format data for one outcome pair ──────────────────
    def _build_long(col_d90, col_w36):
        """
        Stack day90 and week36 outcomes into a long DataFrame.
        Each row = one patient × one time point (patients may appear once or twice).
        Returns: y, pid_arr, z, t, X_std arrays plus descriptive dicts.
        """
        rows90 = df[['participant_id', 'treatment',
                     col_d90]].copy() if col_d90 in df.columns else None
        rows36 = df[['participant_id', 'treatment',
                     col_w36]].copy() if col_w36 in df.columns else None

        pieces = []
        if rows90 is not None:
            r = rows90.rename(columns={col_d90: 'y'}).copy()
            r['t_week36'] = 0
            pieces.append(r)
        if rows36 is not None:
            r = rows36.rename(columns={col_w36: 'y'}).copy()
            r['t_week36'] = 1
            pieces.append(r)
        if not pieces:
            return None

        long = pd.concat(pieces, ignore_index=True)
        long['y'] = pd.to_numeric(long['y'], errors='coerce')
        long = long.dropna(subset=['y'])
        if len(long) < 10:
            return None

        # Attach covariates by participant_id
        cov_df = pd.DataFrame({
            'participant_id': df['participant_id'].values,
            'age_raw':    age_raw.values,
            'vax_raw':    vax_raw.values,
            'comorb_raw': comorb_raw.values,
        }).drop_duplicates('participant_id')
        long = long.merge(cov_df, on='participant_id', how='left')

        pid_arr = long['participant_id'].map(pid_map).values.astype(int)
        z       = long['treatment'].values.astype(float)
        t       = long['t_week36'].values.astype(float)
        y       = long['y'].values

        X_raw = np.column_stack([
            long['age_raw'].fillna(long['age_raw'].mean()).values,
            long['vax_raw'].fillna(long['vax_raw'].median()).values,
            long['comorb_raw'].fillna(0).values,
        ]).astype(float)
        X_std = np.zeros_like(X_raw)
        for j in range(X_raw.shape[1]):
            col = X_raw[:, j]
            sd  = col.std()
            X_std[:, j] = (col - col.mean()) / sd if sd > 0 else col - col.mean()

        # Descriptive stats per timepoint per treatment
        def _desc(tp, tr):
            sub = long[(long['t_week36'] == tp) & (long['treatment'] == tr)]['y']
            return sub

        desc = {
            'd90_pax': _desc(0, 1), 'd90_uc': _desc(0, 0),
            'w36_pax': _desc(1, 1), 'w36_uc': _desc(1, 0),
        }
        return y, pid_arr, z, t, X_std, desc

    # ── Helper: run one Stan mixed-effects model ──────────────────────────────
    def _run_stan_mixed(stan_file, stan_data, out_type):
        fit = CmdStanModel(stan_file=stan_file).sample(
            data=stan_data, chains=4,
            iter_sampling=20000, iter_warmup=10000,
            seed=42, show_progress=False)
        draws = fit.draws_pd()
        return draws

    # ── Helper: frequentist fallback (pooled logit / OLS) ──────────
    def _freq_fallback(y, z, t, X_std, out_type):
        """Simple pooled regression treating time as a fixed covariate (approximation)."""
        X_full  = np.column_stack([z, t, z * t, X_std]).astype(np.float64)
        X_const = sm.add_constant(X_full)
        try:
            if out_type == 'binary':
                mod = sm.Logit(y, X_const).fit(disp=0, maxiter=200)
                coef_d90 = float(mod.params[1])          # gamma
                coef_w36 = float(mod.params[1] + mod.params[3])  # gamma + zeta
                se_d90 = float(mod.bse[1])
                # Approximate SE for Week 36 via delta method (ignore covariance term)
                se_w36 = float(np.sqrt(mod.bse[1]**2 + mod.bse[3]**2))
                def _or(coef, se):
                    return (np.exp(coef),
                            np.exp(coef - 1.96*se),
                            np.exp(coef + 1.96*se),
                            float(stats.norm.cdf(-coef/se)) if se > 0 else 0.5)
                return 'binary', _or(coef_d90, se_d90), _or(coef_w36, se_w36)
            else:
                mod = sm.OLS(y, X_const).fit()
                coef_d90 = float(mod.params[1])
                coef_w36 = float(mod.params[1] + mod.params[3])
                se_d90   = float(mod.bse[1])
                se_w36   = float(np.sqrt(mod.bse[1]**2 + mod.bse[3]**2))
                def _beta(coef, se):
                    return (coef,
                            coef - 1.96*se,
                            coef + 1.96*se,
                            float(stats.norm.cdf(-coef/se)) if se > 0 else 0.5)
                return 'continuous', _beta(coef_d90, se_d90), _beta(coef_w36, se_w36)
        except Exception as e:
            print(f'  [Bayesian] Frequentist fallback failed: {e}')
            return None

    # ── Model pairs ──────────────────────────────────────────────────────────
    model_pairs = [
        ('binary',     'who_longcovid_day90',   'who_longcovid_week36',
         'Long COVID incidence',     STAN_BIN,  0.0),
        ('continuous', 'symptom_burden_day90',  'symptom_burden_week36',
         'Symptom burden score',     STAN_CONT, 0.0),
    ]

    for out_type, col_d90, col_w36, base_label, stan_file, alpha_loc in model_pairs:
        long_data = _build_long(col_d90, col_w36)
        if long_data is None:
            continue
        y, pid_arr, z, t, X_std, desc = long_data

        Np = int(pid_arr.max())
        N  = len(y)
        M  = X_std.shape[1]

        # Descriptive strings
        def _bin_str(sub):
            n_tot = len(sub); n_yes = int(sub.sum())
            pct = 100.0 * n_yes / n_tot if n_tot > 0 else 0.0
            return f"{n_yes}/{n_tot} ({pct:.1f}%)"

        def _cont_str(sub):
            return f"{sub.mean():.1f} (SD {sub.std():.1f})"

        fmt = _bin_str if out_type == 'binary' else _cont_str

        d90_pax_str = fmt(desc['d90_pax'])
        d90_uc_str  = fmt(desc['d90_uc'])
        w36_pax_str = fmt(desc['w36_pax'])
        w36_uc_str  = fmt(desc['w36_uc'])

        # ── Try Bayesian ─────────────────────────────────────────────────────
        effect_d90    = effect_w36    = 'N/A'
        prob_d90      = prob_w36      = np.nan
        method_lbl    = 'N/A'
        bard_d90_str  = bard_w36_str  = 'N/A'   # Bayesian G-computation ARD (binary only)
        bard_prob_d90 = bard_prob_w36 = np.nan

        if stan_ok:
            try:
                if out_type == 'binary':
                    stan_data = {
                        'N': N, 'Np': Np, 'M': M,
                        'y': y.astype(int).tolist(),
                        'pid': pid_arr.tolist(),
                        'z': z.tolist(),
                        't_week36': t.tolist(),
                        'X': X_std.tolist(),
                        'alpha_loc': alpha_loc,
                    }
                    draws = _run_stan_mixed(stan_file, stan_data, out_type)
                    gamma_s = draws['gamma'].values
                    zeta_s  = draws['zeta'].values
                    or_d90  = np.exp(gamma_s)
                    or_w36  = np.exp(gamma_s + zeta_s)

                    def _or_str(or_s):
                        m  = float(np.mean(or_s))
                        lo = float(np.percentile(or_s, 2.5))
                        hi = float(np.percentile(or_s, 97.5))
                        return f"{m:.3f} ({lo:.3f}-{hi:.3f})"

                    effect_d90 = _or_str(or_d90)
                    effect_w36 = _or_str(or_w36)
                    prob_d90   = float(np.mean(or_d90 < 1))
                    prob_w36   = float(np.mean(or_w36 < 1))
                    method_lbl = 'Bayesian OR (Stan ME)'

                    # ── Bayesian G-computation ARD (revision 10) ─────────────────────
                    # Population-averaged ARD per posterior draw, marginalising random
                    # intercepts via alpha0 (population mean intercept):
                    #   ARD_d = mean_i[ sigma(alpha0_d + x_i'*b_d + gamma_d) - sigma(alpha0_d + x_i'*b_d) ]
                    # at Day 90 (t=0) and Week 36 (t=1).
                    try:
                        _step = max(1, len(draws) // 4000)
                        _a0   = draws['alpha0'].values[::_step]
                        _gam  = gamma_s[::_step]
                        _del  = draws['delta'].values[::_step]
                        _zet  = zeta_s[::_step]
                        _bcols = sorted(
                            [c for c in draws.columns if c.startswith('bbeta[')],
                            key=lambda c: int(c.split('[')[1].rstrip(']')))
                        _bmat = draws[_bcols].values[::_step]   # (n_thin, M)
                        _sig  = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
                        _X90  = X_std[t == 0]  # covariate rows for Day 90 obs
                        _X36  = X_std[t == 1]  # covariate rows for Week 36 obs
                        # Fixed linear predictor: (n_obs, n_thin)
                        _fp90 = _a0[np.newaxis, :] + _X90 @ _bmat.T
                        _fp36 = _a0[np.newaxis, :] + _X36 @ _bmat.T + _del[np.newaxis, :]
                        # ARD per draw: (n_thin,)
                        _ard90 = np.mean(_sig(_fp90 + _gam) - _sig(_fp90), axis=0)
                        _ard36 = np.mean(_sig(_fp36 + _gam + _zet) - _sig(_fp36), axis=0)

                        def _ard_pct(s):
                            m  = float(np.mean(s)) * 100
                            lo = float(np.percentile(s, 2.5)) * 100
                            hi = float(np.percentile(s, 97.5)) * 100
                            return f"{m:.2f}% ({lo:.2f}% to {hi:.2f}%)"

                        bard_d90_str  = _ard_pct(_ard90)
                        bard_w36_str  = _ard_pct(_ard36)
                        bard_prob_d90 = float(np.mean(_ard90 < 0))
                        bard_prob_w36 = float(np.mean(_ard36 < 0))
                    except Exception as _e_bard:
                        print(f'  [Bayesian ARD] Failed: {_e_bard}')

                else:
                    # Revision 9.1: standardise outcome using blinded pooled mean/SD
                    y_pooled_mean = float(np.nanmean(y))
                    y_pooled_sd   = float(np.nanstd(y, ddof=1))
                    if y_pooled_sd == 0:
                        y_pooled_sd = 1.0
                    y_std_outcome = (y - y_pooled_mean) / y_pooled_sd
                    stan_data = {
                        'N': N, 'Np': Np, 'M': M,
                        'y': y_std_outcome.tolist(),
                        'pid': pid_arr.tolist(),
                        'z': z.tolist(),
                        't_week36': t.tolist(),
                        'X': X_std.tolist(),
                        'alpha_loc': alpha_loc,
                    }
                    draws = _run_stan_mixed(stan_file, stan_data, out_type)
                    theta_s = draws['theta'].values
                    xi_s    = draws['xi'].values
                    # Back-transform to original scale (multiply by pooled SD)
                    beta_d90 = theta_s * y_pooled_sd
                    beta_w36 = (theta_s + xi_s) * y_pooled_sd

                    def _beta_str(b_s):
                        m  = float(np.mean(b_s))
                        lo = float(np.percentile(b_s, 2.5))
                        hi = float(np.percentile(b_s, 97.5))
                        return f"{m:.3f} ({lo:.3f}-{hi:.3f})"

                    effect_d90 = _beta_str(beta_d90)
                    effect_w36 = _beta_str(beta_w36)
                    prob_d90   = float(np.mean(beta_d90 < 0))
                    prob_w36   = float(np.mean(beta_w36 < 0))
                    method_lbl = 'Bayesian β (Stan ME)'

            except Exception as e_stan:
                print(f'  [Bayesian] Stan failed for {base_label}: {e_stan}')
                stan_ok = False

        ##### Frequentist fallback ──────────────────────────────────────────────
        if effect_d90 == 'N/A':
            fb = _freq_fallback(y, z, t, X_std, out_type)
            if fb is not None:
                _, res_d90, res_w36 = fb
                if out_type == 'binary':
                    or_m, or_lo, or_hi, ps = res_d90
                    effect_d90 = f"{or_m:.3f} ({or_lo:.3f}-{or_hi:.3f})"
                    prob_d90   = ps
                    or_m, or_lo, or_hi, ps = res_w36
                    effect_w36 = f"{or_m:.3f} ({or_lo:.3f}-{or_hi:.3f})"
                    prob_w36   = ps
                    method_lbl = 'Frequentist OR (pooled)'
                else:
                    bm, blo, bhi, ps = res_d90
                    effect_d90 = f"{bm:.3f} ({blo:.3f}-{bhi:.3f})"
                    prob_d90   = ps
                    bm, blo, bhi, ps = res_w36
                    effect_w36 = f"{bm:.3f} ({blo:.3f}-{bhi:.3f})"
                    prob_w36   = ps
                    method_lbl = 'Frequentist β (pooled)'

        def _fmt_prob(p):
            return round(float(p), 3) if not np.isnan(p) else 'N/A'

        results_rows.append({
            'Outcome':                    f"{base_label} at Day 90",
            'Paxlovid':                   d90_pax_str,
            'Usual Care':                 d90_uc_str,
            'Effect estimate (95% CI)':   effect_d90,
            'Probability of superiority': _fmt_prob(prob_d90),
            'Method':                     method_lbl,
            'Bayesian ARD (95% CI)':      bard_d90_str,
            'Bayesian P(ARD<0)':          _fmt_prob(bard_prob_d90),
        })
        results_rows.append({
            'Outcome':                    f"{base_label} at Week 36",
            'Paxlovid':                   w36_pax_str,
            'Usual Care':                 w36_uc_str,
            'Effect estimate (95% CI)':   effect_w36,
            'Probability of superiority': _fmt_prob(prob_w36),
            'Method':                     method_lbl,
            'Bayesian ARD (95% CI)':      bard_w36_str,
            'Bayesian P(ARD<0)':          _fmt_prob(bard_prob_w36),
        })

    results_df = pd.DataFrame(results_rows)
    if len(results_df) > 0:
        out_file = os.path.join(output_dir, 'longcovid_bayesian_results.csv')
        results_df.to_csv(out_file, index=False)
        print(f'  Bayesian results saved to: {out_file}')
        print('\n' + '=' * 70)
        print('Bayesian Mixed-Effects Analysis Results (revision 9.1)')
        print('Model: patient random intercepts + treatment x time interaction')
        print('Continuous outcomes: standardised z-score in Stan, back-transformed to original scale')
        print('Covariates: age, vaccination status, comorbidity')
        print('Binary OR < 1  / Continuous β < 0  ==>  Paxlovid superior')
        print('=' * 70)
        print(results_df.to_string(index=False))
        print('=' * 70)
    return results_df


def check_multicollinearity(combined_df, outcome_var, outcome_type='binary', include_randomization=True):
    """
    Check multicollinearity of logistic regression predictors using VIF.
    VIF > 10 indicates severe multicollinearity; 5-10 is moderate.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    analysis_df = combined_df.copy()
    if include_randomization:
        analysis_df['rand_group_01'] = pd.to_numeric(
            analysis_df['rand_group'].map({'Paxlovid': 1, 'Usual Care': 0}), errors='coerce')

    # Build predictor matrix (mirrors run_regression_analysis)
    analysis_df['age_num']          = pd.to_numeric(analysis_df['age'],               errors='coerce')
    analysis_df['education_num']    = pd.to_numeric(analysis_df['education'],         errors='coerce')
    analysis_df['income_num']       = pd.to_numeric(analysis_df['income'],            errors='coerce')
    analysis_df['food_security_num']= pd.to_numeric(analysis_df['food_security_score'], errors='coerce')
    analysis_df['sex_female']       = (pd.to_numeric(analysis_df['sex'], errors='coerce') == 2).astype(float)
    analysis_df['sex_intersex']     = (pd.to_numeric(analysis_df['sex'], errors='coerce') == 3).astype(float)
    race_num = pd.to_numeric(analysis_df['race'], errors='coerce')
    analysis_df['race_mixed_other'] = race_num.isin([2, 3, 4, 5, 9, 99]).astype(float)
    emp_num  = pd.to_numeric(analysis_df['employment'], errors='coerce')
    analysis_df['employed']         = emp_num.isin([2, 3, 4]).astype(float)
    hous_num = pd.to_numeric(analysis_df['housing'], errors='coerce')
    analysis_df['own_home']         = (hous_num == 2).astype(float)
    analysis_df['rent_home']        = (hous_num == 1).astype(float)

    predictors = (['rand_group_01'] if include_randomization else []) + [
        'age_num', 'education_num', 'income_num', 'food_security_num',
        'sex_female', 'sex_intersex', 'race_mixed_other',
        'employed', 'own_home', 'rent_home'
    ]

    analysis_df['outcome'] = pd.to_numeric(analysis_df[outcome_var], errors='coerce')
    analysis_df = analysis_df.dropna(subset=['outcome'])

    X = analysis_df[predictors].copy()
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer  = IterativeImputer(random_state=42, max_iter=10)
        X_imputed = imputer.fit_transform(X.values.astype(np.float64))
        X = pd.DataFrame(X_imputed, columns=predictors, index=analysis_df.index)
    except Exception:
        X = X.dropna()

    # Remove low-variance predictors
    for pred in list(X.columns):
        if X[pred].std() < 0.01 or X[pred].sum() < 5:
            X = X.drop(columns=[pred])

    # Standardize continuous predictors
    for pred in ['age_num', 'education_num', 'income_num', 'food_security_num']:
        if pred in X.columns:
            std_val = X[pred].std()
            if std_val > 0:
                X[pred] = (X[pred] - X[pred].mean()) / std_val

    X = X.astype(np.float64)
    X_const = sm.add_constant(X)

    vif_rows = []
    for i, col in enumerate(X_const.columns):
        if col == 'const':
            continue
        try:
            vif_val = variance_inflation_factor(X_const.values, i)
        except Exception:
            vif_val = np.nan
        vif_rows.append({
            'outcome':   outcome_var,
            'predictor': col,
            'VIF':       round(float(vif_val), 3),
            'interpretation': ('OK' if vif_val < 5 else ('Moderate' if vif_val < 10 else 'Severe'))
        })

    return pd.DataFrame(vif_rows)


def main():
    """Main execution function for Step 2 analysis."""
    
    print("=" * 80)
    print("Long COVID Secondary Analysis: Step 2 - exploratory logistic regression and bayesian mixed-effects models")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading and preparing data...")
    df_pax = load_and_prepare_data()
    print(f"Total records: {len(df_pax)}")
    print(f"Total participants: {df_pax['participant_id'].nunique()}")
    print()
    
    # Get randomization data
    pax_random = get_randomization_data(df_pax)
    print(f"Randomized participants: {len(pax_random)}")
    print(f"Randomization groups:\n{pax_random['rand_group'].value_counts()}")
    print()
    
    # 1. Calculate hospitalization/ED visits within 28 days
    print("Calculating hospitalization/ED visits within 28 days...")
    hosp_ed_df = calculate_hosp_ed_visits_28days(df_pax, pax_random)
    print(f"Hospitalization/ED data: {len(hosp_ed_df)} participants")
    print()
    
    # 2. Calculate time to recovery
    print("Calculating time to recovery...")
    recovery_df = calculate_time_to_recovery(df_pax, pax_random)
    print(f"Recovery data: {len(recovery_df)} participants")
    print(f"Recovered: {(recovery_df['recover_day14_sustained_binary']==1).sum()}")
    
    # 3. Extract WHO long COVID
    print("Extracting WHO long COVID outcomes...")
    who_lc_df = extract_who_longcovid(df_pax, pax_random)
    print(f"WHO long COVID data: {len(who_lc_df)} participants")
    print()
    
    # 4. Calculate symptom burden scores
    print("Calculating symptom burden scores...")
    symptom_score_df = calculate_symptom_burden_score(df_pax, pax_random)
    print(f"Symptom burden data: {len(symptom_score_df)} participants")
    print()
    
    # 5. Extract demographics
    print("Extracting demographic covariates...")
    demo_df = extract_demographics(df_pax, pax_random)
    print(f"Demographic data: {len(demo_df)} participants")
    print()
    
    # Combine all outcomes and demographics
    print("Combining all data...")
    combined_df = pax_random.copy()
    combined_df = combined_df.merge(hosp_ed_df, on='participant_id', how='left')
    combined_df = combined_df.merge(recovery_df, on='participant_id', how='left')
    combined_df = combined_df.merge(who_lc_df, on='participant_id', how='left')
    combined_df = combined_df.merge(symptom_score_df, on='participant_id', how='left')
    combined_df = combined_df.merge(demo_df, on='participant_id', how='left')
    ## drop 2 columns called 'age' and 'redcap_record_metadata
    combined_df = combined_df.drop(columns=['redcap_subject_screening_number', 'redcap_record_metadata'], errors='ignore')
    print(f"Combined dataset: {len(combined_df)} participants, {len(combined_df.columns)} variables")
    
    # Save combined dataset
    output_dir = '/workspaces/CTC_covid/py_src/results_longcovid'
    os.makedirs(output_dir, exist_ok=True)
    
    combined_file = os.path.join(output_dir, 'longcovid_step2_combined_data.csv')
    combined_df.to_csv(combined_file, index=False, float_format='%.3f')
    print(f"Combined data saved to: {combined_file}")
    
    
    # Run regression analyses
    regression_outcomes = {
        # 'hosp_or_ed_28days': 'binary',
        # 'recover_day14_first_binary': 'binary',
        # 'recover_day14_sustained_binary': 'binary',
        'who_longcovid_day90': 'binary',
        'who_longcovid_week36': 'binary',
        'symptom_burden_day90': 'continuous',
        'symptom_burden_week36': 'continuous'
    }
    
    all_regression_results = []
    pax_regression_results = []
    uc_regression_results = []
    
    for outcome, outcome_type in regression_outcomes.items():
        print(f"  Analyzing {outcome}...")
        print(f"    Total sample (adjusted for treatment)")
        result = run_regression_analysis(combined_df, outcome, outcome_type)
        if result is not None:
            all_regression_results.append(result)
        
        # ── REVISION 11.1: Comment out pax-only and uc-only regressions ──
        # print(f"    Paxlovid subgroup")
        # pax_result = run_regression_analysis(combined_df[combined_df['rand_group'] == 'Paxlovid'], outcome, outcome_type, include_randomization=False)
        # if pax_result is not None:
        #     pax_regression_results.append(pax_result)
        # print(f"    Usual Care subgroup")
        # uc_result = run_regression_analysis(combined_df[combined_df['rand_group'] == 'Usual Care'], outcome, outcome_type, include_randomization=False)
        # if uc_result is not None:
        #     uc_regression_results.append(uc_result)
    
    # ── REVISION 11.1: Save only total sample results ──
    if len(all_regression_results) > 0:
        regression_df = pd.concat(all_regression_results, ignore_index=True)
        regression_file = os.path.join(output_dir, 'longcovid_step2_regression_results.csv')
        regression_df.to_csv(regression_file, index=False, float_format='%.3f')
        print(f"\nRegression results saved to: {regression_file}")
    
    # ── REVISION 11.1: Commented out pax and uc subgroup saves ──
    # if len(pax_regression_results) > 0:
    #     pax_regression_df = pd.concat(pax_regression_results, ignore_index=True)
    #     pax_regression_file = os.path.join(output_dir, 'longcovid_step2_pax_regression_results.csv')
    #     pax_regression_df.to_csv(pax_regression_file, index=False, float_format='%.3f')
    #     print(f"Paxlovid subgroup regression results saved to: {pax_regression_file}")
    # if len(uc_regression_results) > 0:
    #     uc_regression_df = pd.concat(uc_regression_results, ignore_index=True)
    #     uc_regression_file = os.path.join(output_dir, 'longcovid_step2_uc_regression_results.csv')
    #     uc_regression_df.to_csv(uc_regression_file, index=False, float_format='%.3f')
    #     print(f"Usual Care subgroup regression results saved to: {uc_regression_file}")

    #  Format regression results table
    reg_results_table = format_regression_results_table(
        os.path.join(output_dir, 'longcovid_step2_regression_results.csv'),
        outcomes=['who_longcovid_day90', 'symptom_burden_day90']
    )
    if reg_results_table is not None:
        print("\nAdjusted associations table:")
        print(reg_results_table.to_string(index=False))
        # Save to CSV
        table_file = os.path.join(output_dir, "logistic_adjusted_associations_table.csv")
        reg_results_table.to_csv(table_file, index=False)
        print(f"  Saved: {table_file}")
    
    
    # ── VIF multicollinearity check ──────────────────────────────────────────
    print("\nChecking multicollinearity (VIF) for logistic regression models...")
    vif_all = []
    for vif_out in ['who_longcovid_day90', 'who_longcovid_week36']:
        vdf = check_multicollinearity(combined_df, vif_out, 'binary')
        if vdf is not None and len(vdf) > 0:
            vif_all.append(vdf)
    if vif_all:
        vif_table = pd.concat(vif_all, ignore_index=True)
        vif_file  = os.path.join(output_dir, 'longcovid_vif_multicollinearity.csv')
        vif_table.to_csv(vif_file, index=False, float_format='%.3f')
        print(f"VIF results saved to: {vif_file}")
        print(vif_table.to_string(index=False))

    

    # ── Summarize outcomes by group (includes unadjusted ARD) ────────────────
    print("\nSummarizing outcomes variables by randomization group...")
    outcome_summary = summarize_outcomes_by_group(combined_df, regression_outcomes)
    summary_file = os.path.join(output_dir, 'longcovid_step2_outcome_summary.csv')
    outcome_summary.to_csv(summary_file, index=False, float_format='%.3f')
    print(f"Outcome summary saved to: {summary_file}")


    print("=" * 80)
    print("STEP 2 ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files in: {output_dir}")
    
    
    return combined_df, outcome_summary if len(all_regression_results) > 0 else None


if __name__ == '__main__':
    combined_df, outcome_summary = main()

