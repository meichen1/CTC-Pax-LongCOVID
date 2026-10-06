"""
Long COVID Analysis for Paxlovid vs Usual Care
Based on CanTreatCOVID RCC data
Analyzes long COVID symptoms at Day 90 and Week 36
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import os


from paxlovid_baseline import load_and_prepare_data, get_randomization_data


def get_longcovid_symptom_variables():
    """
    Define long COVID symptom variables from the codebook.
    Based on pages 237-259 of the codebook - COMPLETE list of all fup_ variables.
    Returns dictionary with variable names grouped by type.
    """
    
    # WHO Long COVID question (binary) ## day 90 and week 36
    who_longcovid = ['long covid', 'long36 covid']
    
    # ALL Symptom Burden Questionnaire variables - 0-3 scale (0=None, 1=Mild, 2=Moderate, 3=Severe)
    # or binary (0=No, 1=Yes) for some variables
    
    # Breathing symptoms (7 variables)
    breathing_vars = [
        'fup_breath_sob_sit',
        'fup_breath_sob_climb',
        'fup_breath_sob_lying',
        'fup_breath_wakeup',
        'fup_breath_faster',
        'fup_breath_tightness',
        'fup_breath_wheez'
    ]
    
    # Pain symptoms (4 variables)
    pain_vars = [
        'fup_pain_chest',
        'fup_pain_breath',
        'fup_pain_shooting',
        'fup_pain_burning'
    ]
    
    # Circulation symptoms (5 variables)
    circulation_vars = [
        'fup_circulation_pound',
        'fup_circulation_faint',
        'fup_circulation_dizzy',
        'fup_circulation_swell',
        'fup_circulation_cold'  # Binary: 0=No, 1=Yes
    ]
    
    # Fatigue symptoms (4 variables)
    fatigue_vars = [
        'fup_fatiguep_phys',
        'fup_fatigue_energy',
        'fup_fatigue_sleep',
        'fup_fatigue_worse'
    ]
    
    # Memory, thinking, and communication symptoms (10 variables)
    memory_vars = [
        'fup_memory_remember',
        'fup_memory_loss',
        'fup_memory_fog',
        'fup_memory_confuse',
        'fup_memory_concentrate',
        'fup_memory_plan',
        'fup_memory_word',
        'fup_memory_say',
        'fup_memory_speech',
        'fup_memory_read'
    ]
    
    # Movement symptoms (3 variables)
    movement_vars = [
        'fup_movement_tremor',
        'fup_movement_balance',
        'fup_movement_coordinate'
    ]
    
    # Sleep symptoms (4 variables)
    sleep_vars = [
        'fup_sleep_fall',
        'fup_sleep_short',
        'fup_sleep_interrupt',
        'fup_sleep_longer'
    ]
    
    # ENT symptoms (14 variables)
    ent_vars = [
        'fup_ent_smell',
        'fup_ent_taste',
        'fup_ent_sneeze',
        'fup_ent_stuffy',
        'fup_ent_congestion',
        'fup_ent_mucus',
        'fup_ent_cough',
        'fup_ent_throat',
        'fup_ent_voice',
        'fup_ent_swallow',
        'fup_ent_ear',
        'fup_ent_hearing',
        'fup_ent_tinnitus',
        'fup_ent_sound'
    ]
    
    # Stomach/GI symptoms (8 variables)
    stomach_vars = [
        'fup_stomac_pain',
        'fup_stomac_bloating',
        'fup_stomac_nausea',
        'fup_stomac_heartburn',
        'fup_stomac_wgt_loss',
        'fup_stomac_wgt_gain',
        'fup_stomac_diarrhea',
        'fup_stomac_constipation'
    ]
    
    # Muscle and joint symptoms (9 variables)
    muscle_vars = [
        'fup_muscle_pain',
        'fup_muscle_weakness',
        'fup_muscle_stiffness',
        'fup_muscle_jt_pain',
        'fup_muscle_jt_weakness',
        'fup_muscle_jt_stiffness',
        'fup_muscle_twitch',
        'fup_muscle_cramp',
        'fup_muscle_tingling'
    ]
    
    # Mental health symptoms (9 variables)
    mental_vars = [
        'fup_mental_interest',
        'fup_mental_anxiety',
        'fup_mental_sad',
        'fup_mental_harm',
        'fup_mental_mood',
        'fup_mental_appetite',
        'fup_mental_lonely',
        'fup_mental_hopeful',
        'fup_mental_person'
    ]
    
    # Skin symptoms (8 variables)
    skin_vars = [
        'fup_skin_dry',
        'fup_skin_scaly',
        'fup_skin_itchy',
        'fup_skin_red',
        'fup_skin_rash',
        'fup_skin_hives',
        'fup_skin_hair',
        'fup_skin_nails'
    ]
    
    # Eye symptoms (10 variables)
    eye_vars = [
        'fup_eyes_red',
        'fup_eyes_dry',
        'fup_eyes_itchy',
        'fup_eyes_double',
        'fup_eyes_floater',
        'fup_eyes_sense',
        'fup_eyes_watery',
        'fup_eyes_pressure',
        'fup_eyes_pain',
        'fup_eyes_rubbing'
    ]
    
    # Female-specific symptoms (7 variables)
    female_vars = [
        'fup_female_irregular',
        'fup_female_pms',
        'fup_female_blood_cloth',
        'fup_female_vag_dry',
        'fup_female_vag_discharge',
        'fup_female_climax',
        'fup_female_sex'
    ]
    
    # Male-specific symptoms (3 variables)
    male_vars = [
        'fup_male_erection',
        'fup_male_ejaculation',
        'fup_male_sex'
    ]
    
    # Other symptoms (16 variables)
    other_vars = [
        'fup_oth_fever',
        'fup_oth_chills',
        'fup_oth_flushes',
        'fup_oth_ache_body',
        'fup_oth_vertigo',
        'fup_oth_swell',
        'fup_oth_known_alergy',
        'fup_oth_knew_alergy',
        'fup_oth_urine_leakage',
        'fup_oth_urine_difficult',
        'fup_oth_urine_more',
        'fup_oth_sweating',
        'fup_oth_mouth_ulcer',
        'fup_oth_dental',
        'fup_oth_dry_mouth',
        'fup_oth_headache'
    ]
    
    # Life impact symptoms (8 variables)
    life_vars = [
        'fup_life_work',
        'fup_life_shopping',
        'fup_life_housework',
        'fup_life_move_easily',
        'fup_life_bath',
        'fup_life_friends',
        'fup_life_socialize',
        'fup_life_enjoy'
    ]
    
    # Other symptom description (3 variables)
    other_symptom_vars = [
        'fup_othsymp',  # Binary: Do you have other symptoms?
        'fup_othsyp_name',  # Text field
        'fup_othsymp_grade'  # 0-3 scale
    ]
    
    # Combine all symptom variables
    all_symptom_vars = (
        breathing_vars + pain_vars + circulation_vars + fatigue_vars +
        memory_vars + movement_vars + sleep_vars + ent_vars +
        stomach_vars + muscle_vars + mental_vars + skin_vars +
        eye_vars + female_vars + male_vars + other_vars + life_vars +
        other_symptom_vars
    )
    
    # Completion status variables
    completion_vars = {
        'day90_complete': 'Follow up Day 90_complete',
        'week36_complete': 'Follow up Week 36_complete',
        'symptom_burden_complete': 'Symptom Burden Questionnaire_complete'
    }
    
    return {
        'who_binary': who_longcovid,
        'all_symptoms': all_symptom_vars,
        'breathing': breathing_vars,
        'pain': pain_vars,
        'circulation': circulation_vars,
        'fatigue': fatigue_vars,
        'memory': memory_vars,
        'movement': movement_vars,
        'sleep': sleep_vars,
        'ent': ent_vars,
        'stomach': stomach_vars,
        'muscle': muscle_vars,
        'mental': mental_vars,
        'skin': skin_vars,
        'eye': eye_vars,
        'female': female_vars,
        'male': male_vars,
        'other': other_vars,
        'life': life_vars,
        'completion': completion_vars
    }


def calculate_pvalue_categorical(df, var, group_col='rand_group'):
    """Calculate p-value for categorical/binary variables using chi-square test (log ratio test)."""
    try:
        # Filter to only Paxlovid vs Usual Care
        df_filtered = df[df[group_col].isin(['Paxlovid', 'Usual Care'])].copy()
        
        contingency_table = pd.crosstab(df_filtered[var], df_filtered[group_col])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            return p_value
        else:
            return np.nan
    except Exception as e:
        print(f"Error calculating p-value for {var}: {e}")
        return np.nan



def calculate_pvalue_ordinal(df, var, group_col='rand_group'):
    """Calculate p-value for ordinal variables (0-3 scale) using t-test.
     Mann-Whitney U for ordinal data. the Kruskal-Wallis H test for three or more independent groups, and the Wilcoxon signed-rank test for paired or dependent data. 
     Spearman's rank correlation or Kendall's tau are used to test for an association between two ordinal variables. 
     These non-parametric tests are used instead of parametric tests like t-tests or ANOVA because they don't assume normal distribution or equal intervals.
     Other Approaches:
     Chi-Square Test: Can be used for ordinal data but ignores the order, treating categories as nominal (e.g., for independence in a contingency table).
     Ordinal Regression Models: More advanced techniques like ordered logit/probit regression can model the effect of predictors on ordinal outcomes. 
    """
    try:
        # Filter to only Paxlovid vs Usual Care
        df_filtered = df[df[group_col].isin(['Paxlovid', 'Usual Care'])].copy()
        
        paxlovid = df_filtered[df_filtered[group_col] == 'Paxlovid'][var].dropna()
        usual_care = df_filtered[df_filtered[group_col] == 'Usual Care'][var].dropna()
        
        if len(paxlovid) > 0 and len(usual_care) > 0:
            U1, p_value = mannwhitneyu(paxlovid, usual_care)  # Mann-Whitney U test
            return p_value
        else:
            return np.nan
    except Exception as e:
        print(f"Error calculating p-value for {var}: {e}")
        return np.nan


def summarize_binary_variable(df, var, group_col='rand_group'):
    """
    Summarize binary variable (Yes/No).
    FIX 2: Only Paxlovid vs Usual Care, with p-value from log ratio test.
    Returns counts, percentages, and p-value for each group.
    """
    # Filter to only Paxlovid vs Usual Care
    df_filtered = df[df[group_col].isin(['Paxlovid', 'Usual Care'])].copy()
    
    result = {}
    # Overall
    total = df_filtered[var].notna().sum()
    yes_count = (df_filtered[var] == 1).sum()
    result['overall'] = f"{yes_count}/{total} ({100*yes_count/total if total > 0 else 0:.3f}%)"  # FIX 3: 3 decimal places
    result['overall_n'] = total
    result['overall_yes'] = yes_count
    result['overall_pct'] = 100*yes_count/total if total > 0 else 0
    
    # By group (Paxlovid and Usual Care only)
    for group in ['Paxlovid', 'Usual Care']:
        group_df = df_filtered[df_filtered[group_col] == group]
        group_total = group_df[var].notna().sum()
        group_yes = (group_df[var] == 1).sum()
        result[group] = f"{group_yes}/{group_total} ({100*group_yes/group_total if group_total > 0 else 0:.3f}%)"  # FIX 3: 3 decimal places
        result[f'{group}_n'] = group_total
        result[f'{group}_yes'] = group_yes
        result[f'{group}_pct'] = 100*group_yes/group_total if group_total > 0 else 0
    
    # P-value using chi-square (log ratio test)
    result['p_value'] = calculate_pvalue_categorical(df, var, group_col)
    
    return result


def summarize_ordinal_variable(df, var, group_col='rand_group'):
    """
    Summarize ordinal variable (0-3 scale).
    FIX 2: Only Paxlovid vs Usual Care, calculate mean, SD, and t-test p-value.
    Returns counts for each level plus mean, SD, and p-value for each group.
    """
    # Filter to only Paxlovid vs Usual Care
    df_filtered = df[df[group_col].isin(['Paxlovid', 'Usual Care'])].copy()
    
    result = {}
    
    # # Overall statistics
    # total = df_filtered[var].notna().sum()
    # overall_vals = df_filtered[var].dropna()
    # result['overall_n'] = total
    # result['overall_mean'] = overall_vals.mean() if len(overall_vals) > 0 else np.nan
    # result['overall_sd'] = overall_vals.std() if len(overall_vals) > 0 else np.nan
    
    # # Distribution by level for overall
    # for level in [0, 1, 2, 3]:
    #     count = (df_filtered[var] == level).sum()
    #     result[f'overall_level_{level}'] = f"{count}/{total} ({100*count/total if total > 0 else 0:.3f}%)"  # FIX 3: 3 decimal places
    
    # By group statistics
    for group in ['Paxlovid', 'Usual Care']:
        group_df = df_filtered[df_filtered[group_col] == group]
        group_total = group_df[var].notna().sum()
        group_vals = group_df[var].dropna()
        
        result[f'{group}_n'] = group_total
        result[f'{group}_mean'] = group_vals.mean() if len(group_vals) > 0 else np.nan
        result[f'{group}_sd'] = group_vals.std() if len(group_vals) > 0 else np.nan
        
        # Distribution by level
        for level in [0, 1, 2, 3]:
            count = (group_df[var] == level).sum()
            result[f'{group}_level_{level}'] = f"{count}/{group_total} ({100*count/group_total if group_total > 0 else 0:.3f}%)"  # FIX 3: 3 decimal places
    
    # P-value using t-test
    result['p_value'] = calculate_pvalue_ordinal(df, var, group_col)
    
    return result


def summarize_group_symp_vars(df, group_vars, group_col='rand_group'):
    try:
        ## Filter to only Paxlovid vs Usual Care
        df_filtered = df.loc[df[group_col].isin(['Paxlovid', 'Usual Care']), group_vars + [group_col]].copy()
        ## change yes no questions to yes=3, no=0 if the column contains only 0 and 1 and 88
        for col in group_vars:
            unique_vals = df_filtered[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 88}):
                df_filtered[col] = df_filtered[col].replace({1: 3, 0: 0, 88: np.nan})
        ## calculate the mean score for each participant across the group_vars
        df_filtered['mean_score'] = df_filtered[group_vars].mean(axis=1, skipna=True)
        
        paxlovid_mean_score = df_filtered[df_filtered[group_col] == 'Paxlovid']['mean_score'].dropna()
        usual_care_mean_score = df_filtered[df_filtered[group_col] == 'Usual Care']['mean_score'].dropna()
        
        if len(paxlovid_mean_score) > 0 and len(usual_care_mean_score) > 0:
            U1, p_value = mannwhitneyu(paxlovid_mean_score, usual_care_mean_score)  # Mann-Whitney U test
            return p_value, paxlovid_mean_score.mean(), usual_care_mean_score.mean(), paxlovid_mean_score.std(), usual_care_mean_score.std()
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan
    except Exception as e:
        print(f"Error calculating p-value for {group_vars}: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    


def analyze_longcovid_outcomes(df_pax, pax_random):
    """
    Main analysis function for long COVID outcomes.
    FIX 2: Compares Paxlovid vs Usual Care only (exclude Antioxidant).
    Separate analyses for Day 90 and Week 36.
    """
    
    # Get variable definitions
    var_dict = get_longcovid_symptom_variables()
    # FIX 2: Filter randomization data to exclude Antioxidant arm
    pax_random_filtered = pax_random[pax_random['rand_group'].isin(['Paxlovid', 'Usual Care'])].copy()
    
    # Extract Day 90 data
    df_day90 = df_pax[df_pax['redcap_event_name'] == 'Day 90'].copy()
    df_day90 = replace_empty_with_none(df_day90)
    df_day90 = df_day90.dropna(axis=1, how='all')
    
    # Merge with randomization group and filter to Paxlovid vs Usual Care only
    df_day90 = df_day90.merge(
        pax_random_filtered[['participant_id', 'rand_group']], 
        on='participant_id', 
        how='inner'  # Inner join to exclude Antioxidant
    )
    
    # Extract Week 36 data
    df_week36 = df_pax[df_pax['redcap_event_name'] == 'Week 36'].copy()
    df_week36 = replace_empty_with_none(df_week36)
    df_week36 = df_week36.dropna(axis=1, how='all')
    
    # Merge with randomization group and filter to Paxlovid vs Usual Care only
    df_week36 = df_week36.merge(
        pax_random_filtered[['participant_id', 'rand_group']], 
        on='participant_id', 
        how='inner'  # Inner join to exclude Antioxidant
    )
    
    # FIX 2: Separate results for Day 90 and Week 36
    results_day90 = []
    results_week36 = []
    
    # 1. Completion rates - Day 90
    print("Analyzing Day 90 completion rates...")
    
    day90_complete_col = var_dict['completion']['day90_complete']
    if day90_complete_col in df_day90.columns:
        df_day90[day90_complete_col] = pd.to_numeric(df_day90[day90_complete_col], errors='coerce')
        df_day90['completed_day90'] = (df_day90[day90_complete_col] == 7).astype(float)
        result = summarize_binary_variable(df_day90, 'completed_day90')
        result['variable'] = 'Follow-up Day 90 Completion'
        result['type'] = 'Completion'
        results_day90.append(result)
    
    # Symptom burden questionnaire completion at Day 90
    symp_burden_col = var_dict['completion']['symptom_burden_complete']
    if symp_burden_col in df_day90.columns:
        df_day90['completed_symptom_burden'] = (df_day90[symp_burden_col] == '2_7').astype(float)
        result = summarize_binary_variable(df_day90, 'completed_symptom_burden')
        result['variable'] = 'Symptom Burden Questionnaire Completion'
        result['type'] = 'Completion'
        results_day90.append(result)
    
    # WHO Long COVID at Day 90
    longcovid_day90 = 'long covid'
    if longcovid_day90 in df_day90.columns:
        result = summarize_binary_variable(df_day90, longcovid_day90)
        result['variable'] = 'WHO Long COVID Definition'
        result['type'] = 'WHO Long COVID'
        results_day90.append(result)
    
    # 1b. Completion rates - Week 36
    print("Analyzing Week 36 completion rates...")
    
    week36_complete_col = var_dict['completion']['week36_complete']
    if week36_complete_col in df_week36.columns:
        df_week36[week36_complete_col] = pd.to_numeric(df_week36[week36_complete_col], errors='coerce')
        df_week36['completed_week36'] = (df_week36[week36_complete_col] == 7).astype(float)
        result = summarize_binary_variable(df_week36, 'completed_week36')
        result['variable'] = 'Follow-up Week 36 Completion'
        result['type'] = 'Completion'
        results_week36.append(result)
    
    # Symptom burden questionnaire completion at Week 36
    if symp_burden_col in df_week36.columns:
        df_week36['completed_symptom_burden'] = (df_week36[symp_burden_col] == '2_7').astype(float)
        result = summarize_binary_variable(df_week36, 'completed_symptom_burden')
        result['variable'] = 'Symptom Burden Questionnaire Completion'
        result['type'] = 'Completion'
        results_week36.append(result)
    
    # WHO Long COVID at Week 36
    longcovid_week36 = 'long36 covid'
    if longcovid_week36 in df_week36.columns:
        result = summarize_binary_variable(df_week36, longcovid_week36)
        result['variable'] = 'WHO Long COVID Definition'
        result['type'] = 'WHO Long COVID'
        results_week36.append(result)
    
    # 2. Individual symptom burden variables
    print("Analyzing symptom burden variables...")
    
    symptom_categories = ['breathing', 'pain', 'circulation', 'fatigue', 'memory', 
                         'movement', 'sleep', 'ent', 'stomach', 'muscle', 'mental', 
                         'skin', 'eye', 'female', 'male', 'other', 'life']
    
    for category in symptom_categories:
        
        if category in var_dict:
            result = {}
            p_value, paxlovid_mean, usual_care_mean, paxlovid_std, usual_care_std = summarize_group_symp_vars(df_day90, var_dict[category])
            print(f"Day 90 - {category.capitalize()}: p-value={p_value}, Paxlovid mean={paxlovid_mean}, Usual Care mean={usual_care_mean}")
            result['variable'] = 'group'+'_' + category.capitalize()
            result['type'] = f'Group Score'
            result['p_value'] = p_value
            result['group_pax_mean'] = paxlovid_mean
            result['group_uc_mean'] = usual_care_mean
            result['group_pax_std'] = paxlovid_std
            result['group_uc_std'] = usual_care_std
            
            results_day90.append(result)
            
            result = {}
            p_value, paxlovid_mean, usual_care_mean, paxlovid_std, usual_care_std = summarize_group_symp_vars(df_week36, var_dict[category])
            print(f"Week 36 - {category.capitalize()}: p-value={p_value}, Paxlovid mean={paxlovid_mean}, Usual Care mean={usual_care_mean}")
            result['variable'] = 'group'+'_' + category.capitalize()
            result['type'] = f'Group Score'
            result['p_value'] = p_value
            result['group_pax_mean'] = paxlovid_mean
            result['group_uc_mean'] = usual_care_mean
            result['group_pax_std'] = paxlovid_std
            result['group_uc_std'] = usual_care_std
            
            results_week36.append(result)
            
            for symptom_var in var_dict[category]:
                # Analyze Day 90
                if symptom_var in df_day90.columns:
                    df_day90[symptom_var] = pd.to_numeric(df_day90[symptom_var], errors='coerce')
                    unique_vals = df_day90[symptom_var].dropna().unique()
                    
                    if set(unique_vals).issubset({0, 1, 88}):
                        # Binary variable
                        result = summarize_binary_variable(df_day90, symptom_var)
                        result['variable'] = symptom_var
                        result['type'] = f'{category.capitalize()} (Binary)'
                    else:
                        # Ordinal variable (0-3)
                        result = summarize_ordinal_variable(df_day90, symptom_var)
                        result['variable'] = symptom_var
                        result['type'] = f'{category.capitalize()} (0-3)'
                    results_day90.append(result)
                
                
                # Analyze Week 36
                if symptom_var in df_week36.columns:
                    df_week36[symptom_var] = pd.to_numeric(df_week36[symptom_var], errors='coerce')
                    unique_vals = df_week36[symptom_var].dropna().unique()
                    
                    if set(unique_vals).issubset({0, 1}):
                        # Binary variable
                        result = summarize_binary_variable(df_week36, symptom_var)
                        result['variable'] = symptom_var
                        result['type'] = f'{category.capitalize()} (Binary)'
                    else:
                        # Ordinal variable (0-3)
                        result = summarize_ordinal_variable(df_week36, symptom_var)
                        result['variable'] = symptom_var
                        result['type'] = f'{category.capitalize()} (0-3)'
                    results_week36.append(result)
    
    # Convert to DataFrames
    results_day90_df = pd.DataFrame(results_day90)
    results_week36_df = pd.DataFrame(results_week36)
    
    return results_day90_df, results_week36_df, df_day90, df_week36


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("Long COVID Analysis: Paxlovid vs Usual Care")
    print("FIX 2: Separate Day 90 and Week 36 outputs")
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
    
    # Analyze long COVID outcomes
    print("Analyzing long COVID outcomes...")
    results_day90_df, results_week36_df, df_day90, df_week36 = analyze_longcovid_outcomes(df_pax, pax_random)
    
    # Save results to separate files
    output_dir = '/workspaces/CTC_covid/py_src/results_longcovid'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Day 90 results with 3 decimal places
    output_file_day90 = os.path.join(output_dir, 'longcovid_day90_analysis.csv')
    results_day90_df.to_csv(output_file_day90, index=False, float_format='%.3f')  # FIX 3: 3 decimal places
    print(f"\nDay 90 results saved to: {output_file_day90}")
    
    # Save Week 36 results with 3 decimal places
    output_file_week36 = os.path.join(output_dir, 'longcovid_week36_analysis.csv')
    results_week36_df.to_csv(output_file_week36, index=False, float_format='%.3f')  # FIX 3: 3 decimal places
    print(f"Week 36 results saved to: {output_file_week36}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Day 90 summary
    print(f"\nDay 90 analyses conducted: {len(results_day90_df)}")
    print(f"\nDay 90 analyses by type:")
    print(results_day90_df['type'].value_counts())
    
    significant_day90 = results_day90_df[results_day90_df['p_value'] < 0.05]
    if len(significant_day90) > 0:
        print(f"\n\nDay 90 significant findings (p < 0.05): {len(significant_day90)}")
        print(significant_day90[['variable', 'p_value']].to_string(index=False))
    else:
        print("\n\nNo significant Day 90 findings at p < 0.05")
    
    # Week 36 summary
    print(f"\n\nWeek 36 analyses conducted: {len(results_week36_df)}")
    print(f"\nWeek 36 analyses by type:")
    print(results_week36_df['type'].value_counts())
    
    significant_week36 = results_week36_df[results_week36_df['p_value'] < 0.05]
    if len(significant_week36) > 0:
        print(f"\n\nWeek 36 significant findings (p < 0.05): {len(significant_week36)}")
        print(significant_week36[['variable', 'p_value']].to_string(index=False))
    else:
        print("\n\nNo significant Week 36 findings at p < 0.05")
    
    return results_day90_df, results_week36_df, df_day90, df_week36


if __name__ == '__main__':
    results_day90_df, results_week36_df, df_day90, df_week36 = main()

''' 
Excel formula for concatenating mean and se in the output:
=CONCAT(P44," (",R44,")")

'''
