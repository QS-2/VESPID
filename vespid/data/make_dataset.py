# -*- coding: utf-8 -*-
import logging
import re
from pathlib import Path
import re

import click
import fuzzymatcher
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from vespid.data import (expand_dict_lists,
                         extract_named_value_from_lists_of_dicts)
from vespid.data.neo4j_tools import Nodes, Relationships
from vespid.data.neo4j_tools.utils import generate_ingest_code
from vespid.data.semantic_scholar import query_semantic_scholar
from vespid.features.geocoding import get_from_openstreetmap
from vespid import setup_logger

logger = setup_logger(__name__)


# Recodings for SDR data
STANDARD_RECODE = {
    b'Y': 'Yes', 
    b'N': 'No', 
    b'L': 'No'
}

REGION_RECODE = {
    b'01': 'West',
    b'02': 'Midwest',
    b'03': 'Northeast',
    b'04': 'South',
    b'05': 'Abroad',
    b'L': np.nan,
    b'M': 'No Response'
}

FOUR_POINT_LIKERT_RECODE = {
    b'L': np.nan,
    b'1': 'Very satisfied',
    b'2': 'Somewhat satisfied',
    b'3': 'Somewhat dissatisfied',
    b'4': 'Very dissatisfied'
}

DISCIPLINE_RECODE = {
    b'11': 'Computer and information sciences',
    b'12': 'Mathematics and statistics',
    b'21': 'Agricultural and food sciences',
    b'22': 'Biological sciences',
    b'23': 'Environmental life sciences',
    b'31': 'Chemistry, except biochemistry',
    b'32': 'Earth, atmospheric and ocean sciences',
    b'33': 'Physics and astronomy',
    b'34': 'Other physical sciences',
    b'41': 'Economics',
    b'42': 'Political and related sciences',
    b'43': 'Psychology',
    b'44': 'Sociology and anthropology',
    b'45': 'Other social sciences',
    b'51': 'Aerospace, aeronautical and astronautical engineering',
    b'52': 'Chemical engineering',
    b'53': 'Civil and architectural engineering',
    b'54': 'Electrical and computer engineering',
    b'55': 'Industrial engineering',
    b'56': 'Mechanical engineering',
    b'57': 'Other engineering',
    b'60': 'S&E related fields',
    b'70': 'Non-S&E fields',
    b'98': np.nan
}


def find_and_replace_limit_clause(query, limit=20_000):
    '''
    Determines if a LIMIT clause has been used in the provided query and, if
    so, replaces the LIMIT value with ``limit``. If no previous LIMIT clause
    is found, will add one in with the value of ``limit``.


    Parameters
    ----------
    query: str.

    limit: int.


    Returns
    -------
    2-tuple of the form (modified_query, limit_from_original_query)
    '''
    
    limit_full_results = None
    pattern = re.compile(r" LIMIT \d+", re.IGNORECASE)
    regex_search = pattern.search(query)
    
    if regex_search:
        limit_full_results = int(regex_search.group().split(' ')[-1])
        logger.debug(f"User-defined LIMIT in original query is {limit_full_results}")

        # Check that the paging limit isn't higher than the user-requested limit
        # if it is, just use user-requested limit
        if limit < limit_full_results:
            query_modified = re.sub(pattern, f" LIMIT {limit}", query)

        else:
            query_modified = query[:]

    else:
        query_modified = query[:]
        
    return query_modified, limit_full_results



def condense_columns_to_string(df, columns, column_prefix_to_drop, 
    drop_source_columns=False):
    '''
    Takes multiple same-valued columns (e.g. dummy variables) and condenses
    them into a single column comprised of the names of the source columns
    wherever the source column value was 'Yes' and combining the results
    into a single semicolon-delimited string.

    
    Parameters
    ----------
    df: pandas DataFrame containing the columns you want to use.

    columns: list of str column names

    column_prefix_to_drop: str that can be used as the prefix of the column
        names we want to drop when condensing. For example, if each column
        starts with 'Academic_', we will condense the results into values
        that are the same as their source column names minus the 'Academic_'
        prefix string.

    drop_source_columns: bool. If True, will do an in-place drop of ``columns``
        from ``df`` to clean up the DataFrame after condensing.


    Returns
    -------
    pandas Series with the condensed string representations of the data.
    '''

    allowed_values = ['Yes', 'No']

    for column in columns:
        # Check that only values we expect exist 
        uniques = set(df[column].dropna().unique())
        if set(allowed_values) != uniques:
            raise ValueError(
                f"Column ``{column}`` does not only contain values \
                {allowed_values}"
                )

    # Set 0 to NaN, as string concat will skip
    df_temp = df[columns].replace({'No': np.nan})

    # Set all Yes's to the name of the column
    for column in columns:
        df_temp[column].replace('Yes', column[len(column_prefix_to_drop):], 
            inplace=True)

    # TODO: faster way to combine? How to make into lists while skipping NaN values?
    # Concatenate string values into single string per row, so Neo4j can 
    # later separate them into lists
    tqdm.pandas(desc='Concatenating strings')
    output = df_temp.progress_apply(
        lambda row: row.str.cat(sep=';'), axis=1
    )

    # Replace empty strings with nulls
    output.replace({'': np.nan}, inplace=True)

    if drop_source_columns:
        logger.info(
            f"Dropping columns {columns} from DataFrame after condensing..."
            )
        df.drop(columns=columns, inplace=True)

    return output


def etl_public_sdr_neo4j(filepath):
    '''
    Performs various column renaming and value re-coding steps
    to get the public SDR data ready for ingestion into Neo4j.


    Parameters
    ----------
    filepath: path-like str indicating where to find the CSV file that serves
        as the raw source data. Common value when running this script from 
        notebooks/ is "../data/external/SDR_Public_2017/epsd17.sas7bdat".


    Returns
    -------
    pandas DataFrame containing the cleaned and condensed dataset to be used.
    '''

    df = pd.read_sas(filepath)

    # Rules based on survey instrument
    # Note that ones with "_SKIP" are ones I don't think we care too much about
    column_renames = {
        'OBSNUM': 'OBSNUM_SKIP',  # simply an integer tracking row number
        'REFYR': 'DataVintage',
        'SURID': 'SURID_SKIP',  # just constant value of 'SDR'
        'WRKG': 'WRKG_SKIP',
        'WRKGP': 'LongTermEmployed',  # continuous multi-year employment
        # LFSTATUS uses these first three to derive it, so no need for them
        'LOOKWK': 'LOOKWK_SKIP',
        # 1 = Employed, 2 = Unemployed, 3 = Not in Labor Force - need to dummy
        'LFSTAT': 'CurrentEmploymentStatus',
        'NWOTP': 'NWOTP_SKIP',  # Skip due to LFSTATUS?
        'NWLAY': 'NWLAY_SKIP',  # Skip due to LFSTATUS?
        'NWSTU': 'NWSTU_SKIP',  # Skip due to LFSTATUS?
        'NWFAM': 'NWFAM_SKIP',  # Skip due to LFSTATUS?
        'NWOCNA': 'NWOCNA_SKIP',  # Skip due to LFSTATUS?
        'NWNOND': 'NWNOND_SKIP',  # Skip due to LFSTATUS?
        'N2OCMLST': 'LatestJobCode',  # 7 bins + logical skip
        'FTPRET': 'PreviouslyRetired',  # usual Y/N/L
        'EMUS': 'Employer_US_Based',
        'EMRGP': 'Employer_USRegionCode',  # values 1-5 or L; assumed US
        'EMSIZE': 'EmployerSize',  # Codes 1-8 for size bins + L
        'NEWBUS': 'NewBusiness',
        # a few codes + L - MIGHT be better than EMTPP's education focus, but not clear
        'EMSECDT': 'EMSECDT_SKIP',
        'EMSECSM': 'EMSECSM_SKIP',  # generalized sector code, unnecessary
        # not sure why, but is all employer types EXCEPT educational...
        'NEDTPP': 'NEDTPP_SKIP',
        # Most detailed codes, things like 'non-profit' or 'self-employed', but education-oriented for some reason?
        'EMTPP': 'EmployerType',
        'EMED': 'EMED_SKIP',  # sector info on educational
        'EDTP': 'EDTP_SKIP',  # sector info on type of edu
        # combine all academics into single job title column. IsAcademic column will flag
        'ACADADJF': 'Academic_AdjunctFaculty',
        'ACADADMN': 'Academic_DeanToPresident',
        'ACADNA': 'IsAcademic',
        'ACADOTHP': 'Academic_RA_TA_Other',
        'ACADPDOC': 'Academic_Postdoc',
        'ACADRCHF': 'Academic_ResearchFaculty',
        'ACADTCHF': 'Academic_TeachingFaculty',
        # Skip? Seems a little too granular/specific...
        'FACRANK': 'Academic_FacultyRank',
        'TENSTA': 'TENSTA_SKIP',
        'TENI': 'IsTenured',  # 1: N/A, 2: No, 3: Yes, 'l'
        'N2OCPRMG': 'N2OCPRMG_SKIP',  # too general of job codes
        'N2OCPRNG': 'Job',  # detailed codes we can map to
        # TODO: combine with other MGR variables first to get CollegeLevelDifficulty variable
        'MGRNAT': 'MGRNAT_SKIP',
        # TODO: combine with other MGR variables first to get CollegeLevelDifficulty variable
        'MGRSOC': 'MGRSOC_SKIP',
        # TODO: combine with other MGR variables first to get CollegeLevelDifficulty variable
        'MGROTH': 'MGROTH_SKIP',
        'PDIX': 'PDIX_SKIP',
        'PDTRAIN': 'PDTRAIN_SKIP',
        'PDTROUT': 'PDTROUT_SKIP',
        'PDPERPL': 'PDPERPL_SKIP',
        'PDEMPL': 'PDEMPL_SKIP',
        'PDEXPECT': 'PDEXPECT_SKIP',
        'PDOTHER': 'PDOTHER_SKIP',
        'PDPRI': 'PDPRI_SKIP',
        'PDSEC': 'PDSEC_SKIP',
        'OCEDRLP': 'JobSimilarityToPhD',  # Low num = higher alignment
        # the 'why' of working outside PhD field doesn't seem necessary for our use case
        'NRPAY': 'NRPAY_SKIP',
        'NRCON': 'NRCON_SKIP',  # the 'why'...
        'NRLOC': 'NRLOC_SKIP',  # the 'why'...
        'NRCHG': 'NRCHG_SKIP',  # the 'why'...
        'NRFAM': 'NRFAM_SKIP',  # the 'why'...
        'NROCNA': 'NROCNA_SKIP',  # the 'why'...
        'NROT': 'NROT_SKIP',  # the 'why'...
        'NRREA': 'NRREA_SKIP',  # the 'why'...
        'NRSEC': 'NRSEC_SKIP',  # the 'why'...
        # Only using primary and secondary results from WAPRI and WASEC
        'ACTCAP': 'ACTCAP_SKIP',
        'ACTDED': 'ACTDED_SKIP',
        'ACTMGT': 'ACTMGT_SKIP',
        'ACTRD': 'ACTRD_SKIP',
        'ACTRDT': 'ACTRDT_SKIP',
        'ACTRES': 'ACTRES_SKIP',
        'ACTTCH': 'ACTTCH_SKIP',
        'WAACC': 'WAACC_SKIP',
        'WABRSH': 'WABRSH_SKIP',
        'WAAPRSH': 'WAAPRSH_SKIP',
        'WADEV': 'WADEV_SKIP',
        'WADSN': 'WADSN_SKIP',
        'WACOM': 'WACOM_SKIP',
        'WAEMRL': 'WAEMRL_SKIP',
        'WAMGMT': 'WAMGMT_SKIP',
        'WAPROD': 'WAPROD_SKIP',
        'WASVC': 'WASVC_SKIP',
        'WASALE': 'WASALE_SKIP',
        'WAQM': 'WAQM_SKIP',
        'WATEA': 'WATEA_SKIP',
        'WAOT': 'WAOT_SKIP',
        'WAPRSM': 'WorkActivity_Primary',
        'WASCSM': 'WorkActivity_Secondary',
        'SUPWK': 'IsSupervisor',
        'WKSYR': 'WKSYR_SKIP',
        'WKSWK': 'WKSWK_SKIP',
        'WKSLYR': 'WKSLYR_SKIP',
        # binned to $1000 buckets, with 509000 probably corresponding to $509K+ (since there are 862 with this value) - NOTE that logical skip here is 9999998
        'SALARYP': 'Salary_MainJob',
        # binned same as SALARYP, altho goes higher and has a $652K+ bin at the top
        'EARNP': 'Salary_AllJobs',
        'SATSAL': 'JobSatisfaction_Salary',
        'SATBEN': 'JobSatisfaction_Benefits',
        'SATSEC': 'JobSatisfaction_JobSecurity',
        'SATLOC': 'JobSatisfaction_Location',
        'SATADV': 'JobSatisfaction_Advancement',
        'SATCHAL': 'JobSatisfaction_Challenge',
        'SATRESP': 'JobSatisfaction_Responsibility',
        'SATIND': 'JobSatisfaction_Independence',
        'SATSOC': 'JobSatisfaction_SocietalContribution',
        'JOBSATIS': 'JobSatisfaction_Overall',
        'HRSWKP': 'HoursWorkedWeekly',
        'PJWTFT': 'PJWTFT_SKIP',  # desire to work 35+ hours
        'PJRET': 'PJRET_SKIP',  # reason for working less than 35 hrs - previously retired
        'PJSTU': 'PJSTU_SKIP',
        'PJFAM': 'PJFAM_SKIP',
        'PJOCNA': 'PJOCNA_SKIP',
        'PJHAJ': 'PJHAJ_SKIP',
        'PJNOND': 'PJNOND_SKIP',
        'PJOT': 'PJOT_SKIP',
        'JOBINS': 'JOBINS_SKIP',  # health insurance offered by employer
        'JOBPENS': 'JOBPENS_SKIP',
        'JOBPROFT': 'JOBPROFT_SKIP',
        'JOBVAC': 'JOBVAC_SKIP',
        'GOVSUP': 'GOVSUP_SKIP',  # can be derived from the next columns
        'FSDOD': 'Funding_DeptDefense',  # Dept of Defense funding received
        'FSDED': 'Funding_DeptEducation',
        'FSDOE': 'Funding_DeptEnergy',
        'FSNIH': 'Funding_NIH',
        'FSHHS': 'Funding_DeptHHS',
        'FSNASA': 'Funding_NASA',
        'FSNSF': 'Funding_NSF',
        'FSOT': 'Funding_OtherFedAgencies',
        'FSDK': 'Funding_UnknownFedAgency',
        # covers no change, different job same employer, same job different employer, and all changed
        'EMSMI': 'JobChange',
        'CHPAY': 'CHPAY_SKIP',  # CH* variables give reason for changing job
        'CHCON': 'CHCON_SKIP',
        'CHLOC': 'CHLOC_SKIP',
        'CHCHG': 'CHCHG_SKIP',
        'CHFAM': 'CHFAM_SKIP',
        'CHSCH': 'CHSCH_SKIP',
        'CHLAY': 'CHLAY_SKIP',
        'CHRET': 'CHRET_SKIP',
        'CHOT': 'CHOT_SKIP',
        'WKTRNI': 'RecentProTraining',  # did you do any pro training in last year
        'WTRSKL': 'WTRSKL_SKIP',  # WT* is reason for training
        'WTROPPS': 'WTROPPS_SKIP',
        'WTRLIC': 'WTRLIC_SKIP',
        'WTRCHOC': 'WTRCHOC_SKIP',
        'WTREM': 'WTREM_SKIP',
        'WTRPERS': 'WTRPERS_SKIP',
        'WTROT': 'WTROT_SKIP',
        'WTREASN': 'WTREASN_SKIP',
        'PROMTGI': 'RecentConference',
        'PRMBRPB': 'NumProSocieties',  # number of assn's respondent is a member in
        # similar to FS* variants but provides importance on hypothetical, not current, job
        'FACSAL': 'FACSAL_SKIP',
        'FACBEN': 'FACBEN_SKIP',
        'FACSEC': 'FACSEC_SKIP',
        'FACLOC': 'FACLOC_SKIP',
        'FACADV': 'FACADV_SKIP',
        'FACCHAL': 'FACCHAL_SKIP',
        'FACRESP': 'FACRESP_SKIP',
        'FACIND': 'FACIND_SKIP',
        'FACSOC': 'FACSOC_SKIP',
        # all about degrees beyond PhD completed in last 2 years, seems too specific
        'TCDGCMP': 'TCDGCMP_SKIP',
        'MRDG': 'MRDG_SKIP',
        'NMRMEMG': 'NMRMEMG_SKIP',
        'NMRMENGP': 'NMRMENGP_SKIP',
        'MR5YRP': 'MR5YRP_SKIP',
        # 'MRPBP15C': 'MRPBP15C_SKIP', # only in one code book and not the other...
        'MRRGNP': 'MRRGNP_SKIP',
        'MRDGRUS': 'MRDGRUS_SKIP',
        'MRCAR': 'MRCAR_SKIP',
        'MRCHG': 'MRCHG_SKIP',
        'MRSKL': 'MRSKL_SKIP',
        'MRLIC': 'MRLIC_SKIP',
        'MRADV': 'MRADV_SKIP',
        'MRINT': 'MRINT_SKIP',
        'MROTP': 'MROTP_SKIP',
        # all these are about if respondent is currently enrolled in courses
        'ACSIN': 'ACSIN_SKIP',
        'ACFPT': 'ACFPT_SKIP',
        'ACDRG': 'ACDRG_SKIP',
        'NACEDMG': 'NACEDMG_SKIP',
        'ACCAR': 'ACCAR_SKIP',
        'ACCHG': 'ACCHG_SKIP',
        'ACSKL': 'ACSKL_SKIP',
        'ACLIC': 'ACLIC_SKIP',
        'ACADV': 'ACADV_SKIP',
        'ACINT': 'ACINT_SKIP',
        'ACOTP': 'ACOTP_SKIP',
        'ACCCEP': 'ACCCEP_SKIP',
        'MARIND': 'MARIND_SKIP',  # removes too much detail from relationship status
        'MARSTA': 'RelationshipStatus',
        'SPOWK': 'SpousePartnerWorking',
        # TODO: combine with other SP variables first to get CollegeLevelDifficulty_SpousePartner variable
        'SPNAT': 'SPNAT_SKIP',
        # TODO: combine with other SP variables first to get CollegeLevelDifficulty_SpousePartner variable
        'SPSOC': 'SPSOC_SKIP',
        # TODO: combine with other SP variables first to get CollegeLevelDifficulty_SpousePartner variable
        'SPOT': 'SPOT_SKIP',
        'CHLVIN': 'CHLVIN_SKIP',  # only asks about children at home, can derive from counts
        # Only tells you 0, 1, 2+ but we can derive more than that from age-grouped columns
        'CHTOTPB': 'TotalChildrenInHome',
        'CHU2IN': 'HaveChildrenUnder2',  # under age 2, y/n
        'CH25IN': 'HaveChildren2to5',
        'CH611IN': 'HaveChildren6to11',
        'CH1218IN': 'HaveChildren12to18',
        'CH19IN': 'HaveChildren19Plus',  # 19+ years old, y/n
        'FNINUS': 'LivingInUS',  # y/n
        # includes things like permanent resident, etc. but no info on specific visas
        'CTZN': 'USCitizenshipStatus',
        'CTZUSIN': 'CTZUSIN_SKIP',
        'CTZUS': 'CTZUS_SKIP',
        'CTZFOR': 'CTZFOR_SKIP',
        # 5 year intervals, starting at "29 or younger" which is pretty low-res but low-response-rate too (858 responses!)
        'AGEGRP': 'AgeBin',
        'HCAPIN': 'HaveDisability',
        'DIFAGEGR': 'AgeOfFirstDisability',  # 5 year intervals
        'BAAYR5P': 'YearOfBachelors',  # 5 year bins
        'BADGRUS': 'BachelorsInUS',
        'BARGNP': 'Bachelors_USRegionCode',
        'BTHUS': 'BTHUS_SKIP',  # US birth status is in citizenship question
        'CH6IN': 'CH6IN_SKIP',
        'CHUN12': 'CHUN12_SKIP',
        'COHORT': 'COHORT_SKIP',  # just a constant value of 'SDR'
        'CTZN_DRF': 'CTZN_DRF_SKIP',
        # the D<numeric>* ones are all about extra degrees beyond the 'principal doctorate'
        'D25YRP': 'D25YRP_SKIP',
        'D2DG': 'D2DG_SKIP',
        'D2DGRUS': 'D2DGRUS_SKIP',
        'D2RGNP': 'D2RGNP_SKIP',
        'D35YRP': 'D35YRP_SKIP',
        'D3DG': 'D3DG_SKIP',
        'D3DGRUS': 'D3DGRUS_SKIP',
        'D3RGNP': 'D3RGNP_SKIP',
        'DIFBIR': 'DIFBIR_SKIP',
        'DIFNO': 'DIFNO_SKIP',
        # why aren't these in the survey instrument but they have data???
        'EDDAD': 'EducationLevel_Father',
        'EDMOM': 'EducationLevel_Mother',
        'GENDER': 'Gender',  # only male or female? WOW.
        'HDAY5P': 'PhDYear',
        'HDDGRUS': 'PhDInUS',
        'HDRGNP': 'PhD_USRegionCode',
        'MINRTY': 'MINRTY_SKIP',  # get better race details with RACTHMP
        'NBAMEMG': 'NBAMEMG_SKIP',
        'NBAMENGP': 'BachelorsDiscipline',
        'ND2MEMG': 'ND2MEMG_SKIP',
        'ND3MEMG': 'ND3MEMG_SKIP',
        'NDGMEMG': 'NDGMEMG_SKIP',
        'NDGMENGP': 'PhDDiscipline',
        # differentiates first US PhD from main PhD potentially
        'NSDRMEDTOD': 'NSDRMEDTOD_SKIP',
        'NSDRMEMTOD': 'NSDRMEMTOD_SKIP',
        'NSDRMENTOD': 'NSDRMENTOD_SKIP',
        'PDUSFOR': 'PDUSFOR_SKIP',
        'RACETHMP': 'Race',
        'SDR5YRP': 'SDR5YRP_SKIP',
        'SDRRGNP': 'SDRRGNP_SKIP',
        'SEHTOD': 'SEHTOD_SKIP',
        'SRVMODE': 'SRVMODE_SKIP',
        'WAPRI': 'WAPRI_SKIP',
        'WASEC': 'WASEC_SKIP',
        'WTSURVY': 'SurveyWeight'
    }

    # Rename columns
    df.rename(columns=column_renames, inplace=True)

    # Drop columns we flagged in the dict as being unhelpful
    df.drop(columns=df.columns[df.columns.str.contains("_SKIP")],
            inplace=True)

    # Note that {**dict1, **dict2} will effectively concetenate dicts into a new dict
    value_replacements = {  # format is 'column_name': {'old_value1': new_value1, ...}

        # combine this with general jobs eventually for single column
        **{col: STANDARD_RECODE for col in df.columns if 'Academic_' in col and 'FacultyRank' not in col},
        'Academic_FacultyRank': {
            b'L': np.nan,
            b'1': np.nan,
            b'2': np.nan,
            b'3': 'Professor',
            b'4': 'Associate Professor',
            b'5': 'Assistant Professor',
            b'6': 'Instructor',
            b'7': 'Lecturer',
            b'8': 'Other'
        },
        # Have to make L = 1 because skipping is the same as "I'm not an academic"
        'IsAcademic': {
            b'Y': 'No', # 1
            b'N': 'Yes', # 0
            b'L': 'No' # 1
        },
        'BachelorsInUS': {
            b'L': np.nan, 
            b'M': 'No response', 
            b'Y': 'Yes',
            b'N': 'No'
        },
        'Bachelors_USRegionCode': REGION_RECODE,
        **{col: STANDARD_RECODE for col in df.columns if 'HaveChildren' in col},

        # TODO: private SDR data will presumably give actual numbers
        # for each age group, making this irrelevant
        'TotalChildrenInHome': {b'1': '1', b'2': '2+', b'98': '0'},
        'USCitizenshipStatus': {
            b'1': 'U.S. citizen, Native',
            b'2': 'U.S. citizen, Naturalized',
            b'3': 'Permanent Resident',
            b'4': 'Temporary Resident',
            b'5': 'Non-Citizen living outside U.S.'
        },
        'AgeOfFirstDisability': {  # Uses max age of bin as value
            98: np.nan,
            20: 24,
            25: 29,
            30: 34,
            35: 39,
            40: 44,
            45: 49,
            50: 54,
            55: 59,
            60: 64,
            65: 69,
            70: 75
        },
        'YearOfBachelors': {9998: np.nan},  # Uses bottom of year bins
        # TODO: play with making this np.nan and into an indicator variable with pipeline grid search
        'Salary_AllJobs': {9999998: 0},
        # TODO: play with making this np.nan and into an indicator variable with pipeline grid search
        'Salary_MainJob': {9999998: 0},
        **{col: {
            b'1': 'Less than high school',
            b'2': 'High school diploma or equivalent',
            b'3': 'Some college, vocational, or trade school (including 2-year degrees)',
            b'4': 'Bachelors degree',
            b'5': 'Masters degree',
            b'6': 'Professional degree (e.g. JD, MD, etc.)',
            b'7': 'Doctorate',
            b'8': np.nan,
            b'M': "No response"
        } for col in df.columns if 'EducationLevel' in col},  # Mom and dad edu levels, missing and N/A both get NaN
        # does including M for missing in the dict screw things up (since it's not included in this column)?
        'Employer_USRegionCode': REGION_RECODE,
        'EmployerSize': {  # TODO: grid search on midpoint-numeric-values vs. categoricals
            b'1': '10 or fewer',
            b'2': '11-24',
            b'3': '25-99',
            b'4': '100-499',
            b'5': '500-999',
            b'6': '1000-4999',
            b'7': '5000-24999',
            b'8': '25000+',
            b'L': np.nan
        },
        'JobChange': {
            b'L': np.nan,
            b'1': 'Same employer, same job',
            b'2': 'Same employer, different job',
            b'3': 'Different employer, same job',
            b'4': 'Different employer, different job'
        },
        'EmployerType': {
            b'01': 'Primary or secondary school',
            b'02': '2-year/junior college or technical institute',
            b'03': '4-year college',
            b'04': 'Medical school',
            b'05': 'University research institute',
            b'06': 'Other (educational institution)',
            b'10': 'Private for-profit',
            b'11': 'Private non-profit',
            b'12': 'Self-employed, not incorporated',
            b'13': 'Self-employed, incorporated',
            b'14': 'Local government',
            b'15': 'State government',
            b'17': 'U.S. government',
            b'18': 'Other',
            b'19': 'Non-U.S. government',
            b'L': np.nan  # why did they skip numbers seemingly at random?
        },
        'Employer_US_Based': STANDARD_RECODE,
        'LivingInUS': STANDARD_RECODE,
        **{col: STANDARD_RECODE for col in df.columns if 'Funding_' in col},
        'PreviouslyRetired': STANDARD_RECODE,
        # TODO: keep as category for now, but numerify in grid search
        'Gender': {b'M': 'Male', b'F': 'Female'},
        'HaveDisability': STANDARD_RECODE,
        'PhDInUS': STANDARD_RECODE,
        'PhD_USRegionCode': REGION_RECODE,
        # TODO: try numerics in pipeline grid search
        'HoursWorkedWeekly': {1: '20 or less', 2: '21-35', 3: '36-40', 4: '40+', 98: np.nan},
        'CurrentEmploymentStatus': {b'1': 'Employed', b'2': 'Unemployed', b'3': 'Not in labor force'},
        'RelationshipStatus': {
            b'1': 'Married',
            b'2': 'Living in a marriage-like relationship',
            b'3': 'Widowed',
            b'4': 'Separated',
            b'5': 'Divorced',
            b'6': 'Never married'
        },
        'LatestJobCode': {
            b'1': 'Computer and mathematical scientists',
            b'2': 'Biological, agricultural and other life scientists',
            b'3': 'Physical and related scientists',
            b'4': 'Social and related scientists',
            b'5': 'Engineers',
            b'6': 'S&E related occupations',
            b'7': 'Non-S&E Occupations',
            b'8': np.nan
        },
        'Job': {
            b'11':  'Computer and information scientists',
            b'12':  'Mathematical scientists',
            b'18':  'Postsecondary teachers - computer and math sciences',
            b'21':  'Agricultural & food scientists',
            b'22':  'Biological and medical scientists',
            b'23':  'Environmental life scientists',
            b'28':  'Postsecondary teachers - life and related sciences',
            b'31':  'Chemists, except biochemists',
            b'32':  'Earth scientists, geologists and oceanographers',
            b'33':  'Physicists and astronomers',
            b'34':  'Other physical & related scientists',
            b'38':  'Postsecondary teachers - physical and related sciences',
            b'41':  'Economists',
            b'42':  'Political scientists',
            b'43':  'Psychologists',
            b'44':  'Sociologists and anthropologists',
            b'45':  'Other social and related scientists',
            b'48':  'Postsecondary teachers - social and related sciences',
            b'51':  'Aerospace, aeronautical or astronautical engineers',
            b'52':  'Chemical engineers',
            b'53':  'Civil, architectural or sanitary engineers',
            b'54':  'Electrical or computer hardware engineers',
            b'55':  'Industrial engineers',
            b'56':  'Mechanical engineers',
            b'57':  'Other engineers',
            b'58':  'Postsecondary teachers - engineering',
            b'61':  'Health-related occupations',
            b'62':  'S&E managers',
            b'63':  'S&E Pre-college Teachers',
            b'64':  'S&E technicians and technologists',
            b'65':  'Other S&E-related occupations',
            b'71':  'Non-S&E Managers',
            b'72':  'Management-related occupations',
            b'73':  'Non-S&E precollege teachers',
            b'74':  'Non-S&E postsecondary teachers',
            b'75':  'Social services and related occupations',
            b'76': 'Sales and marketing occupations',
            b'77': 'Art, humanities and related occupations',
            b'78': 'Other non-S&E occupations',
            b'98': np.nan
        },
        'BachelorsDiscipline': DISCIPLINE_RECODE,
        'PhDDiscipline': DISCIPLINE_RECODE,
        'NewBusiness': STANDARD_RECODE,
        'JobSimilarityToPhD': {
            b'L': np.nan,
            b'1': 'Closely related',
            b'2': 'Somewhat related',
            b'3': 'Not related'
        },  # Need to create indicator variable for the NAs when numeric
        # 6 is actually 6+; TODO: try making categorical for pipeline grid search
        'NumProSocieties': {98: 0},
        'RecentConference': STANDARD_RECODE,
        'Race': {
            b'1': 'Asian ONLY, non-Hispanic',
            b'3': 'Black ONLY, non-Hispanic',
            b'4': 'Hispanic, any race',
            b'5': 'White ONLY, non-Hispanic',
            b'7': 'Other races including multiracial individuals, non-Hispanic'
        },
        'JobSatisfaction_Overall': FOUR_POINT_LIKERT_RECODE,
        **{col: FOUR_POINT_LIKERT_RECODE for col in df.columns if 'JobSatisfaction_' in col},
        'SpousePartnerWorking': {b'L': np.nan, b'1': 'Full-time', b'2': 'Part-time', b'3': 'No'},
        'IsSupervisor': STANDARD_RECODE,
        # Could replace with TENSTA original variable for more response levels
        'IsTenured': {
            b'L': np.nan, 
            b'1': np.nan, 
            b'2': 'No', # 0
            b'3': 'Yes' # 1
        },
        'WorkActivity_Primary': {
            b'L': np.nan,
            b'1': 'R&D',
            b'2': 'Teaching',
            b'3': 'Management and Administration',
            b'4': 'Computer Applications',
            b'5': 'Other'
        },
        'WorkActivity_Secondary': {
            b'L': np.nan,
            b'1': 'R&D',
            b'2': 'Teaching',
            b'3': 'Management and Administration',
            b'4': 'Computer Applications',
            b'5': 'Other',
            b'6': np.nan
        },
        'RecentProTraining': STANDARD_RECODE,
        'LongTermEmployed': STANDARD_RECODE
    }

    df.replace(value_replacements, inplace=True)

    # Combine Academic_* into a single AcademicTitle column
    columns = [col for col in df.columns if 'Academic_' in col
               and 'Rank' not in col and col != 'IsAcademic']

    df['AcademicTitle'] = condense_columns_to_string(df, columns=columns,
        column_prefix_to_drop='Academic_', drop_source_columns=True)

    # Do the same string-combo process with Funding_ columns
    columns = [col for col in df.columns if 'Funding_' in col]

    df['FederalFundingSources'] = condense_columns_to_string(df, columns=columns,
        column_prefix_to_drop='Funding_', drop_source_columns=True)

    # Set EducationLevel_* columns to be ordered categoricals
    # Note that NaN is not considered a category by pandas
    categories_ordered = [
        'No response',
        'Less than high school',
        'High school diploma or equivalent',
        'Some college, vocational, or trade school (including 2-year degrees)',
        'Bachelors degree',
        'Masters degree',
        'Professional degree (e.g. JD, MD, etc.)',
        'Doctorate'
    ]

    for column in ['EducationLevel_Father', 'EducationLevel_Mother']:
        df[column] = df[column].astype('category')\
        .cat.set_categories(categories_ordered, ordered=True)

    return df


def get_unique_authors(author_dicts, keys_to_return):
    '''
    Given an iterable of lists of author dictionaries, find the 
    unique authors in the dataset.


    Parameters
    ----------
    author_dicts: iterable (e.g. pandas Series) of lists of dictionaries,
        one list per record (usually a publication) and one dictionary per
        unique author in that list.

    keys_to_return: list of str that name the keys from each author's 
        dictionary that should be returned (e.g. their authorId).


    Returns
    -------
    pandas DataFrame containing the unique author data.
    '''

    output = expand_dict_lists(author_dicts, unique_ids=['authorId'])

    return output[keys_to_return]


def get_unique_funders(series):
    '''
    Given funding information that includes at least the funding entity (e.g.
    government agency), extract the unique funders and give them unique IDs.


    Parameters
    ----------
    series: pandas Series of lists of dicts that contain funding information.
        Each dict should have at least a 'funder' key.


    Returns
    -------
    pandas DataFrame of funding organizations and a unique ID for each.
    '''

    output = pd.DataFrame(expand_dict_lists(series, 'funder')['funder'])

    #output['id'] = output.reset_index().index
    logger.warning("Funder node IDs are currently their names but will be more formally defined in the future")
    output['id'] = output['funder']

    return output.rename(columns={'funder': 'name'})


def get_unique_institutions(series, unique_ids='institution'):
    '''
    Parses all of the author-level institution information and creates 
    a DataFrame with nothing but unique institution information included.


    Parameters
    ----------
    series: pandas Series containing lists of dicts of author information.
        Each dict corresponds to a single author and is expected to be of the
        form {'name': <author_name>, <other top-level key-value pairs>, 
        'institutions': [{'institution': <name>, 'address': <address_str>, ...}]}

        Note that each institutions dict is a list of dicts itself, one dict
        per institution-address combination.

    unique_ids: str or list of str. Indicates the key(s) from each dict that 
        should be considered unique identifier(s) for the dict and can be 
        used to de-duplicate the records.


    Returns
    -------
    pandas DataFrame containing unique institution names and all addresses
    associated with them.
    '''
    tqdm.pandas(desc='Extracting institution info from author dicts')
    lists_of_dicts = series.dropna().explode()\
        .progress_apply(lambda value: value['institutions'])

    output = expand_dict_lists(lists_of_dicts, unique_ids)\
    .rename(columns={'institution': 'name'})

    # Group on institution name
    logger.info("Aggregating institutional locations...")
    output = output.groupby('name').agg(list).reset_index()
    logger.info("Institutional location aggregation complete!")

    #output['id'] = output.index
    logger.warning("Institution node IDs are currently their names but will be more formally defined in the future")
    output['id'] = output['name']

    # Drop information wherein the institution name is blank - not too useful
    output = output[output['name'] != '']

    return output.dropna()


def add_author_ids(df, inplace=False):
    '''
    Using merged records of publications connecting data from Web of Science
    to Semantic Scholar, combine author information into a single column that
    captures unique author IDs as well as institutional affiliation information.


    Parameters
    ----------
    df: pandas DataFrame in which every row is a publication. Assumes that
        you have columns 'authors_ss' and 'authors_wos' that each contain
        lists of author dicts.
        
    inplace: bool. If True, indicates that `df` should be changed in-memory
        instead of creating a copy.


    Returns
    -------
    A copy of ``df`` with a new column 'authors' that contains lists of dicts,
    one dict per author. The author dicts have been enriched further to include,
    where available, Semantic Scholar author IDs and other metadata from that
    corpus.
    '''
    
    # Skip it all if there's no S2 authors data
    if df.dropna(subset=['authors_ss', 'authors_wos']).empty:
        logger.warning("No S2 author data found")
        if inplace:
            df['authors'] = pd.Series([]*len(df))
            return df
        else:
            output = df.copy()
            output['authors'] = pd.Series([]*len(df))
            return output

    def _add_id_single_publication(row):
        '''
        Adds author IDs for just a single publication.
        '''
        
        df_ss = pd.DataFrame(row['authors_ss'])
        df_wos = pd.DataFrame(row['authors_wos'])
        
        # SQLite has some reserved keywords we need to avoid
        illegal_tokens = ["AND", "OR", "NEAR", "NOT"]
        
        # Build pattern that identifies when these tokens are present
        # then replace them with an empty string
        pattern = re.compile(
            '|'.join([fr'(\b{t}\b)' for t in illegal_tokens]), 
            flags=re.IGNORECASE
        )
        df_ss['name'] = df_ss['name'].str.replace(pattern, "", regex=True)
        df_wos['name'] = df_wos['name'].str.replace(pattern, "", regex=True)
        
        #FIXME: this package hasn't been updated since 2019, 
        # so we'll need to start managing it ourselves as a forked submodule
        results = fuzzymatcher.fuzzy_left_join(
            df_ss,
            df_wos,
            'name',
            'name'
            )

        #TODO: consider only allowing matches above a certain score
        results.drop(columns=['best_match_score', '__id_left', '__id_right'],
            inplace=True)

        results.rename(columns={
            'name_left': 'name', 
            'name_right': 'name_wos'
            },
            inplace=True)

        # While we're here and looking at the author dict,
        # check if we're alphabetized by last name!
        results['authorListAlphabetizedLastName'] = \
        (results['last_name'] > results['last_name'].shift(-1)).sum() == 0 and \
        len(results) > 1

        return results.to_dict(orient='records')

    tqdm.pandas(
        desc='Matching Semantic Scholar author IDs to WoS author names'
    )
    if inplace:
        df['authors'] =  df.dropna(
            subset=['authors_ss', 'authors_wos']
            ).progress_apply(_add_id_single_publication, axis=1)

        return df
        
    else:
        output = df.copy()
        output['authors'] =  df.dropna(
            subset=['authors_ss', 'authors_wos']
            ).progress_apply(_add_id_single_publication, axis=1)

        return output


def add_semantic_scholar_to_wos(
    df, 
    api_key,
    max_concurrent_requests=None,
    n_jobs=None
):
    '''
    Given a dataset of paper records from Web Of Science (WoS), add in fields
    from matching records we pull from Semantic Scholar to augment it.


    Parameters
    ----------
    df: pandas DataFrame of paper records

    api_key: str. Semantic Scholar API key, if one is available. API keys can 
        be requested directly from Semantic Scholar (usually only granted for 
        research purposes). If None, rate limit is 100 queries every 5 
        minutes. If there is a key, rates are decided at the time of key 
        generation and that information should be provided by the relevant
        Semantic Scholar contact providing the key.
        
    max_concurrent_requests: int. If not None, this value will be used to limit
        how many concurrent Semantic Scholar requests are allowed per second. 
        Otherwise, will default to the hard-coded max (usually 30).
        
    n_jobs: int or None. If n_jobs is 0 or None, no parallelization is assumed.
        If n_jobs is -1, uses all but one available CPU core.


    Returns
    -------
    Copy of ``df`` with extra columns coming from the Semantic Scholar API.
    '''
    
    output = df.copy()
    
    good_id_index = df[
        (df['DOI'].notnull())
        | (df['id_ss'].notnull())
    ].index
    num_records_with_id = len(good_id_index)
    pct_records_with_id = round(num_records_with_id / len(df), 4) * 100
    logger.info(f"{num_records_with_id:,} records ({pct_records_with_id}%) "
                "have either a DOI or a Semantic Scholar paper ID. "
"As such, they will be augmented with Semantic Scholar data.")
    
    if num_records_with_id == 0:
        logger.info("No identifiers found, returning un-augmented data...")
        return output

    # First the ones with Semantic Scholar ID, 
    # as this is guaranteed to work as expected
    if df['id_ss'].notnull().sum() > 0:
        logger.info("Finding S2 records based off of S2 paper ID...")
        df_ss = query_semantic_scholar(
            df.loc[df['id_ss'].notnull(), 'id_ss'],
            query_type='S2 Paper ID',
            api_key=api_key,
            max_concurrent_requests=max_concurrent_requests,
            n_jobs=n_jobs,
            fields_of_interest=None # Use default
        )
        
    else:
        logger.warning("No S2 IDs found, skipping S2 ID queries")
        df_ss = pd.DataFrame()

    # Now look at ones that have a DOI but no SS ID
    remaining_records_index = df[
        (df['id_ss'].isnull())
        & (df['DOI'].notnull())
    ].index
    if len(remaining_records_index) > 0:
        logger.info("Finding S2 records based off of DOI...")
        df_ss = df_ss.append(
            query_semantic_scholar(
                df.loc[remaining_records_index, 'DOI'], 
                query_type='DOI', 
                api_key=api_key,
                max_concurrent_requests=max_concurrent_requests,
                n_jobs=n_jobs
            )
        ).sort_index()


    if not df_ss.empty:
        # Will rename things with _wos or _ss *only* if duplicate columns exist...
        abstracts_from_wos = 'abstract' in df.columns
        
        output = df.join(df_ss, how='left', lsuffix='_wos', rsuffix='_ss')

        output = output.drop(columns=['id_ss']).rename(columns={
                'authors': 'authors_ss', 
                'url': 'url_ss',
                'paperId': 'id_ss',
                'id': 'id_wos'
            })
        
        # This indicates that we only got abstracts from one source during JOIN
        if 'abstract' in output.columns:
            if abstracts_from_wos:
                logger.warning(f"Looks like we have no S2 abstracts!")
                output.rename(columns={'abstract': 'abstract_wos'},
                              inplace=True)
                output['abstract_ss'] = np.nan
                
            else:
                logger.warning(f"Looks like we have no WoS abstracts!")
                output.rename(columns={'abstract': 'abstract_ss'},
                              inplace=True)
                output['abstract_wos'] = np.nan
            
        # Track which data source, if either, is missing an abstract
        output['abstract_missing'] = output[['abstract_wos', 'abstract_ss']].apply(
                list_null_columns,
                rename={
                    'abstract_wos': 'Web of Science', 
                    'abstract_ss': 'Semantic Scholar'
                },
                axis=1
            )    
        
        # Make sure empty lists are nullified to save memory
        output.loc[
            output['abstract_missing'].str.len() == 0, 
            'abstract_missing'
            ] = np.nan
        
        # Ensure that abstracts from WoS are used by default
        # and that SS abstracts fill in where WoS ones aren't available 
        output['abstract'] = output['abstract_wos']
        
        # Now check S2 for viability of abstract       
        missing_abstracts_index = output.loc[output['abstract'].isnull()].index
        output.loc[missing_abstracts_index, 'abstract'] = \
        output.loc[missing_abstracts_index, 'abstract_ss']
        
        # Track where it came from
        output.loc[
            output['abstract_wos'].notnull(), 
            'abstract_source'
            ] = 'Web of Science'
        
        output.loc[
            output['abstract_ss'].notnull(), 
            'abstract_source'
            ] = 'Semantic Scholar'
        
    else:
        logger.warning("Entirety of S2 results (DOI and S2 ID) null!")
        output.rename(columns={'id': 'id_wos'}, inplace=True)
        
        # All missing abstracts must be from WoS!
        output['abstract_missing'] = np.nan
        output.loc[output['abstract'].isnull(), 'abstract_missing'] = \
            'Web of Science'
            
        output['authors_ss'] = np.nan
        output['url_ss'] = np.nan
        output['id_ss'] = np.nan
        
        output['abstract_source'] = 'Web of Science'
    
    return output
    
def list_null_columns(series, rename=None):
    '''
    Check individual pandas Series for which index labels are null and 
    track them via a list. Intended to be used via 
    pd.DataFrame.apply(list_null_columns, rename=None, axis=1)

    Parameters
    ----------
    series : pandas Series
        The row/column to check for null values
    rename : dict of form {'old_label': 'new_label'}, optional
        Lets you rename index labels to something more intuitive 
        (e.g. `{'abstract_wos': 'Web of Science'}`), by default None

    Returns
    -------
    pandas Series of lists
        Each list contains the index label(s) for a given axis that were
        null, if used via DataFrame.apply().
    '''
    output = series[series.isnull()].index.to_series()    
    if rename is not None:
        output.replace(rename, inplace=True)
    return output.tolist()


def make_author_nodes(
    df, 
    filepath=None,
    graph=None,
    batch_size=2_000
):
    '''
    Given data wherein each record is a publication, extract the unique
    authors and their metadata, then save to CSV.


    Parameters
    ----------
    df: pandas DataFrame that must contain, at least, the column 'authors',
        with data represented in that column as lists of dictionaries, one 
        dict per author on a given paper.

    filepath: str. indicates where the CSV for neo4j ingest should be written.
        Should be of the form 'path/to/file.csv'. If None, ``graph`` must not
        be None.
        
    graph: Neo4jConnection object. If not None, indicates that a Neo4j
        graph should be used as the place to save node data.
    

    Returns
    -------
    Nodes object representing unique authors.
    '''

    columns_of_interest = [
        'authorId',
        'name',
        'url',
        #'aliases'
    ]

    properties = pd.DataFrame([
        ['name', 'name', np.nan],
        ['url', 'semanticScholarURL', np.nan],
        #['aliases', 'aliases', 'string[]']
    ], columns=['old', 'new', 'type'])
    
    if graph is not None:
        properties['type'] = np.nan

    author_nodes = Nodes(
        parent_label='Person', 
        additional_labels=['Author'],
        data=get_unique_authors(df['authors'], 
            keys_to_return=columns_of_interest), 
        id_column='authorId', 
        reference='author', 
        properties=properties
    )

    if filepath is not None:
        author_nodes.export_to_csv(filepath)
    
    elif graph is not None:
        # Check that constraint exists and create it if not
        logger.debug("Creating authors constraint if it doesn't exist...")
        query = "CREATE CONSTRAINT people IF NOT EXISTS ON (a:Person) ASSERT a.id IS UNIQUE"
        graph.cypher_query_to_dataframe(query, verbose=False)
        
        logger.info("Saving author nodes data to Neo4j...")
        author_nodes.export_to_neo4j(graph, batch_size=batch_size)
        
    return author_nodes


def make_publication_nodes(
    df,
    filepath=None,
    graph=None,
    batch_size=2_000
):
    '''
    Given data wherein each record is a publication, extract the paper 
    metadata we need to make nodes in Neo4j, then save to CSV.


    Parameters
    ----------
    df: pandas DataFrame containing one record per publication.

    filepath: str. indicates where the CSV for neo4j ingest should be written.
        Should be of the form 'path/to/file.csv'. If None, ``graph`` must not
        be None.
        
    graph: Neo4jConnection object. If not None, indicates that a Neo4j
        graph should be used as the place to save node data.


    Returns
    -------
    Nodes object representing unique publications and their references.
    '''

    # Columns we need to treat especially carefully come ingest time
    text_columns = ['abstract', 'title_wos', 'fund_text']

    data = df.copy()
    for column in text_columns:
        data[column] = data[column].str.replace(r'\n', '', regex=True).str.replace(r'\\n', '', regex=True)
        
    properties = pd.DataFrame(
        [
            ['date', 'publicationDate', 'datetime'],
            ['DOI', 'doi', np.nan],
            ['author_names', 'authorNames', 'string[]'], # Have to keep for the authors we couldn't link from SS
            ['cat_subject', 'categoriesWOS', 'string[]'],
            ['source', 'publicationName', np.nan],
            ['pubtype', 'publicationTypes', 'string[]'], # This is a list and may as well stay that way
            ['title_wos', 'title', np.nan],
            ['fund_text', 'fundingText', np.nan],
            ['page_count', 'pageCount', 'int'],
            ['abstract', 'abstract', np.nan],
            ['abstract_source', 'abstract_source', np.nan],
            ['abstract_missing', 'abstract_missing', 'string[]'],
            ['doctype', 'publicationDocumentTypes', 'string[]'],
            ['grant_id', 'fundingIDs', 'string[]'],
            ['fieldsOfStudy', 'categoriesSS', 'string[]'],
            ['isOpenAccess', 'openAccess', 'boolean'],
            ['id_ss', 'semanticScholarID', np.nan],
            ['url_ss', 'semanticScholarURL', np.nan],
            ['embedding.model', 'embeddingModel', np.nan],
            ['embedding.vector', 'embedding', 'float[]']
           # ['topic_names', 'topics', 'string[]']
        ],
        columns=['old', 'new', 'type']
    )
        
    # Ignore neo4j data type strings
    if graph is not None:
        properties['type'] = np.nan
    

    publication_nodes = Nodes(
        parent_label='Publication',
        data=data, 
        id_column='id_wos', 
        reference='paper', 
        properties=properties
    )
    
    # Pull in all the references too, to get a more dense network
    ref_ids = data['ref_id'].explode(ignore_index=True).drop_duplicates().dropna().values
    df_refs = pd.DataFrame({
        'id_wos': ref_ids
    })
    
    reference_nodes = Nodes(
        parent_label='Publication', 
        data=df_refs, 
        id_column='id_wos', 
        reference='paper', 
        properties=None
    )

    if filepath is not None:
        publication_nodes.export_to_csv(filepath)
        
    elif graph is not None:
        # Check that constraint exists and create it if not
        logger.info("Creating publications constraint if it doesn't exist...")
        query = "CREATE CONSTRAINT publications IF NOT EXISTS ON (p:Publication) ASSERT p.id IS UNIQUE"
        graph.cypher_query_to_dataframe(query, verbose=False)
        
        logger.info("Saving publication nodes data to Neo4j...")
        publication_nodes.export_to_neo4j(graph, batch_size=batch_size)
        
        logger.info("Saving paper reference nodes to Neo4j...")
        reference_nodes.export_to_neo4j(graph, batch_size=batch_size)
        
    # Join together the papers and references data so we have 
    # everything needed for edge creation
    df_refs = pd.DataFrame(
        np.full((ref_ids.shape[0], data.shape[1]), np.nan),
        columns=data.columns
    )
    df_refs['id_wos'] = ref_ids
    data = data.append(df_refs, ignore_index=True)
    
    output = Nodes(
        parent_label='Publication', 
        data=data, 
        id_column='id_wos', 
        reference='paper', 
        properties=properties
    )
        
    return output

def geocode_institutions(df, geocoder='open_street_map'):
    '''
    Using location data about institutions, query a geocoder service
    for latitude and longitude coordinates that can be used to place
    the institutions on a map.

    Note that currently this simply extracts the first location
    available for each institution. It does not geolocate every
    location associated with each institution.


    Parameters
    ----------
    df: pandas DataFrame containing information about
        each unique institution to be geocoded.

    geocoder: str. Can be one of ['open_street_map']. Indicates
        which geocoding service to use.


    Returns
    -------
    A pandas DataFrame with latitudes and longitudes and an index 
    that is at least a subset of that used in ``df``.
    '''

    #TODO: setup to geocode every location of an institution, presumably by creating Location nodes
    geo_columns = {
        'street': 'addresses:string[]',
        'city': 'cities:string[]',
        'state': 'states:string[]',
        'postalcode': 'postalCodes:string[]',
        'country': 'countries:string[]'
    }

    columns_that_cannot_be_null = [
        geo_columns['city'], 
        geo_columns['country']
    ]

    # Make sure we only bother with the ones that have enough address data!
    logger.info(f"Ignoring institutions that have null city and/or country location data...")
    output = df.dropna(subset=columns_that_cannot_be_null)

    # Extract the first location's data only
    for column in geo_columns.values():
        if column in output.columns:
            output[column] = output[column].str.get(0)

    if geocoder == 'open_street_map':
        output_geocoded = get_from_openstreetmap(output, **geo_columns)

    else:
        raise ValueError(f"'{geocoder}' is not a recognized value for ``geocoder``")
    
    num_geocoded = output_geocoded['address'].notnull().sum()
    pct_geocoded = round(num_geocoded / len(output) * 100, 2)
    logger.info(f"Successfully geocoded {num_geocoded} ({pct_geocoded}%) of the locations with non-null location data.")

    geocode_columns = ['address', 'latitude', 'longitude']
    return output_geocoded[geocode_columns].copy()


def make_institution_nodes(
    df, 
    filepath=None, 
    graph=None, 
    geocode=False,
    batch_size=2_000
):
    '''
    Given data wherein each record is a publication, extract the institutional 
    affiliation metadata we need to make nodes in Neo4j, then save to CSV.


    Parameters
    ----------
    df: pandas DataFrame containing one record per publication.

    filepath: str. indicates where the CSV for neo4j ingest should be written.
        Should be of the form 'path/to/file.csv'. If None, ``graph`` must not
        be None.
        
    graph: Neo4jConnection object. If not None, indicates that a Neo4j
        graph should be used as the place to save node data.

    geocode: bool. Indicates if unique institutions should have their address
        data used to geocode them, producing latitude/longitude coordinates for
        each whenever possible. Note that this currently uses the first availabe
        set of address data for an institution, if more than one was found in 
        the data.


    Returns
    -------
    Nodes object representing unique institutions/funders.
    '''
    #TODO: find a way to get a more consistent and useful unique ID per institution
        # Probably via DUNS number or some such
    # Get base data, including de-duplication
    df_funders = get_unique_funders(df['funding'])
    df_institutions = get_unique_institutions(df['authors'])
    
    # Merge together
    # Keeping first and pushing in institutions first makes sure 
    # we keep max metadata (e.g. addresses)
    data = pd.concat([df_institutions, df_funders], ignore_index=True)\
        .drop_duplicates(subset='id', keep='first')
    

    # Pull out the extra metadata from df_institutions into the funders df
    df_funders = data.merge(
        df_funders,
        on='id', 
        how='inner',
        suffixes=('_keep', '_drop')
        ).rename(columns={'name_keep': 'name'}).drop(columns=['name_drop'])
    
    # Make sure institutions df has no funders double-counted
    funders_to_drop_ids = df_institutions.merge(
        df_funders,
        on='id', 
        how='inner',
        suffixes=('_keep', '_drop')
        )['id'].tolist()
    
    df_institutions = df_institutions[~df_institutions['id'].isin(funders_to_drop_ids)]
    
    # ----------------------------------------------------------------------------------
    # FUNDERS
    properties = pd.DataFrame([
        ['name', 'name', np.nan]
    ], columns=['old', 'new', 'type'])
    
    if graph is not None and properties is not None:
        properties['type'] = np.nan

    funder_nodes = Nodes(
        parent_label='Institution',
        additional_labels=['Funder'], 
        data=df_funders, 
        id_column='id',
        reference='org', 
        properties=properties
    )
    
    # ----------------------------------------------------------------------------------
    # INSTITUTIONS
    
    # Get researcher affiliation info
    properties = pd.DataFrame([
        ['name', 'name', np.nan],
        ['address', 'addresses', 'string[]'],
        ['city', 'cities', 'string[]'],
        ['state', 'states', 'string[]'],
        ['postal_code', 'postalCodes', 'string[]'],
        ['country', 'countries', 'string[]']
    ], columns=['old', 'new', 'type'])
    
    if graph is not None:
        properties['type'] = np.nan

    institution_nodes = Nodes(
        parent_label='Institution', 
        data=df_institutions, 
        id_column='id',
        reference='org', 
        properties=properties
    )
    
    # ----------------------------------------------------------------------------------
    # ALL ORGS
    # Must have them together for now, as Pipeline expects only one output
    
    output = Nodes(
        parent_label='Institution', 
        data=data, 
        id_column='id',
        reference='org', 
        properties=properties
    )


    if geocode:
        output_geocoded = geocode_institutions(output.data, geocoder='open_street_map')
        output.data.loc[output_geocoded.index, output_geocoded.columns] = \
            output_geocoded#.astype(float) # For some reason, they end up being object dtype...

        # Remove address column as it's mostly useful only for debugging
        # Rename columns to identify their data types properly to neo4j import
        lat_long_column_mapping = {
            'latitude': 'latitude:float',
            'longitude': 'longitude:float'
        }
        output.data = output.data.drop(columns=['address']).rename(columns=lat_long_column_mapping)

        # Add lat and long to Nodes object properties
        output.properties.extend(output_geocoded.columns.tolist())

    # Export results
    if filepath is not None:
        output.export_to_csv(filepath)
        
    elif graph is not None:
        # Check that constraint exists and create it if not
        logger.info("Creating institutions constraint if it doesn't exist...")
        queries = [
            'CREATE CONSTRAINT institution_ids IF NOT EXISTS ON (i:Institution) ASSERT (i.id) IS UNIQUE',
            #'CREATE CONSTRAINT institution_names IF NOT EXISTS ON (i:Institution) ASSERT (i.name) IS UNIQUE'
        ]
        [graph.cypher_query_to_dataframe(query, verbose=False) for query in queries]
        
        logger.info("Saving Funder nodes data to Neo4j...")
        funder_nodes.export_to_neo4j(graph, batch_size=batch_size)
        
        logger.info("Saving Institution-only nodes data to Neo4j...")
        institution_nodes.export_to_neo4j(graph, batch_size=batch_size)

    return output


def make_citation_relationships(
    df, 
    paper_nodes, 
    filepath=None,
    graph=None,
    batch_size=2_000
):
    '''
    Derives the relationship between a publication and all the other 
    publications it references and saves the results in a format that Neo4j
    can ingest.


    Parameters
    ----------
    df: pandas DataFrame containing one record per reference/citation.

    paper_nodes: Node object that must be provided for various data consistency
        checks.

    filepath: str. indicates where the CSV for neo4j ingest should be written.
        Should be of the form 'path/to/file.csv'. If None, ``graph`` must not
        be None.
        
    graph: Neo4jConnection object. If not None, indicates that a Neo4j
        graph should be used as the place to save relationship data.


    Returns
    -------
    Relationships object for each unique citation.
    '''

    references = df[['id_wos', 'ref_id', 'date']].explode('ref_id')\
        .dropna(subset=['id_wos', 'ref_id']).drop_duplicates()

    properties = pd.DataFrame(
        [
            ['date', 'publicationDate', 'datetime']
        ],
        columns=['old', 'new', 'type']
    )
    
    if graph is not None:
        properties['type'] = np.nan

    output = Relationships(
        type='CITED_BY',
        id_column_start='ref_id',
        id_column_end='id_wos',
        data=references,
        start_node=paper_nodes,
        end_node=paper_nodes,
        allow_unknown_nodes=False, # This may not matter since we write all Pub nodes now from refs
        properties=properties
    )

    if filepath is not None:
        logger.info(f"Saving {len(output)} relationships to disk...")
        output.export_to_csv(filepath)
        
    elif graph is not None:
        #TODO: if we ever have Enterprise Neo4j, put constraint that CITED_BY requires publicationDate property
        logger.info(f"Saving {len(output)} citation relationships to Neo4j...")
        output.export_to_neo4j(graph, batch_size=batch_size)

    return output


def make_authorship_relationships(
    df,
    author_nodes,
    paper_nodes,
    filepath=None,
    graph=None,
    batch_size=2_000
    ):
    '''
    Derives the relationship between an author of a publication and the 
    publication they wrote. Also saves the results in a format that Neo4j
    can ingest.


    Parameters
    ----------
    df: pandas DataFrame containing one record per author-publication 
        connection.

    author_nodes: Nodes object referring to the unique authors being ingested.

    paper_nodes: Nodes object referring to the publications being ingested.

    filepath: str. indicates where the CSV for neo4j ingest should be written.
        Should be of the form 'path/to/file.csv'. If None, ``graph`` must not
        be None.
        
    graph: Neo4jConnection object. If not None, indicates that a Neo4j
        graph should be used as the place to save relationship data.


    Returns
    -------
    Relationships object with each unique authorship.
    '''

    # Columns with data we don't need to retain for the authorship connections
    # Mostly used for earlier ETL steps or node creation
    extraneous_columns = [
        'institutions',
        'name', 
        'url',
        'name_wos',
        'last_name'
    ]

    data = expand_dict_lists(
            df['authors'],
            keep_index=True,
            drop_duplicates=False
        ).drop(columns=extraneous_columns).dropna(subset=['authorId'])

    data = data.join(df[['id_wos', 'date']], how='inner')

    properties = pd.DataFrame(
        [
            ['date', 'publicationDate', 'datetime'],
            ['authorListPosition', 'authorListPosition', 'int'],
            ['authorListPositionReversed', 'authorListPositionReversed', 
                'int'],
            ['authorListAlphabetized', 'authorListAlphabetized', 'boolean'],
            ['authorListAlphabetizedLastName', 
                'authorListAlphabetizedLastName', 'boolean']
        ],
        columns=['old', 'new', 'type']
    )
    
    if graph is not None:
        properties['type'] = np.nan
        
    output = Relationships(
        type='WROTE',
        data=data,
        start_node=author_nodes,
        id_column_start='authorId',
        id_column_end='id_wos',
        end_node=paper_nodes,
        allow_unknown_nodes=False,
        properties=properties
    )

    if filepath is not None:
        logger.info(f"Saving {len(output)} relationships to disk...")
        output.export_to_csv(filepath)
        
    elif graph is not None:
        logger.info(f"Saving {len(output)} authorship relationships to Neo4j...")
        output.export_to_neo4j(graph, batch_size=batch_size)

    return output


def extract_author_institution_relationships(series, institutions):
    '''
    Creates mapping of authors to institutions with which they are affiliated
    through employment (or, more accurately, at least not only funding).


    Parameters
    ----------
    series: pandas Series of authorship information. Each record should have
        a list of dicts, with each dict containing information about a single
        paper's author, including a nested list of dicts for institutional
        affiliation information (one dict per unique affiliation).

        So many dictionaries!

    institutions: a Nodes object representing unique institutions.


    Returns
    -------
    pandas DataFrame with institutional IDs mapped to author IDs (two columns),
    in addition .
    '''

    # Extract institution information from each paper's authorship data
    insts = expand_dict_lists(
            series,
            keep_index=True,
            drop_duplicates=False
        ).dropna(subset=['authorId'])[['authorId', 'institutions']]\
        .explode('institutions').dropna()

    # Map authors to insitutions
    output = pd.DataFrame(insts['institutions'].tolist(), 
        index=insts.index).join(insts['authorId'], how='inner')

    # Merge unique institutions data (specifically the ID) with the 
    # author-institution mapping, using the institution name as the join key
    output = output.merge(
        institutions.data[['name', f'id:ID({institutions.reference}-ref)']],
        how='inner',
        left_on='institution',
        right_on='name'
        )[['authorId', institutions.id]]\
        .dropna().drop_duplicates()

    return output


def make_affiliations(
    df,
    author_nodes,
    institution_nodes,
    filepath=None,
    graph=None,
    batch_size=2_000
    ):
    '''
    Derives the relationship between an author of a publication and the 
    publication they wrote. Also saves the results in a format that Neo4j
    can ingest.


    Parameters
    ----------
    df: pandas DataFrame containing one record per author-institution (employer) 
        connection.

    author_nodes: Nodes object referring to the unique authors being ingested.

    institution_nodes: Nodes object referring to the organizations being 
        ingested.

    filepath: str. indicates where the CSV for neo4j ingest should be written.
        Should be of the form 'path/to/file.csv'. If None, ``graph`` must not
        be None.
        
    graph: Neo4jConnection object. If not None, indicates that a Neo4j
        graph should be used as the place to save relationship data.


    Returns
    -------
    Relationships object with each unique affiliation.
    '''

    #FIXME: this can probably be revamped now that we can generate Funder nodes separate from non-Funder Institution nodes, just need two inputs
    data = extract_author_institution_relationships(
            df['authors'],
            institution_nodes
            )

    properties = None

    output = Relationships(
        type='AFFILIATED_WITH',
        data=data,
        start_node=author_nodes,
        id_column_start='authorId',
        end_node=institution_nodes,
        allow_unknown_nodes=False,
        properties=properties
    )

    if filepath is not None:
        logger.info(f"Saving {len(output)} {output.type} relationships to disk...")
        output.export_to_csv(filepath)
        
    if graph is not None:
        logger.info(f"Saving {len(output)} {output.type} relationships to Neo4j...")
        output.export_to_neo4j(graph, batch_size=batch_size)

    return output


def make_funding_relationships(
    df,
    paper_nodes,
    institution_nodes,
    filepath=None,
    graph=None,
    batch_size=2_000
    ):
    '''
    Derives the relationship between an author of a publication and the 
    publication they wrote. Also saves the results in a format that Neo4j
    can ingest.


    Parameters
    ----------
    df: pandas DataFrame containing one record per paper-funder (org) 
        connection.

    paper_nodes: Nodes object referring to the unique papers being ingested.

    institution_nodes: Nodes object referring to the fundign organizations 
        being ingested.

    filepath: str. indicates where the CSV for neo4j ingest should be written.
        Should be of the form 'path/to/file.csv'. If None, ``graph`` must not
        be None.
        
    graph: Neo4jConnection object. If not None, indicates that a Neo4j
        graph should be used as the place to save relationship data.


    Returns
    -------
    Relationships object with each unique affiliation.
    '''

    grants = expand_dict_lists(
        df['funding'], 
        keep_index=True, 
        drop_duplicates=False
        ).replace({'': np.nan}).dropna(subset=['funder'])

    # Add paper IDs and dates of publication/funding
    grants = grants.join(df[['id_wos', 'date']], how='left')

    data = grants.merge(
        institution_nodes.data,
        how='inner',
        left_on='funder',
        right_on='name'
    )[['grantId', 'funder', institution_nodes.id, 'id_wos', 'date']]

    data.rename(columns={'id_wos': paper_nodes.id}, inplace=True)

    properties = pd.DataFrame(
        [
            ['date', 'date', 'datetime'],
            ['grantId', 'grantId', np.nan]
        ],
        columns=['old', 'new', 'type']
    )
    
    if graph is not None:
        properties['type'] = np.nan

    output = Relationships(
        type='FUNDED',
        data=data,
        start_node=institution_nodes,
        end_node=paper_nodes,
        allow_unknown_nodes=False,
        properties=properties
    )

    if filepath is not None:
        logger.info(f"Saving {len(output)} {output.type} relationships to disk...")
        output.export_to_csv(filepath)
        
    if graph is not None:
        logger.info(f"Saving {len(output)} {output.type} relationships to Neo4j...")
        output.export_to_neo4j(graph, batch_size=batch_size)

    return output

def make_neo4j_ingest_command(
        nodes=None, 
        nodes_stages=None, 
        relationships=None,
        relationships_stages=None
    ):
    '''
    Using generated Nodes and Relationship objects, creates the command
    string needed to load the data into Neo4j via neo4j-admin import.


    Parameters
    ----------
    nodes: iterable of Nodes objects you want included in the ingest.

    nodes_stages: iterable of Stage objects whose output is one Nodes object 
        per Stage. If ``nodes`` is not None, this will be ignored.

    relationships: iterable of Relationships objects you want included in the
        ingest.

    relationships_stages: iterable of Stage objects whose output is one 
        Relationships object per Stage. If ``relationships`` is not None, 
        this will be ignored.


    Returns
    -------
    A string that contains the command needed to execute the data ingest.
    '''

    if nodes is None:
        nodes_filepaths = \
        [stage.get_results().filepath for stage in nodes_stages]

    else:
        nodes_filepaths = [node.filepath for node in nodes]

    # Extract only filenames
    nodes_filepaths = pd.Series(nodes_filepaths).str.split('/', expand=True).iloc[:, -1].tolist()


    if relationships is None:
        relationships_filepaths = \
        [stage.get_results().filepath for stage in relationships_stages]

    else:
        relationships_filepaths = [rel.filepath for rel in relationships]

    # Extract only filenames
    relationships_filepaths = pd.Series(relationships_filepaths).str.split('/', expand=True).iloc[:, -1].tolist()

    return generate_ingest_code(
        nodes_filepaths, 
        relationships_filepaths
    )



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('df_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
