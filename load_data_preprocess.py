import numpy as np
import pandas as pd
import os
import string
import ast
import regex as re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from tqdm import tqdm






execfile("regexes_py.py")

# NLTK setup
nltk.download('wordnet')
nltk.download('stopwords')  # Download the stopwords dataset
lemmatizer = WordNetLemmatizer()


# Load English stopwords
stop_words = set(stopwords.words('english'))

#top line added on 20231226
#solution, suggest, initiation, nih, aki, stop, peg, ivf, ivbmbp, nrbc, came, young, might, swab, early, added, syndrome, closure, rather, previously, troponin, stop, sample, initial, location, tmax, syringe, covid,gait,especially,blast,esr,pcp,event,life,later,size,error,ivbmbp,unfortunately,essential,eval,spot,angio,clinically,slight,palpation,ivbmbp

input_string = """

solution, suggest, initiation, nih, aki, stop, peg, ivf, ivbmbp, nrbc, came, young, might, swab, early, added, syndrome, closure, rather, previously, troponin, stop, sample, initial, location, tmax, syringe,


provider,visit,impression,slightly,unable,related,drug,dyspnea,sinus,three,similar,first,already,pertinent,dysuria,ent,come,calcium,thank,sent,one,got,abdominal,reaction,ace,cardiovascular,cough,left,clinical,hypertension,diarrhea,loss,intravenous,mlhr,physical,meei,picc,june,diff,returned,resolved,temp,factor,creatinine,includes,rue,mgkg,intake,partner,crbc,hyperlipidemia,subsequent,
making,ivpb,lue,study,disposition,cva,tid,soon, 
favor, hent, muscle, pursue, social, motoprolol, unit, voice, inject, facilityadministered, called,sweat,admit,kidney,intermittent,sob,chief,fine,pregnancy,method,nerve,
albumin, alt, appetite, bleeding, bowel, case, chronic, close, cold, complete, considered, constipation, diagnosed, differential, difficulty, disorder, done, drop, electrolyte, emergency, episode, etiology, fatigue, final, fully, gdl, improved, injection, inpatient, insomnia, intact, line, management, mul, night, nightly, noticed, nurse, old, outpatient, palpitation, potassium, presenting, procedure, rare, screen, speech, spoke, surgical, suspected, twelve, uri, vision, vitals, weight, workup, yesterday,
arrival, atrial,transitioned, routine,tenderness,recommendation,associated,term,earlier,incontinence,crp,away,spent,room,do,joint,placed,except,multiple,small,psychiatricbehavioral,
motion,took,onset,rdw,issue,standing,do,asthma,condition,andor,red,allergen,renal,reflux,twice,setting,woman,urinary,continuous,think,kul,please,egfr,sore,florence,metoprolol,
comprehensive,end,severe,make,mpv,head,michael,dr,illness,secondary,describes,status,visual,do,disturbance,unspecified,ear,much,flp,include,remains,
strength,memory,rate,resp,site,stable,instructed,scheduled,lower,free,minute,needed,feasible,fluoride,feat,
side, recommended,vein,liver,mrn,benign,going,gastrointestinal,
ion,given,cal,axone,secondary,consistent,date, cal,abnormality,
osa,





    serum,concern, 




    also, and, as, cal, change, consistent, cus, cycles, d1, daily, dat, date, dated, day, days, december, deer, definitive, definitively, delayed, delays, demonstrat, due, for, have, his, in, ion, likely, no, not, of, on, or, puncture, right, such, to, was, with, would, wout, 
    assessment, disease, complaint, anxiety, persistent, seen, consider, month, past, possible, history, since, around, daily, leg, following, likely, evaluation, evaluated, mgdl, reviewed, symptom, sleep, pending, evidence, index, exam, lymph, diagnosis, musculoskeletal, given, last, admitted, concerning, due, present, new, otherwise, result, interval, dose, taking, without, testing, lab, treatment, medical, mild, system, report, therapy, note, time, gastroesophageal, chest, prior, transferred, treated, course, show, wife, continue,  though, male,  blood, continues, received, finding, consult, primary, back, fluid,  followup, started, intensity, recurrent, unclear, lung, return,  able, take,  help,  found, ankle, presented, recovery, htn,  uti,  presentation, primarily, reason, admission, intervention, bid,   consulted, flow,
    shortness, manual, hepatitis, slp, ache, frequency, duration


    mgh, discharged, md, started, possible, iv,
    fu, last, plt, panel, overall, followup, reason,
    taking, course, medication, cb, cf, continue,
    index, plan, datetime, dr, hpi, known, detected,
    antigen, allergy, mgdl, pt, well, hct,yo, female, 
    history,am,pm, able, active, activity, alb, alcohol,
    alkp, assessment, assessmentplan, bun, ca, car, cbc, cl, co, 
    coags, component, cre, data, dbili, degenerative, 
    diabetes, disc, discussed, disease, drive, estimated, eye, follow, 
    gfr, glu, hgb, hypercholesterolemia, input, inr, iso, lab, list, 
    mellitus, mg, ocular, patient, pdmp, problem, ptt, recent, refill, 
    result, reviewed, sonopalpation, sp, tablet, tbili, thickening, 
    thoracic, tp, tsh, type, ucre, value, vitamin, walk, walker, 
    within, respiratory, amlodipine, speak, lfts, morning, bundle, compliant, foot, acceptable,
    nasal, heart,rehab,increase,regarding,stopped,underwent,surgery,march,however,yet,gtt,bolus,felt,home,patient,tylenol,oral,still, chloride,glycol,arthroscopy,

    
    still, see, due, additional, seen, notable, appointment, use, including, order, like, need, person, would, none, otherwise, matter, agree, tried,
    also, could, today, current, new, past, may, take, home, appears, ago, given, better, good, change, return, detail, question, care, starting, family, since, 
    followed, previous, back, presentation, general, primarily, based, ref, saw, thought, raising, performed, without, daily, month, week, still, see, due, 
    additional, seen, notable, appointment, use, including, order, like, need, person, would, none, otherwise, matter, agree, tried,
    obtain, date, high, feeling, day, per, denies, note, symptom, using, currently, name, lab, prn, continues, sign, however, state, went, patient, 
    basic, am, finding, son, code, also, could, today, current, new, past, may, take, home, appears, ago, given, better, good, change, return, detail, 
    question, care, starting, family, since, followed, previous, back, presentation, general, primarily, based, ref, saw, thought, raising,htn,
    get, right, low, file, work, clear, without, note, initially, range, male, year, several, full, feel, check, control, department, completed, 
    recommend, ongoing, obtained, level, though, service, state, regarding, about, obtain, date, high, feeling, day, per, denies, symptom, using, 
    currently, name, lab, prn, continues, sign, however, state, went, patient, basic, am, finding, son, code, also, could, today, current, new, past,
    may, take, home, appears, ago, given, better, good, change, return, detail, question, care, starting, family, since, followed, previous, back, 
    presentation, general, primarily, based, ref, saw, thought, raising,
    total, skin, showed, placement, unknown, every, mood, treatment, month, system, discus, baseline, start, record, medication, day, 
     repeat, sig, developed, blood, mouth, med, le, chloride, musculoskeletal, transfer, murmur, presented, urine, noted, source, cause, 
    two, medical, pressure, respiratory, fall, hospital, question, comment, center, pneumonia, sodium, rbc, reported, 
    encounter, note, allergy, outside, risk, glucose, cancer, review, infusion, pulse, discharge, affected, 
    dose, limited, lisinopril, improving, anxiety, clinic, consulted, regimen, consult, april, oral, extremity, 
    transferred, involvement, test, continued, senna, service, state, subjective, hour, bwh, rehab, pending, mdm, found, capsule, osh, felt, 
    patient, weekly, attending, october, bruising, dispense, finding, primary,
    also, bwh, continues, current, future, given, home, htn, may, note, ordered, past, past surgical, post, rehab, take, time, today, transferred
    """

# Remove all whitespaces from the input string
input_string = input_string.replace(" ", "")


# Remove leading and trailing whitespace and split the string into lines
lines = input_string.strip().split('\n')

# Initialize an empty list to store the words
words_to_add = []

# Iterate through the lines, ignoring commented lines
for line in lines:
    line = line.strip()  # Remove leading and trailing whitespace
    if not line.startswith("#"):  # Check if the line is not a comment
        # Split the line at commas, remove extra whitespace, and extend the words list
        words_to_add.extend([word.strip() for word in line.split(',')])

# Split the content at commas and remove empty words
words_to_add = [word for word in words_to_add if word]


print(words_to_add)

#words_to_add=''



# Extend the set of stopwords
stop_words.update(words_to_add)

# Words to remove from stop_words
words_to_remove = ["no", "without"]

# Remove the words from stopwords
for word in words_to_remove:
    stop_words.discard(word)


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-ASCII characters
    text = ''.join(char for char in text if ord(char) < 128)
    
    # Remove consecutive whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove numbers
    #text = re.sub(r'\d+', 'd', text)
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)  
    #text = str(TextBlob(text).correct())

    # Lemmatize, remove stopwords, and filter out words of less than 2 characters
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and (len(word) > 2 or word=='lp')]
    lemmatized_words = [word for word in lemmatized_words if word not in stop_words and (len(word) > 2 or word=='lp')]
    processed_text = ' '.join(lemmatized_words).strip()
    
    # Replace empty or whitespace-only text with a single whitespace
    if not processed_text:
        processed_text = ' '
    
    return processed_text

#all_regexes = new_nidx_pos+new_nidx_neg+new_drugs_nidx+new_extra_marks+new_neck_pos+new_neck_neg
all_regexes = new_nidx_pos+new_nidx_neg+new_drugs_nidx+new_neck_pos+new_neck_neg
#all_regexes = new_nidx_pos+new_nidx_neg+new_neck_pos+new_neck_neg
#all_regexes = new_nidx_pos+new_nidx_neg+new_drugs_nidx+new_neck_pos+new_neck_neg+new_extra_marks

def extract_highlighted_with_buffer(text, patterns, buffer=100):
    indices = []
    text_length = len(text)
    i=0
    for pattern in patterns:
        # if i >1868:
        #     buffer=10
        for match in re.finditer(pattern, text):
            start, end = match.span()

            # Apply buffer, ensuring boundaries of the text
            start_buffered = max(0, start - buffer)
            end_buffered = min(text_length, end + buffer)

            # Adjust the buffered region to include whole words only
            while start_buffered > 0 and text[start_buffered] not in (' ', '\n', '\t'):
                start_buffered -= 1
            while end_buffered < text_length and text[end_buffered] not in (' ', '\n', '\t'):
                end_buffered += 1

            indices.append((start_buffered, end_buffered))

    if not indices:  # Check if indices list is empty
        return ''

    # Sort and merge overlapping or adjacent indices
    indices.sort(key=lambda x: x[0])
    merged = [indices[0]]
    for current in indices[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    # Extract and join highlighted text
    return ' '.join(text[start:end] for start, end in merged)
    #return ' lalalalalalala '.join(text[start:end] for start, end in merged)



def match_regexes(df, note_column, buffer=100):
   """Transforms a DataFrame by extracting highlighted text with a buffer from a specified column and combining matches.

   Args:
       df (pd.DataFrame): The DataFrame to transform.
       note_column (str): The name of the column containing the text to process.
       buffer (int, optional): The buffer size for extracting text. Defaults to 100.

   Returns:
       pd.DataFrame: The transformed DataFrame with a new column 'NoteTXT_matched' containing the combined matches.
   """

   # Create a new column to store the combined matches
   df['NoteTXT_matched'] = ''
   
   for index, row in tqdm(df.iterrows(), total=len(df)):
       all_matches = []

       # Extract matches with buffer
       matches = extract_highlighted_with_buffer(row[note_column], all_regexes, buffer=buffer)  # Assuming 'all_regexes' is defined elsewhere
       all_matches.append(matches)

       # Combine matches into a single string
       combined_matches = ' '.join(all_matches)

       # Update the transformed column
       df.at[index, 'NoteTXT_matched'] = combined_matches

   return df
# Usage example
# match_regexes(df_test, 'note_column_name')

holdout_empi=[616, 2022, 204, 1197, 864, 986, 2681, 308, 376, 140, 17, 2298, 1073, 1210, 2513, 372, 1583, 2671, 2627, 1704, 1228, 708, 1578, 2520, 1678, 380, 2082, 2462, 692, 1189, 458, 2012, 37, 1334, 1960, 2760, 627, 220, 433, 2745, 1629, 2479, 1832, 260, 1131, 2209, 2903, 2722, 464, 784, 1455, 2864, 1159, 1310, 1361, 723, 2414, 1536, 1944, 382, 2013, 1735, 80, 2385, 1137, 2553, 314, 2337, 446, 1500, 2156, 1166, 1161, 2668, 1754, 2875, 484, 998, 2600, 1992, 1684, 2982, 619, 1660, 470, 7, 2684, 118, 2402, 2170, 2931, 104, 35, 172, 269, 937, 57, 1577, 2616, 2095, 1271, 2395, 1534, 1505, 1545, 2981, 1685, 942, 2210, 756, 2924, 882, 2391, 38, 903, 860, 233, 912, 503, 117, 2991, 997, 2849, 2539, 2639, 1572, 2228, 1056, 1528, 2847, 2661, 2449, 1429, 1215, 2989, 2499, 1440, 142, 1155, 2329, 444, 1858, 1974, 2060, 900, 1781, 2761, 206, 1643, 744, 1427, 481, 939, 2451, 1213, 1235, 698, 377, 1499, 1435, 649, 1565, 548, 1587, 1648, 1338, 606, 1965, 1365, 2718, 2826, 1886, 240, 2000, 198, 1321, 799, 1705, 597, 1177, 1679, 728, 322, 2792, 2635, 1099, 1573, 1, 800, 2379, 566, 102, 618, 2591, 2696, 505, 1855, 993, 1354, 1936, 1540, 2051, 2410, 497, 252, 1212, 1750, 2459, 2857, 1430, 943, 2025, 855, 564, 1029, 164, 389, 707, 2582, 681, 1683, 806, 2223, 1556, 1882, 2902, 1259, 2384, 2656, 1411, 948, 2409, 145, 569, 1637, 1086, 1108, 9, 1051, 329, 1575, 815, 1045, 584, 635, 2917, 2518, 512, 1413, 2174, 84, 2107, 2224, 1494, 2238, 2502, 1917, 2396, 2791, 2121, 712, 149, 753, 1217, 480, 2020, 748, 461, 572, 2867, 1938, 579, 1777, 2659, 1395, 2392, 284, 598, 1109, 979, 2842, 711, 901, 1620, 2088, 1766, 293, 889, 2763, 350, 2613, 667, 410, 1335, 2345, 2404, 2726, 277, 808, 1977, 2783, 449, 1993, 2768, 1356, 2397, 1502, 1285, 2839, 2287, 106, 666, 2047, 2665, 403, 2651, 1915, 246, 2704, 1453, 238, 2407, 1298, 457, 563, 1615, 1112, 2211, 1266, 632, 926, 304, 53, 2564, 300, 411, 1772, 2628, 1380, 1592, 2805, 202, 2943, 273, 2068, 1032, 2913, 2364, 1987, 935, 2342, 1206, 1299, 2322, 2776, 2126, 1686, 1495, 2889, 2077, 2061, 2841, 549, 2545, 690, 1141, 637, 1501, 1805, 303, 1005, 991, 2394, 2260, 1775, 1053, 1729, 683, 407, 1765, 1115, 1257, 1113, 2465, 2538, 1682, 1626, 1964, 2400, 1424, 934, 1147, 1538, 2891, 2837, 973, 724, 1672, 866, 504, 879, 2432, 1081, 970, 360, 1793, 2140, 1747, 1333, 1523, 1267, 1670, 1568, 550, 1895, 1970, 2309, 2923, 2807, 1815, 2173, 208, 910, 638, 1530, 2765, 2992, 846, 2963, 155, 490, 2368, 668, 2416, 1160, 66, 2927, 2856, 870, 721, 1527, 1748, 2789, 0, 2244, 1410, 1272, 1906, 1605, 2044, 736, 2509, 1098, 1902, 623, 2508, 2054, 2596, 722, 2624, 1433, 1168, 2525, 468, 152, 466, 709, 585, 92, 2485, 2653, 2560, 2277, 2965, 2964, 840, 2729, 2027, 2300, 2340, 2297, 762, 794, 1611, 384, 73, 1953, 415, 2926, 1876, 1616, 2725, 2652, 353, 2879, 85, 91, 363, 441, 1653, 554, 1208, 2547, 1633, 237, 2638, 641, 32, 787, 100, 1656, 258, 184, 813, 915, 648, 1532, 344, 231, 633, 1862, 2561, 341, 2016, 1963, 1720, 2983, 562, 2774, 2706, 2099, 436, 14, 1859, 547, 418, 1883, 191, 2733, 980, 2896, 2477, 1481, 2299, 440, 175, 2929, 1273, 2433, 2815, 2321, 364, 1606, 2006, 2810, 571, 1035, 2326, 839, 834, 1783, 132, 1138, 2487, 2281, 949, 2344, 2543, 158, 29, 1445, 1252, 1918, 540, 886, 740, 2994, 1186, 1767, 1595, 1402, 763, 1378, 1466, 639, 1269, 2438, 1332, 2944, 1110, 2527, 2089, 135, 1649, 994, 130, 10, 2164, 2686, 2712]

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def process_data(df, holdout_empi):
 """Processes data for text analysis, including splitting into training and testing sets,
    vectorizing text, and creating a DataFrame of features.

 Args:
   df: A pandas DataFrame containing the data to be processed.
   holdout_empi: A list of empirical values to be held out as a test set.

 Returns:
   A tuple containing:
     vectorizer: The fitted CountVectorizer object.
     X_train_transformed: The transformed training data.
     X_test_transformed: The transformed test data.
     features: The transformed features for the entire dataset.
     features_df: A DataFrame containing the features and target variable.
     features_bow: A list of the bag-of-words features.
 """

 # Copying the DataFrame
 data3 = df.copy()

 # Creating training and testing sets
 print("Creating training and testing sets...")
 test_mask3 = data3['current_empi'].isin(holdout_empi)
 X_test = data3[test_mask3]['NoteTXT_matched']
 y_test = data3[test_mask3]['val']
 X_train = data3[~test_mask3]['NoteTXT_matched']
 y_train = data3[~test_mask3]['val']
 print("Sets created.")

 # Initialize and fit the CountVectorizer
 print("Initializing and fitting CountVectorizer...")
 vectorizer = CountVectorizer(ngram_range=(1, 3))
 X_train_transformed = vectorizer.fit_transform(X_train)

 # Transform X_test using the fitted vectorizer
 X_test_transformed = vectorizer.transform(X_test)

 # Transform the entire dataset
 print("Transforming the entire dataset...")
 features = vectorizer.transform(data3['NoteTXT_matched'])

 # Create a DataFrame
 features_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())
 features_df['val'] = data3['val'].values

 # Get the bag-of-words features (excluding 'val')
 print("Extracting bag-of-words features...")
 features_bow = list(features_df.columns.difference(['val']))

 print("Data processing complete.")

 return vectorizer, X_train_transformed, y_train, X_test_transformed, y_test, features, features_df, features_bow
