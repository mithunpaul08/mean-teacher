#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm


def print_confusion_matrix(cm):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))


def report_score(actual,predicted):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)

    print_confusion_matrix(cm)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score

'''
FNC Fully Lex test: 68.99%
FNC Smart NER test= 65.85%
Fever Fully Lex test test= 41.47%
Fever smart NER test= 33.94%
'''
import json
# FNC_fully_lex_test= "2-May-Predictions/test_fever_Fully_Lex_NER_predictions_2_May.jsonl"
# FNC_fully_lex_test= "2-May-Predictions/test_fnc_smart_NER_predictions_2_May.jsonl"
# FNC_fully_lex_test= "fnc_predictions/dev_fnc_4-label-SS-Tags_predictions_12_May.jsonl"
# FNC_fully_lex_test= "2-May-Predictions/test_fever_smart_NER_predictions_2_May.jsonl"
# FNC_fully_lex_test= "fnc_predictions/fnc_fully lexicalized_19_April.jsonl" # 48.86
# FNC_fully_lex_test= "fnc_predictions/fnc_No_NER_predictions_19_April.jsonl" # 40.23
# FNC_fully_lex_test= "fnc_predictions/fnc_simple_NER_predictions_19_April.jsonl" # 46.27
# FNC_fully_lex_test= "fnc_predictions/fnc_smart_NER_Only_Embeddings_19_April.jsonl" # 46.27
eval_file= "cross_domain_fnc.jsonl" # 46.27
# eval_file= "fnc_predictions/TEST_fnc_4-label-ss_predictions_2_May_in_domain.jsonl" # 46.27
# FNC_fully_lex_test= "2-May-Predictions/test_fnc_Fully_Lex_NER_predictions_2_May.jsonl"
actual=[]
predicted=[]

with open(eval_file, 'r+') as wrongfile:
    i=0
    for w_line in wrongfile:
        current_line = json.loads(w_line)
        actual.append(current_line["label"].lower())
        predicted.append(current_line["predicted_label"].lower())

report_score(actual,predicted)
    # report_score([LABELS[e] for e in actual],[LABELS[e] for e in predicted])