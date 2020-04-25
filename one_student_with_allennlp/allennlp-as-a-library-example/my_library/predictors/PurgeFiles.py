#
# import json
# from collections import defaultdict, Counter
# # from overrides import overrides
# #
# # from allennlp.common.util import JsonDict
# # from allennlp.data import Instance
# # from allennlp.predictors.predictor import Predictor
# # import logging
# # import json
#
# new_dict=defaultdict(dict)
# wrong_claim_evidence=set()
# mismatch_list=[]
# home_list=[]
# # wrong_file= "../../23_Mar_Model-FN-13Feb-Data-fever-wrong-predictions.jsonl"
# wrong_file= "../../12_Apr_Model-FN-10-Apr-Data-fever-wrong-predictions.jsonl"
# with open(wrong_file, 'r+') as wrongfile:
#     # We need lines which are in both files not just one of them
#     # This intersection is limited to claim and evidence not the LABELS etc.
#     i=0
#
#     for w_line in wrongfile:
#         current_line = json.loads(w_line)
#         uniqueLine = current_line['claim'] + current_line['evidence']
#         wrong_claim_evidence.add(uniqueLine)
#         new_dict[uniqueLine]["wrong_evidence_words"]= current_line["key_evidence_words"]
#         new_dict[uniqueLine]["wrong_predicted_label"] = current_line["predicted_label"]
#
# # right_file= "../../23_Mar_Model-Fever-13Feb-Data-fever-right-predictions.jsonl"
# # home_right_alien_wrong_file= "../../23_Mar_Model-Fever-13Feb-fever-right-FN-wrong-predictions.jsonl"
# right_file= "../../12_Apr_Model-Fever-10-Apr-Data-fever-right-predictions.jsonl"
# home_right_alien_wrong_file= "../../13_Apr_Model-Fever-10-Apr-fever-right-FN-wrong-predictions.jsonl"
# with open(right_file, 'r+') as rightFile, \
#         open(home_right_alien_wrong_file, 'w+') as f1:
#     for r_line in rightFile:
#         current_line = json.loads(r_line)
#         r_uniqueLine = current_line['claim'] + current_line['evidence']
#         if r_uniqueLine in wrong_claim_evidence:
#             alien_file_evidence_words=list()
#             home_file_evidence_words=list()
#             # print(current_line["key_evidence_words"])
#             home_file_evidence_words.extend(current_line["key_evidence_words"])
#             alien_file_evidence_words.extend(new_dict[r_uniqueLine]["wrong_evidence_words"])
#             # print("Label-> "+ current_line["label"]+ "---> "+ current_line['claim'] + "<- claim | Evidence ->" + current_line["evidence"])
#             # print("Home Key Ev."+str(home_file_evidence_words) + " <-||-> " +"Alien Key Ev."+ str(alien_file_evidence_words))
#             key_evi_words_in_alien_wrong_but_not_in_home_right= set(alien_file_evidence_words) - set(home_file_evidence_words)
#             # print(key_evi_words_in_alien_wrong_but_not_in_home_right)
#             words_in_home_right_but_not_in_alien_wrong= set(home_file_evidence_words)-set(alien_file_evidence_words)
#             home_list.extend(home_file_evidence_words)
#             # if len(words_in_home_right_but_not_in_alien_wrong)<1:
#             #     print(home_file_evidence_words)
#             #     i+=1
#             #     print(i)
#             mismatch_list.extend(key_evi_words_in_alien_wrong_but_not_in_home_right)
#             # print(words_in_alien_wrong_but_not_in_home_right)
#             # current_line["key_wrong_FN_words"]
#             # f1.write(json.dumps(current_line) + '\n')
#
# # counter_dict= Counter(mismatch_list)
# print(mismatch_list)
# # print(counter_dict)
#
# # counter_dict= Counter(home_list)
# # print(counter_dict)
#
#
