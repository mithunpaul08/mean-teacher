from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import logging
import json
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Predictor.register('drwiki-te')

class PaperClassifierPredictor(Predictor):

    """"Predictor wrapper for the AcademicPaperClassifier"""
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        if len(inputs["claim"])<1:
            # labels_list=["agree", "diagree", "discuss", "unrelated"]
            #the rf is the dev/test file that we want to read records from and predict
            # test_file='my_library/predictors/fever_dev_mithun_converted.jsonl'

            test_file='my_library/predictors/fnc_test_files/fnc_test_smartner.jsonl'
            predicitons_file= 'cross_domain_fnc.jsonl'
            with open(test_file) as rf, open(predicitons_file,'w+') as wf:
                for line in rf:
                    current_line = json.loads(line)
                    # logger.info(current_line)
                    inputs['claim'] = current_line['claim']
                    inputs['evidence'] = current_line['evidence']
                    # instance = self._json_to_instance(inputs)
                    instance, tokenized_claim, tokenized_evidence = self._json_to_instance(inputs)
                    tokenized_claim_list = [str(each) for each in tokenized_claim]
                    tokenized_evidence_list = [str(each) for each in tokenized_evidence]
                    label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
                    all_labels = [label_dict[i] for i in range(len(label_dict))]
                    output_dict = self.predict_instance(instance)
                    output_dict["all_labels"] = all_labels
                    max_lab = output_dict["label_probs"].index(max(output_dict["label_probs"]))
                    predicted_label = output_dict["all_labels"][max_lab]
                    evidence_totals = [sum(x) for x in zip(*output_dict["h2p_attention"])]
                    output_dict["evidence_cum_weights"] = evidence_totals
                    key_evidence_indexes = sorted(range(len(evidence_totals)),\
                                            key=lambda i: evidence_totals[i], reverse=True)[:3]
                    # logger.info(key_evidence_indexes)
                    key_evidence_words = [tokenized_evidence_list[each] for each in key_evidence_indexes]
                    output_dict["key_evidences"] = key_evidence_indexes
                    # logger.info(key_evidence_words)

                    output_dict["tokenized_claim"] = tokenized_claim_list
                    output_dict["tokenized_evidence"] = tokenized_evidence_list
                    output_dict["key_evidence_words"] = key_evidence_words
                    current_line['predicted_label'] = predicted_label.lower()
                    current_line['key_evidence_words']=key_evidence_words
                    current_line_new={}
                    current_line_new['predicted_label'] = predicted_label.lower()
                    current_line_new['key_evidence_words'] = key_evidence_words
                    wf.write(json.dumps(current_line) + '\n')
                    # if(current_line['label'].lower()) == predicted_label.lower():
                    #     logger.info("Found")
                    #     wf.write(json.dumps(current_line) + '\n')
        else:
            instance,tokenized_claim, tokenized_evidence = self._json_to_instance(inputs)
            tokenized_claim_list=[str (each) for each in tokenized_claim]
            tokenized_evidence_list=[str(each) for each in tokenized_evidence]
            label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
            all_labels = [label_dict[i] for i in range(len(label_dict))]
            output_dict = self.predict_instance(instance)
            output_dict["all_labels"] = all_labels
            max_lab = output_dict["label_probs"].index(max(output_dict["label_probs"]))
            predicted_label = output_dict["all_labels"][max_lab]
            #Sum of evidence weights- basically column weights of h2p array
            evidence_totals = [sum(x) for x in zip(*output_dict["h2p_attention"])]
            output_dict["evidence_cum_weights"] =evidence_totals
            key_evidence_indexes = sorted(range(len(evidence_totals)), key=lambda i: evidence_totals[i], reverse=True)[:3]
            logger.info(key_evidence_indexes)
            key_evidence_words=[tokenized_evidence_list[each] for each in key_evidence_indexes]
            output_dict["key_evidences"] = key_evidence_indexes
            logger.info(key_evidence_words)
            output_dict["tokenized_claim"] = tokenized_claim_list
            output_dict["tokenized_evidence"] = tokenized_evidence_list
            output_dict["key_evidence_words"] = key_evidence_words

        return output_dict


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        claim = json_dict['claim']
        evidence = json_dict['evidence']
        return self._dataset_reader.text_to_instance(evidence=evidence, claim=claim)

