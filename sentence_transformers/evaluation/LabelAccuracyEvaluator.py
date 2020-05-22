from . import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from ..util import batch_to_device
import os
import csv
from student_teacher.mean_teacher.scorers.fnc_scorer import report_score
from sentence_transformers.util import calculate_micro_f1
from sentence_transformers.readers import NLIDataReader

class LabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model = None,grapher=None,logger=None,nlireader=None):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.softmax_model = softmax_model
        self.softmax_model.to(self.device)
        self.draw_graphs=grapher
        self.logging=logger
        self.nlireader=nlireader

        if name:
            name = "_"+name

        self.csv_file = "accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        self.logging.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        total_gold=[]
        total_predictions=[]
        for step, batch in enumerate(tqdm(self.dataloader, desc="dev batches")):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                prediction = self.softmax_model(features,labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()

            predictions_classes=torch.argmax(prediction, dim=1)
            total_gold.extend(label_ids)
            total_predictions.extend(predictions_classes)

        accuracy = correct / total
        accuracy_graph_name=self.name+" accuracy per epoch"
        self.draw_graphs.log_metric(accuracy_graph_name,accuracy*100 ,
                                    step=epoch,include_context=False)

        self.logging.info("Accuracy on : {:.4f} ({}/{}) correct\n".format(accuracy, correct, total))

        fnc_score = 0.00

        if ("fnc" in self.name):
            predictions_str_labels = NLIDataReader.get_label_given_index(NLIDataReader, total_predictions)
            gold_str_labels = NLIDataReader.get_label_given_index(NLIDataReader, total_gold)
            fnc_score = report_score(gold_str_labels, predictions_str_labels)
            fnc_score_graph_name = self.name + " fnc_score per epoch"
            self.draw_graphs.log_metric(fnc_score_graph_name, fnc_score,
                                    step=epoch, include_context=False)
            self.logging.info("FNC score on : {:.4f} \n".format(fnc_score))

        microf1_without_unrelated_class = calculate_micro_f1(total_predictions, total_gold, 3)
        microf1_with_only_unrelated_class = calculate_micro_f1(total_predictions, total_gold, 3, True)
        self.logging.info(f"microf1_without_unrelated_class:{microf1_without_unrelated_class}")
        self.logging.info(f"microf1_with_only_unrelated_class:{microf1_with_only_unrelated_class}")

        mf1_without_unrelated_graph_name = self.name + " microf1_without_unrelated"
        self.draw_graphs.log_metric(mf1_without_unrelated_graph_name, microf1_without_unrelated_class,
                                    step=epoch, include_context=False)

        mf1_only_unrelated_graph_name = self.name + " microf1_only_unrelated"
        self.draw_graphs.log_metric(mf1_only_unrelated_graph_name, microf1_with_only_unrelated_class,
                                    step=epoch, include_context=False)

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])

        return accuracy