# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase
from typing import Dict
import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.common.util import JsonDict
from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.models.model import Model
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from overrides import overrides

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
@Predictor.register('drwiki-te')
@DatasetReader.register("fever")
@Model.register("paper_classifier_2")
# @Model.register("paper_classifier")

class TestPaperClassifierPredictor(Predictor,Model):

    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 abstract_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 lazy: bool = False,
                 claim_tokenizer: Tokenizer = None,
                 sentence_level=False,
                 wiki_tokenizer: Tokenizer = None,

                 initializer: InitializerApplicator = InitializerApplicator(),
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        # super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("LABELS")
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.classifier_feedforward = classifier_feedforward
        initializer(self)

    def test_uses_named_inputs(self):
        inputs = {
            "claim": "Prison Break received a nomination for Best Television Series Drama at the 2005 Golden Globes .",
            "evidence": "Prison Break was nominated for several industry awards , including the 2005 Golden Globe Award for Best Television Series Drama and the 2006 People 's Choice Award for Favorite New TV Drama , which it won ."
        }
        archive = load_archive('tests/fixtures/decomposable_attention.tar.gz')
        instance = self._json_to_instance(inputs)
        # predictor = Predictor.from_archive(archive, 'drwiki-te')
        logger.info(instance)
        result = self.predict_instance(instance)
        label = result.get("label")
        # assert label in {'AGREE', 'ML', 'ACL'}
        all_labels = result.get("all_labels")
        print(all_labels)
        # assert all_labels == ['AI', 'ACL', 'ML']
        class_probabilities = result.get("class_probabilities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)
        assert sum(class_probabilities) == approx(1.0)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        logger.info(json_dict)
        # title = json_dict['title']
        claim = json_dict['claim']
        logger.info(json_dict)
        # abstract = json_dict['paperAbstract']
        evidence = json_dict['evidence']
        return self.text_to_instance(claim, evidence)

    def text_to_instance(self, claim: str, evidence: str, venue: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_title = self._tokenizer.tokenize(claim)
        tokenized_abstract = self._tokenizer.tokenize(evidence)
        title_field = TextField(tokenized_title, self._token_indexers)
        abstract_field = TextField(tokenized_abstract, self._token_indexers)
        fields = {'premise': title_field, 'hypothesis': abstract_field}
        if venue is not None:
            fields['label'] = LabelField(venue)
        return Instance(fields)

    @overrides
    def forward(self,  # type: ignore
                title: Dict[str, torch.LongTensor],
                abstract: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        title : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        abstract : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_title = self.text_field_embedder(title)
        title_mask = util.get_text_field_mask(title)
        encoded_title = self.title_encoder(embedded_title, title_mask)

        embedded_abstract = self.text_field_embedder(abstract)
        abstract_mask = util.get_text_field_mask(abstract)
        encoded_abstract = self.abstract_encoder(embedded_abstract, abstract_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_title, encoded_abstract], dim=-1))
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string LABELS, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="LABELS")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


inst= TestPaperClassifierPredictor()
inst.test_uses_named_inputs()