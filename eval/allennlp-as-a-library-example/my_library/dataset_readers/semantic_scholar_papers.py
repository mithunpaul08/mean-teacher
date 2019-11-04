from typing import Dict
import json
import logging
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("fever")
class FEVERReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the venue of the paper.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 lazy: bool = False,
                 claim_tokenizer: Tokenizer = None,
                 sentence_level=False,
                 wiki_tokenizer: Tokenizer = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                title = paper_json['claim']
                abstract = paper_json['abstract']
                venue = paper_json['venue']
                yield self.text_to_instance(title, abstract, venue)

    @overrides
    def text_to_instance(self, evidence: str, claim: str,\
                         venue: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_evidence = self._tokenizer.tokenize(evidence)
        tokenized_claim = self._tokenizer.tokenize(claim)
        evidence_field = TextField(tokenized_evidence, self._token_indexers)
        claim_field = TextField(tokenized_claim, self._token_indexers)
        # logger.info('evidence:: '+str(evidence_field))
        # logger.info('claim:: ' + str(claim_field))
        fields = {'premise': evidence_field, 'hypothesis': claim_field}
        if venue is not None:
            fields['label'] = LabelField(venue)
        return Instance(fields), tokenized_claim, tokenized_evidence
