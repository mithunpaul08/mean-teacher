import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import io
from . import data
from .utils import export

from .processNLPdata.processNECdata import *
import os
import contextlib

words_in_glove =0

@export
def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/ilsvrc2012/',
        'num_classes': 1000
    }


@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }

@export
def ontonotes():

    if NECDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(NECDataset.NUM_WORDS_TO_REPLACE, NECDataset.OOV, NECDataset.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(NECDataset.WORD_NOISE_TYPE)

    return {
        'train_transformation': data.TransformTwiceNEC(addNoise),
        'eval_transformation': None,
        'datadir': 'data-local/nec/ontonotes',
        'num_classes': 11
    }



@export
def conll():

    if NECDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(NECDataset.NUM_WORDS_TO_REPLACE, NECDataset.OOV, NECDataset.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(NECDataset.WORD_NOISE_TYPE)

    return {
        'train_transformation': data.TransformTwiceNEC(addNoise),
        'eval_transformation': None,
        'datadir': 'data-local/nec/conll',
        'num_classes': 4
    }

##### USING Torchtext ... now reverting to using custom code
# def simple_tokenizer(datapoint):
#     fields = datapoint.split("__")
#     return fields
######################################################################



#mithun: ohh the Dataset is a pytorch class, not a python class..and this class NECDataset just inherits from that class. Apparently in python you inherit by passing the parent into the constructor. WHo knew
class NECDataset(Dataset):

    PAD = "@PADDING"
    OOV = "</s>"
    ENTITY = "@ENTITY"
    OOV_ID = 0
    ENTITY_ID = -1
    NUM_WORDS_TO_REPLACE = 1
    WORD_NOISE_TYPE = "drop"

    #mithun this is called using:#dataset = datasets.NECDataset(traindir, args, train_transformation)
    #transform means, the kind of dropping you want to do. look for function called fever():
    def __init__(self, dir, args, transform=None):
        entity_vocab_file = dir + "/entity_vocabulary.emboot.filtered.txt"
        context_vocab_file = dir + "/pattern_vocabulary_emboot.filtered.txt"
        dataset_file = dir + "/training_data_with_labels_emboot.filtered.txt"
        w2vfile = dir + "/../../vectors.goldbergdeps.txt"

        self.args = args
        self.entity_vocab = Vocabulary.from_file(entity_vocab_file)
        self.context_vocab = Vocabulary.from_file(context_vocab_file)

        #This is the place where actual reading of data happens- i think the mentions and contexts can be replaced with headline and bodies-
        #and labels can be the one of the three AGREE, DISAGREE , NEUTRAL
        self.mentions, self.contexts, self.labels_str = Datautils.read_data(dataset_file, self.entity_vocab, self.context_vocab)
        self.word_vocab, self.max_entity_len, self.max_pattern_len, self.max_num_patterns = self.build_word_vocabulary()

        if args.pretrained_wordemb:
            if args.eval_subdir not in dir:  # do not load the word embeddings again in eval
                self.gigaW2vEmbed, self.lookupGiga = Gigaword.load_pretrained_embeddings(w2vfile)
                self.word_vocab_embed = self.create_word_vocab_embed()

        else:
            print("Not loading the pretrained embeddings ... ")
            assert args.update_pretrained_wordemb, "Pretrained embeddings should be updated but " \
                                                   "--update-pretrained-wordemb = {}".format(args.update_pretrained_wordemb)
            self.word_vocab_embed = None

        # NOTE: Setting some class variables
        NECDataset.OOV_ID = self.word_vocab.get_id(NECDataset.OOV)
        NECDataset.ENTITY_ID = self.word_vocab.get_id(NECDataset.ENTITY)

        type_of_noise, size_of_noise = args.word_noise.split(":")
        NECDataset.WORD_NOISE_TYPE = type_of_noise

        NECDataset.NUM_WORDS_TO_REPLACE = int(size_of_noise)

        categories = sorted(list({l for l in self.labels_str}))
        self.lbl = [categories.index(l) for l in self.labels_str]

        self.transform = transform

    def sanitise_and_lookup_embedding(self, word_id):

        word = Gigaword.sanitiseWord(self.word_vocab.get_word(word_id))

        if word in self.lookupGiga:
            word_embed = Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga[word]])
        else:
            word_embed = Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<unk>"]])

        return word_embed

    def create_word_vocab_embed(self):

        word_vocab_embed = list()

        # leave last word = "@PADDING"
        for word_id in range(0, self.word_vocab.size()-1):
            word_embed = self.sanitise_and_lookup_embedding(word_id)
            word_vocab_embed.append(word_embed)

        # NOTE: adding the embed for @PADDING
        word_vocab_embed.append(Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<pad>"]]))
        return np.array(word_vocab_embed).astype('float32')

    def build_word_vocabulary(self):
        word_vocab = Vocabulary()

        max_entity_len = 0
        max_pattern_len = 0
        max_num_patterns = 0

        max_entity = ""
        max_pattern = ""

        for mentionId in self.mentions:
            words = [w for w in self.entity_vocab.get_word(mentionId).split(" ")]
            for w in words:
                word_vocab.add(w)

            if len(words) > max_entity_len:
                max_entity_len = len(words)
                max_entity = words

        for context in self.contexts:
            for patternId in context:
                words = [w for w in self.context_vocab.get_word(patternId).split(" ")]
                for w in words:
                    word_vocab.add(w)

                if len(words) > max_pattern_len:
                    max_pattern_len = len(words)
                    max_pattern = words

            if len(context) > max_num_patterns:
                max_num_patterns = len(context)

        word_vocab.add(NECDataset.PAD, 0)  # Note: Init a count of 0 to PAD, as we are not using it other than padding
        # print (max_entity)
        # print (max_entity_len)
        # print (max_pattern)
        # print (max_pattern_len)
        return word_vocab, max_entity_len, max_pattern_len, max_num_patterns

    def __len__(self):
        return len(self.mentions)

    def pad_item(self, dataitem, isPattern=True):
        if isPattern: # Note: precessing patterns .. consisting of list of lists (add pad to each list) and a final pad to the list of list
            dataitem_padded = list()
            for datum in dataitem:
                datum_padded = datum + [self.word_vocab.get_id(NECDataset.PAD)] * (self.max_pattern_len - len(datum))
                dataitem_padded.append(datum_padded)
            for _ in range(0, self.max_num_patterns - len(dataitem)):
                dataitem_padded.append([self.word_vocab.get_id(NECDataset.PAD)] * self.max_pattern_len)
        else:  # Note: padding an entity (consisting of a seq of tokens)
            dataitem_padded = dataitem + [self.word_vocab.get_id(NECDataset.PAD)] * (self.max_entity_len - len(dataitem))

        return dataitem_padded

    def get_num_classes(self):
        return len(list({l for l in self.lbl}))

    def get_labels(self):
        return self.lbl

    #mithun:apparently __getitem__ is a function of pytorch's Dataset class. Which this class inherits. Here he is just overriding it
    # go to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html and search for __getitem__
    # this is some internal memory saving thing to not load the entire dataset into memory at once.
    #askajay: so if i want to do some data processing on the raw data that i read from disk, is this the point where i do it? for each data point kind of thing?

    def __getitem__(self, idx):
        entity_words = [self.word_vocab.get_id(w) for w in self.entity_vocab.get_word(self.mentions[idx]).split(" ")]
        entity_words_padded = self.pad_item(entity_words, isPattern=False)
        entity_datum = torch.LongTensor(entity_words_padded)

        context_words_str = [[w for w in self.context_vocab.get_word(ctxId).split(" ")] for ctxId in self.contexts[idx]]
        context_words = [[self.word_vocab.get_id(w) for w in self.context_vocab.get_word(ctxId).split(" ")] for ctxId in self.contexts[idx]]

        if self.transform is not None:
            # 1. Replace word with synonym word in Wordnet / NIL (whichever is enabled)
            context_words_dropout_str = self.transform(context_words_str, NECDataset.ENTITY)

            if NECDataset.WORD_NOISE_TYPE == 'replace':
                assert len(context_words_dropout_str) == 2, "There is some issue with TransformTwice ... " #todo: what if we do not want to use the teacher ?
                new_replaced_words = [w for ctx in context_words_dropout_str[0] + context_words_dropout_str[1]
                                        for w in ctx
                                        if not self.word_vocab.contains(w)]

                # 2. Add word to word vocab (expand vocab)
                new_replaced_word_ids = [self.word_vocab.add(w, count=1)
                                         for w in new_replaced_words]

                # 3. Add the replaced words to the word_vocab_embed (if using pre-trained embedding)
                if self.args.pretrained_wordemb:
                    for word_id in new_replaced_word_ids:
                        word_embed = self.sanitise_and_lookup_embedding(word_id)
                        self.word_vocab_embed = np.vstack([self.word_vocab_embed, word_embed])

                # print("Added " + str(len(new_replaced_words)) + " words to the word_vocab... New Size: " + str(self.word_vocab.size()))

            context_words_dropout = list()
            context_words_dropout.append([[self.word_vocab.get_id(w)
                                            for w in ctx]
                                           for ctx in context_words_dropout_str[0]])
            x.append([[self.word_vocab.get_id(w)
                                            for w in ctx]
                                           for ctx in context_words_dropout_str[1]])

            if len(context_words_dropout) == 2:  # transform twice (1. student 2. teacher): DONE
                context_words_padded_0 = self.pad_item(context_words_dropout[0])
                context_words_padded_1 = self.pad_item(context_words_dropout[1])
                context_datums = (torch.LongTensor(context_words_padded_0), torch.LongTensor(context_words_padded_1))
            else: # todo: change this to an assert (if we are always using the student and teacher networks)
                context_words_padded = self.pad_item(context_words_dropout)
                context_datums = torch.LongTensor(context_words_padded)
        else:
            context_words_padded = self.pad_item(context_words)
            context_datums = torch.LongTensor(context_words_padded)

        # print ("label : " + self.labels[idx])
        # print ("label id : " + str(self.label_ids_all[idx]))
        label = self.lbl[idx]  # Note: .. no need to create a tensor variable

        #mithun: askajay i think this is the place where the data that is read from the disk is sent back to main.py
        #ans: no. this is the per data point enumerator. getitem is called internally by pytorch
        # that is because in evaluation we won't add noise.
        if self.transform is not None:
            return (entity_datum, context_datums[0]), (entity_datum, context_datums[1]), label
        else:
            return (entity_datum, context_datums), label

        ##### USING Torchtext ... now reverting to using custom code
        # print ("Dir in NECDataset : " + dir)
        # data_file = "training_data_with_labels_emboot.filtered.txt.processed"
        #
        # LABEL = Field(sequential=False, use_vocab=True)
        # ENTITY = Field(sequential=False, use_vocab=True, lower=True)
        # PATTERN = Field(sequential=True, use_vocab=True, lower=True, tokenize=simple_tokenizer)
        #
        # datafields = [("label", LABEL), ("entity", ENTITY), ("patterns", PATTERN)]
        # dataset, _ = TabularDataset.splits(path=dir, train=data_file, validation=data_file, format='tsv',
        #                                  fields=datafields)
        #
        # LABEL.build_vocab(dataset)
        # ENTITY.build_vocab(dataset)
        # PATTERN.build_vocab(dataset)

        # APPLY THE TRANSFORMATION HERE
        # transform = transform

        # return dataset
        ######################################################################

@export
def riedel():

    if REDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(REDataset.NUM_WORDS_TO_REPLACE, REDataset.OOV, REDataset.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(REDataset.WORD_NOISE_TYPE)

    return {
        'train_transformation': data.TransformTwiceNEC(addNoise),
        'eval_transformation': None,
        'datadir': 'data-local/re/Riedel2010'
    }

@export
def gids():

    if REDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(REDataset.NUM_WORDS_TO_REPLACE, REDataset.OOV, REDataset.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(REDataset.WORD_NOISE_TYPE)

    return {
        'train_transformation': data.TransformTwiceNEC(addNoise),
        'eval_transformation': None,
        'datadir': 'data-local/re/gids'
    }






@export
def fever():

    if RTEDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(NECDataset.NUM_WORDS_TO_REPLACE, NECDataset.OOV, NECDataset.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(NECDataset.WORD_NOISE_TYPE)

    return {
        'train_transformation': data.TransformTwiceNEC(addNoise),
        'eval_transformation': None,
        #'datadir': 'data-local/rte/fever'
        #ask ajay what does this do? why comment out?
        # 'num_classes': 11
    }

class RTEDataset(Dataset):

    PAD = "@PADDING"
    OOV = "</s>"
    ENTITY = "@ENTITY"
    OOV_ID = 0
    ENTITY_ID = -1
    NUM_WORDS_TO_REPLACE = 1
    WORD_NOISE_TYPE = "drop"

    def create_word_vocab_embed(self, args):

        word_vocab_embed = list()

        # leave last word = "@PADDING"
        for word_id in range(0, self.word_vocab.size() - 1):
            word_embed = self.sanitise_and_lookup_embedding(word_id, args)
            word_vocab_embed.append(word_embed)

        # NOTE: adding the embed for @PADDING
        word_vocab_embed.append(Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<pad>"]]))
        return np.array(word_vocab_embed).astype('float32')

    #mithun this is called using:#dataset = datasets.NECDataset(traindir, args, train_transformation)
    def __init__(self, dataset_file, args, LOG,transform=None):


        self.claims, self.evidences, self.labels_str = Datautils.read_rte_data(dataset_file,args)

        assert len(self.claims)== len(self.evidences)==len(self.labels_str), "claims and evidences are not of equal length"

        #to find the top 10 longest evidences adn remove them. am doing this because GPU was getting memory overloaded because of padding
        list_of_longest_ev_lengths=[]
        list_of_longest_evidences=[]
        max_evidence_len=0
        for each_ev in self.evidences:
            words = [w for w in each_ev.split(" ")]
            if len(words) > max_evidence_len:
                    max_evidence_len = len(words)
                    longest_evidence_words = words
                    list_of_longest_ev_lengths.append(max_evidence_len)
                    list_of_longest_evidences.append(longest_evidence_words)

        # s = sorted(list_of_longest_evidences, key=len, reverse=True)
        # top10 = s[:10]
        # # LOG.debug(f"list_of_longest_evidences.sort(:{top10}")
        # s_lengths = sorted(list_of_longest_ev_lengths, reverse=True)
        # #LOG.debug(f"list_of_longest_ev_lengths.sort(:{s_lengths[:10]}")
        #
        #

        self.word_vocab, self.max_claims_len, self.max_ev_len = self.build_word_vocabulary(LOG)

        LOG.debug(f"inside datasets.py line 431r value of word_vocab.size()={self.word_vocab.size()}")



        #askfan :can i do this above word count thing later?- right now i want all words, maybe, for starters? Ans: yes
        # for word in self.word_counts:
        #     if self.word_counts[word] >= args.word_frequency:
        #         self.word_vocab.add(word, self.word_counts[word])


        self.word_vocab.add("@PADDING", 0)
        self.pad_id = self.word_vocab.get_id(RTEDataset.PAD)

        #todo: load pretrained wordemb

        if args.pretrained_wordemb:
            if args.eval_subdir not in dir:  # do not load the word embeddings again in eval

                #todo for mithun: should come up with a saner test than checking in dir.
                # Right now , jan 28th2019, i have removed dir..should pass a flag from command line explicitly when doing dev or something. this check in dir is realy stupid

                self.gigaW2vEmbed, self.lookupGiga = Gigaword.load_pretrained_embeddings(w2vfile)
                self.word_vocab_embed = self.create_word_vocab_embed()

        else:
            print("Not loading the pretrained embeddings ... ")
            assert args.update_pretrained_wordemb, "Pretrained embeddings should be updated but " \
                                                   "--update-pretrained-wordemb = {}".format(args.update_pretrained_wordemb)
            self.word_vocab_embed = None




        print("self.word_vocab.size=", self.word_vocab.size())

        self.categories = sorted(list({l for l in self.labels_str}))
        self.lbl = [self.categories.index(l) for l in self.labels_str]

        #write the vocab file to disk so that you can load it later
        if(LOG is not None):
            LOG.info("self.word_vocab.size=", self.word_vocab.size())
            LOG.info("self.words.size=", len(self.word_vocab.words))

        dir=args.output_folder
        vocab_file = dir + 'vocabulary_train_' + '.txt'
        self.word_vocab.to_file(vocab_file)


        print("num of types of labels considered =", len(self.categories))

        #write the list of labels to disk
        label_category_file = dir + 'label_category_train_'  + '.txt'
        with io.open(label_category_file, 'w', encoding='utf8') as f:
            for lbl in self.categories:
                f.write(lbl + '\n')

        #self.transform = transform
        self.transform = None

    def __len__(self):
        return len(self.claims)

    def build_word_vocabulary(self,LOG):
        word_vocab = Vocabulary()

        max_claim_len = 0
        max_evidence_len = 0
        max_num_evidences = 0

        max_claim = ""
        longest_evidence_words = ""

        list_of_longest_ev_lengths=[]
        list_of_longest_evidences=[]
        list_of_longest_claim_lengths = []


        for each_claim in self.claims:
            words = [w for w in each_claim.split(" ")]
            for w in words:
                word_vocab.add(w)

                if len(words) > max_claim_len:
                    max_claim_len = len(words)
                    max_claim = words
                    list_of_longest_claim_lengths.append(max_claim_len)

        for each_ev in self.evidences:
            words = [w for w in each_ev.split(" ")]
            for w in words:
                word_vocab.add(w)

                if len(words) > max_evidence_len:
                    max_evidence_len = len(words)
                    longest_evidence_words = words
                    list_of_longest_ev_lengths.append(max_evidence_len)
                    list_of_longest_evidences.append(longest_evidence_words)

            # if len(context) > max_num_patterns:
            #     max_num_patterns = len(context)


        ######So looked like the longest sentence of 18000 words was a bug. Somehow the data had an entire html dump. So right now we are going to pick
        # the biggest number that is  less than 1000
        # Todo: Note that this is a hack and we are biasing the data. Need to find a cleaner way to do this.

        # for x in sorted(list_of_longest_ev_lengths,reverse=True):
        #     if(x<1000):
        #         max_evidence_len=x
        #         break


        #for debug: find the top 10 longest sentences
        #  and their length
        s=sorted(list_of_longest_evidences,key=len,reverse=True)
        top10=s[:10]
        #LOG.debug(f"list_of_longest_evidences.sort(:{top10}")
        s_lengths=sorted(list_of_longest_ev_lengths,reverse=True)
        #LOG.debug(f"list_of_longest_ev_lengths.sort(:{s_lengths[:10]}")

        claim_sorted_len=sorted(list_of_longest_claim_lengths,reverse=True)
        x=claim_sorted_len[:10]
        #LOG.debug(f"claim_sorted_len_t10.(:{x}")



        #LOG.debug (f"max_claim:{max_claim}")
        #LOG.debug (max_claim_len)
        #LOG.debug (longest_evidence_words)
        #LOG.debug (max_evidence_len)
        # import sys
        # sys.exit(1)


        return word_vocab, max_claim_len, max_evidence_len


    def pad_item(self, dataitem,isev=False):
        if(isev):
            dataitem_padded = dataitem + [self.word_vocab.get_id(NECDataset.PAD)] * (self.max_ev_len - len(dataitem))
        #ask becky : right now am padding with the max entity length. that is what fan also is doing .shouldn't i be padding both claim and evidence -with its own max length (eg:20 and 18719)
        # or should i pad upto  the biggest amongst both, i.e 18719 words in evidence
        else:
            dataitem_padded = dataitem + [self.word_vocab.get_id(NECDataset.PAD)] * (self.max_claims_len - len(dataitem))

        return dataitem_padded

    def get_num_classes(self):
        return len(list({l for l in self.lbl}))

    def get_labels(self):
        return self.lbl

    # __getitem__ is a function of pytorch's Dataset class. Which this class inherits. Here he is just overriding it
    # go to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html and search for __getitem__
    # this is some internal memory saving thing to not load the entire dataset into memory at once.
    #askajay: so if i want to do some data processing on the raw data that i read from disk, is this the point where i do it? for each data point kind of thing?

    def __getitem__(self, idx):

        # for each word in claim (and evidence in turn) get the corresponding unique id



        label = self.lbl[idx]

        # ask fan: can we just use the common word vocabulary dictionary. do we need to have separate dict
        # Ans: the dictionar is still the same, its two different sentences and two different words

        # entity_words = [self.word_vocab.get_id(w) for w in self.entity_vocab.get_word(self.mentions[idx]).split(" ")]
        # entity_words_padded = self.pad_item(entity_words, False)
        # entity_datum = torch.LongTensor(entity_words_padded)
        #
        # context_words_str = [[w for w in self.context_vocab.get_word(ctxId).split(" ")] for ctxId in self.contexts[idx]]
        # context_words = [[self.word_vocab.get_id(w) for w in self.context_vocab.get_word(ctxId).split(" ")] for ctxId in
        #                  self.contexts[idx]]

        #todo: ask becky if we should do lowercase for all words in claims and evidence

        claims_words_str = [[w for w in (self.claims[idx].split(" "))]]
        ev_words_str= [[w for w in (self.evidences[idx].split(" "))]]

        claims_words_id = [self.word_vocab.get_id(w) for w in (self.claims[idx].split(" "))]
        ev_words_id = [self.word_vocab.get_id(w) for w in (self.evidences[idx].split(" "))]


        len_claims_words=len(claims_words_id)
        len_evidence_words = len(ev_words_id)

        claims_words_id_padded = self.pad_item(claims_words_id)
        ev_words_id_padded = self.pad_item(ev_words_id,True)


        if self.transform is not None:

            #add noise to both claim and evidence anyway. on top of it, if you want to add replacement,
            # or make it
            #mutually exclusive, do it later. Also note that this function will return two strings
            # each for one string given.
            #that is because it assumes different transformation for student and teacher.

            claim_words_dropout_str = self.transform(claims_words_str, RTEDataset.ENTITY)
            ev_words_dropout_str = self.transform(ev_words_str, RTEDataset.ENTITY)

            # 1. Replace word with synonym word in Wordnet / NIL (whichever is enabled)

            if RTEDataset.WORD_NOISE_TYPE == 'replace':
                assert len(claim_words_dropout_str) == 2, "There is some issue with TransformTwice ... " #todo: what if we do not want to use the teacher ?
                new_replaced_words = [w for ctx in claim_words_dropout_str[0] + claim_words_dropout_str[1]
                                        for w in ctx
                                        if not self.word_vocab.contains(w)]

                # 2. Add word to word vocab (expand vocab)
                new_replaced_word_ids = [self.word_vocab.add(w, count=1)
                                         for w in new_replaced_words]

                # 3. Add the replaced words to the word_vocab_embed (if using pre-trained embedding)
                if self.args.pretrained_wordemb:
                    for word_id in new_replaced_word_ids:
                        word_embed = self.sanitise_and_lookup_embedding(word_id)
                        self.word_vocab_embed = np.vstack([self.word_vocab_embed, word_embed])

                # print("Added " + str(len(new_replaced_words)) + " words to the word_vocab... New Size: " + str(self.word_vocab.size()))


            #back to drop world: now pad 4 things separately i.e claim for teacher, claim for student, evidence for teacher, evidence for student
            claim_dropout_word_ids = list()

            #for each word in the claim (note, this is after drop out), find its corresponding ids from the vocabulary dictionary
            claim_dropout_word_ids.append([[self.word_vocab.get_id(w)
                                         for w in ctx]
                                        for ctx in claim_words_dropout_str[0]])
            claim_dropout_word_ids.append([[self.word_vocab.get_id(w)
                                         for w in ctx]
                                        for ctx in claim_words_dropout_str[1]])

            if len(claim_dropout_word_ids) == 2:  # i.e if its ==2 , it means transform twice (1. student 2. teacher)
                claims_words_padded_0 = self.pad_item(claim_dropout_word_ids[0][0])
                claims_words_padded_1 = self.pad_item(claim_dropout_word_ids[1][0])
                claims_datum = (torch.LongTensor(claims_words_padded_0), torch.LongTensor(claims_words_padded_1))
            else:
                # todo: change this to an assert (if we are always using the student and teacher networks)
                context_words_padded = self.pad_item(claim_dropout_word_ids)
                claims_datum = torch.LongTensor(context_words_padded)

            #do the same for evidence also
            evidence_words_dropout = list()
            evidence_words_dropout.append([[self.word_vocab.get_id(w)
                                         for w in ctx]
                                        for ctx in ev_words_dropout_str[0]])
            evidence_words_dropout.append([[self.word_vocab.get_id(w)
                                         for w in ctx]
                                        for ctx in ev_words_dropout_str[1]])

            if len(evidence_words_dropout) == 2:  # transform twice (1. student 2. teacher): DONE

                #if its evidence , and not claim, pad_item requires the second argument to be True
                evidence_words_padded_0 = self.pad_item(evidence_words_dropout[0][0],True)
                evidence_words_padded_1 = self.pad_item(evidence_words_dropout[1][0],True)
                evidence_datum = (torch.LongTensor(evidence_words_padded_0), torch.LongTensor(evidence_words_padded_1))
            else:
                # todo: change this to an assert (if we are always using the student and teacher networks)
                context_words_padded = self.pad_item(evidence_words_dropout)
                evidence_datum = torch.LongTensor(context_words_padded)

        #if we are not doing any transformation or adding any noise, just pad plain claim adn evidence
        else:
            claims_datum = torch.LongTensor(claims_words_id_padded)
            evidence_datum = torch.LongTensor(ev_words_id_padded)



        # transform means, if you want a different noise for student and teacher
        # so if you are transforming (i.e adding noise to both claim and evidence, for both student and teacher
        #  , you will be returning two different types of claim and evidence. else just one.

        if self.transform is not None:
            return (claims_datum[0], evidence_datum[0]), (claims_datum[1], evidence_datum[1]), label, (len_claims_words,len_evidence_words)
        else:
            return (claims_datum, evidence_datum), label,(len_claims_words,len_evidence_words)



