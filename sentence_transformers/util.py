import requests
from torch import Tensor, device
from typing import Tuple, List
from tqdm import tqdm
import sys
import importlib
from sklearn import metrics

def calculate_micro_f1(y_pred, y_target,label,accept=False):
        '''
            #calculate per class microf1
            use label=Unrelated is needed for measurements in fake news datasets, because in fake news RTE world,
            we really don't care much about it when two documents are unrelated, except for an initial filtering out process
        :param y_pred:
        :param y_target:
        :param labels:
        :param accept: if True, include only labels in labels for microf1. if False included everything except these labels
        :return:
        '''
        assert len(y_pred) == len(y_target), "lengths are different {len(y_pred)}"
        labels_to_include =[]
        for index,l in enumerate(y_target):
            if (accept==False):
                if not (l==label):
                    labels_to_include.append(index)
            else:
                if (l==label):
                    labels_to_include.append(index)
        mf1=metrics.f1_score(y_target,y_pred, average='micro', labels=labels_to_include)
        return mf1


def batch_to_device(batch, target_device: device):
    """
    send a batch to a device

    :param batch:
    :param target_device:
    :return: the batch sent to the device
    """
    features = batch['features']
    for paired_sentence_idx in range(len(features)):
        for feature_name in features[paired_sentence_idx]:
            features[paired_sentence_idx][feature_name] = features[paired_sentence_idx][feature_name].to(target_device)

    labels = batch['labels'].to(target_device)
    return features, labels



def http_get(url, path):
    with open(path, "wb") as file_binary:
        req = requests.get(url, stream=True)
        if req.status_code != 200:
            print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
            req.raise_for_status()

        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)
    progress.close()


def fullname(o):
  # o.__module__ + "." + o.__class__.__qualname__ is an example in
  # this context of H.L. Mencken's "neat, plausible, and wrong."
  # Python makes no guarantees as to whether the __module__ special
  # attribute is defined, so we take a more circumspect approach.
  # Alas, the module name is explicitly excluded from __qualname__
  # in Python 3.

  module = o.__class__.__module__
  if module is None or module == str.__class__.__module__:
    return o.__class__.__name__  # Avoid reporting __builtin__
  else:
    return module + '.' + o.__class__.__name__

def import_from_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)

