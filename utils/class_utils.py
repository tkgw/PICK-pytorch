# @Author: Wenwen Yu
# @Created Time: 7/8/2020 9:26 PM
from collections import Counter
from pathlib import Path
from typing import Any, List, Tuple, Union

from torchtext.vocab import Vocab

from . import entities_list


class ClassVocab(Vocab):

    def __init__(self, classes: Union[str, Path, List[str]], specials: Tuple[str, ...] = ('<pad>', '<unk>'), **kwargs: Any) -> None:
        """
        convert key to index(stoi), and get key string by index(itos)
        :param classes: list or str, key string or entity list
        :param specials: list, special tokens except <unk> (default: {['<pad>', '<unk>']})
        :param kwargs:
        """
        cls_list = None
        if isinstance(classes, str):
            cls_list = list(classes)
        if isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('Key file is not found')
            with p.open(encoding='utf8') as f:
                classes = f.read()
                classes = classes.strip()
                cls_list = list(classes)
        elif isinstance(classes, list):
            cls_list = classes
        c: Counter = Counter(cls_list)
        self.special_count = len(specials)
        super().__init__(c, specials=specials, **kwargs)


def entities2iob_labels(entities: List[str]) -> List[str]:
    """
    get all iob string label by entities
    :param entities:
    :return:
    """
    tags: List[str] = []
    for e in entities:
        tags.append('B-{}'.format(e))
        tags.append('I-{}'.format(e))
    tags.append('O')
    return tags


keys_vocab_cls = ClassVocab(Path(__file__).parent.joinpath('keys.txt'), specials_first=False)
iob_labels_vocab_cls = ClassVocab(entities2iob_labels(entities_list.Entities_list), specials_first=False)
entities_vocab_cls = ClassVocab(entities_list.Entities_list, specials_first=False)
