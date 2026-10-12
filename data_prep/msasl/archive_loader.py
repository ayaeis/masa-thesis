"""Select the prepared archive loader without changing dataset-specific filtering."""

from data_prep.msasl.msasl1000_archive_loader import MSASL1000Archive
from data_prep.msasl.msasl200_archive_loader import MSASLArchive


def load_archive(data_root, data_split="train", class_num=200,
                 use_cache=False, **kwargs):
    loader = MSASL1000Archive if class_num == 1000 else MSASLArchive
    return loader(
        data_root=data_root,
        data_split=data_split,
        class_num=class_num,
        use_cache=use_cache,
        **kwargs,
    )
