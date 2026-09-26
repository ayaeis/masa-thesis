import json
from pathlib import Path

import numpy as np

from feeder.single_dataset.MSASL import MSASL


class MSASLArchive(MSASL):
    """Native MSASL feeder with documented unavailable/unusable records removed."""

    def __init__(self, data_root, data_split="train", class_num=200,
                 use_cache=False, **kwargs):
        super().__init__(
            data_root=data_root,
            data_split=data_split,
            class_num=class_num,
            use_cache=use_cache,
            **kwargs,
        )

        root = Path(data_root)
        records = json.loads(
            (root / "unavailable_msasl200_records.json").read_text()
        )
        records += json.loads(
            (root / "native_feeder_exclusions.json").read_text()
        )
        excluded = {
            item["filename"]
            for item in records
            if item["split"] == data_split
        }

        if data_split == "train":
            train = [
                row for row in self.video_list[:self.flag].tolist()
                if row[0] not in excluded
            ]
            val = [
                row for row in self.video_list[self.flag:].tolist()
                if row[0] not in excluded
            ]
            self.flag = len(train)
            self.video_list = np.asarray(train + val, dtype=str)
        else:
            self.video_list = np.asarray(
                [row for row in self.video_list.tolist()
                 if row[0] not in excluded],
                dtype=str,
            )

        print(f"MSASL archive-available {data_split}: {len(self.video_list)}")
