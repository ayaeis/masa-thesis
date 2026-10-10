import json
from pathlib import Path

import numpy as np

from feeder.single_dataset.MSASL import MSASL


class MSASL1000Archive(MSASL):
    """Native MSASL1000 feeder excluding only documented unavailable clips."""

    def __init__(self, data_root, data_split="train", class_num=1000,
                 use_cache=False, **kwargs):
        super().__init__(
            data_root=data_root,
            data_split=data_split,
            class_num=class_num,
            use_cache=use_cache,
            **kwargs,
        )

        root = Path(data_root)
        exclusions = []
        for name in ("unavailable_records.json", "native_feeder_exclusions.json"):
            path = root / name
            if path.is_file():
                exclusions.extend(json.loads(path.read_text()))
        if data_split == "train":
            # The native MSASL feeder combines training and validation into
            # one dataset, so exclusions from both source splits apply here.
            excluded = {
                item["filename"]
                for item in exclusions
                if item["split"] in {"train", "val"}
            }
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
            excluded = {
                item["filename"]
                for item in exclusions
                if item["split"] == data_split
            }
            self.video_list = np.asarray(
                [row for row in self.video_list.tolist() if row[0] not in excluded],
                dtype=str,
            )

        print(f"MSASL1000 archive-available {data_split}: {len(self.video_list)}")
