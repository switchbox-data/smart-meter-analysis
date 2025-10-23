from __future__ import annotations

import re
from pathlib import Path

# e.g., ANONYMOUS_DATA_202312_60651-1013.csv
_PAT = re.compile(r".*_(\d{6})_(\d{5})-(\d{4})\.csv$", re.I)


def zip4_from_key(s3_key: str) -> tuple[str, str]:
    """
    Extract (zip5, plus4) from a key/filename like:
    ANONYMOUS_DATA_YYYYMM_ZIP5-PLUS4.csv
    """
    name = Path(s3_key).name
    m = _PAT.match(name)
    if not m:
        # TRY003
        raise ValueError
    _ym, zip5, plus4 = m.groups()
    return zip5, plus4
