# jma2json

jma2json is a small Python helper that turns raw Earthquake Early Warning (EEW) telegrams from the Japan Meteorological Agency (JMA) into structured JSON. It follows the official telegram specification and ships with lookup tables and localized strings so that downstream tools can work with human-friendly field names.

## Features

- Parses single and multi-part EEW telegrams into rich Python dataclasses or JSON.
- Normalizes timestamps, magnitude, hypocenter coordinates, intensity codes, and reliability flags.
- Reassembles split messages by tracking the `CNF` remainder flag so the caller only has to deal with complete payloads.
- Includes bundled tables (`tables.json`) and strings (`strings/ja_JP.json`) that map JMA codes to human-readable descriptions. 

## Installation

This project targets Python 3.9+ and has no runtime dependencies beyond the standard library. Clone the repository and install it in editable mode if you want to import it elsewhere:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you simply want to experiment inside the repository you can skip the editable install and rely on the package being on `PYTHONPATH`.

## Usage

A minimal end-to-end example that fetches the latest publicly mirrored EEW telegrams and decodes it to JSON. Find it in `example.py`:

```python
import httpx
from jma2json import decode_to_json

# fetches the latest EEW report from JMA
r = httpx.get("https://api.wolfx.jp/jma_eew.json").json()

json_payload = decode_to_json(r['OriginalText'], ensure_ascii=False, indent=2)
print(json_payload)
```

For offline processing you can feed `EEWDecoder.add_telegram()` with raw telegram text (one message per string). The method buffers partial telegrams and returns decoded `EEWMessage` objects as soon as the final segment arrives:

```python
from jma2json import EEWDecoder

decoder = EEWDecoder()

decoder.add_telegram("<first chunk>")          # complete: False
result = decoder.add_telegram("<final chunk>")  # complete: True

if result["complete"]:
    for message in result["messages"]:
        print(message.to_dict())
```

`EEWMessage` instances expose the decoded data both as attributes and through `to_dict()` for easy JSON serialization.

## Custom tables and localization

EEW telegrams rely heavily on numeric codes. Region, epicentre, intensity, and reliability tables are included in `tables.json`, plus localized labels in `strings/ja_JP.json`. You can extend or replace them:

```python
from jma2json import EEWDecoder, load_tables

decoder = EEWDecoder()
load_tables(decoder, epicenter_json_path="my_epicenters.json", region_json_path="my_regions.json")
```

The JSON files should contain simple `code -> label` mappings. If you omit a mapping, the raw code is returned so you can handle it yourself.

## Limitations

- The library focuses on telegram type VXSE (EEW). Other telegram types may decode partially.
- Fields with `//` or `///` placeholders from the source message are surfaced as `None`.
- Network access is not required for decoding; the bundled example fetches data from a community api [Wolfx API](https://api.wolfx.jp) for convenience only.

## Resources

* https://www.data.jma.go.jp/suishin/shiyou/pdf/no40202
* https://github.com/skubota/eew
