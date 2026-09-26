# Third-party notices

## Silent-Face-Anti-Spoofing

The optional `kyc_engine.minifasnet` adapter contains limited adaptations of
[Minivision's Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
project, Copyright 2020 Minivision. The upstream project is licensed under the
Apache License, Version 2.0.

Adapted inference logic is limited to the behavior of upstream
`src/anti_spoof_predict.py`, `src/generate_patches.py`, `src/utility.py`,
`src/data_io/transform.py`, and the V2/V1SE portions of
`src/model_lib/MiniFASNet.py`. Training, data-loading, UI, and CLI code were
not copied. Modified source headers identify this origin. The complete
Apache-2.0 license is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
