# Photo overrides

Drop a file here named `<researcher-id>.<ext>` (jpg/jpeg/png/webp) and it will be used as that researcher's headshot, overriding whatever the parser extracted from the PPTX.

The id is the slug of the researcher's name as it appears in `pipeline/researchers.json`. For example:

- `jessica-staddon.jpg` — overrides Jessica Staddon's headshot
- `john-a-guerra-gomez.jpg` — overrides John A. Guerra Gomez's headshot

Use this when the PPTX slide has no real headshot (e.g., a group photo) or when you want to swap in a higher-quality photo. The pipeline re-encodes overrides to a max 512×512 JPEG, so any reasonable input format works.

After adding a file here, re-run the pipeline:

```bash
python pipeline/run_all.py
```
