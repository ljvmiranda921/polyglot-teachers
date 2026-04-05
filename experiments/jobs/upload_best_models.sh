#!/bin/bash
# Upload the best models (gemma-3-27b-it teacher) for each language

# ar - Gemma-3-4B base
python -m scripts.export_model \
    --branch "20251228T111030-msde-google_gemma-3-4b-pt-lora-4bit-msde-S1-ar_gemma-3-27b-it" \
    --output_repo "ljvmiranda921/Polyglot-Gemma3-4B-SFT-ar" \
    --language ar

# ar - OLMo-3-7B base
python -m scripts.export_model \
    --branch "20251213T225239-msde-allenai_Olmo-3-1025-7B-lora-4bit-msde-S1-ar_gemma-3-27b-it" \
    --output_repo "ljvmiranda921/Polyglot-OLMo3-7B-SFT-ar" \
    --language ar

# cs - OLMo-3-7B base
python -m scripts.export_model \
    --branch "20251214T215219-msde-allenai_Olmo-3-1025-7B-lora-4bit-msde-S1-cs_gemma-3-27b-it" \
    --output_repo "ljvmiranda921/Polyglot-OLMo3-7B-SFT-cs" \
    --language cs

# de - Gemma-3-4B base
python -m scripts.export_model \
    --branch "20251228T153447-msde-google_gemma-3-4b-pt-lora-4bit-msde-S1-de_gemma-3-27b-it" \
    --output_repo "ljvmiranda921/Polyglot-Gemma3-4B-SFT-de" \
    --language de

# de - OLMo-3-7B base
python -m scripts.export_model \
    --branch "20251215T003613-msde-allenai_Olmo-3-1025-7B-lora-4bit-msde-S1-de_gemma-3-27b-it" \
    --output_repo "ljvmiranda921/Polyglot-OLMo3-7B-SFT-de" \
    --language de

# es - OLMo-3-7B base
python -m scripts.export_model \
    --branch "20251215T035832-msde-allenai_Olmo-3-1025-7B-lora-4bit-msde-S1-es_gemma-3-27b-it" \
    --output_repo "ljvmiranda921/Polyglot-OLMo3-7B-SFT-es" \
    --language es

# id - Gemma-3-4B base
python -m scripts.export_model \
    --branch "20251228T195029-msde-google_gemma-3-4b-pt-lora-4bit-msde-S1-id_gemma-3-27b-it" \
    --output_repo "ljvmiranda921/Polyglot-Gemma3-4B-SFT-id" \
    --language id

# id - OLMo-3-7B base
python -m scripts.export_model \
    --branch "20251216T094000-msde-allenai_Olmo-3-1025-7B-lora-4bit-msde-S1-id_gemma-3-27b-it" \
    --output_repo "ljvmiranda921/Polyglot-OLMo3-7B-SFT-id" \
    --language id

# ja - OLMo-3-7B base
python -m scripts.export_model \
    --branch "20251216T122520-msde-allenai_Olmo-3-1025-7B-lora-4bit-msde-S1-ja_gemma-3-27b-it" \
    --output_repo "ljvmiranda921/Polyglot-OLMo3-7B-SFT-ja" \
    --language ja

# tl - Gemma-3-4B base (25k)
python -m scripts.export_model \
    --branch "20260116T173212-msde-google_gemma-3-4b-pt-lora-4bit-tgl_25k-Gemma3" \
    --output_repo "ljvmiranda921/Polyglot-Gemma3-4B-SFT-tl" \
    --language tl
