import json
import torchaudio

from pathlib import Path
from typing import Dict, Set, Tuple

from src.f5_tts.model.utils import get_tokenizer

# 기본 심볼 정의
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱᵝʲʷˠˤ˞↓↑→↗↘\'̩ᵻ"
_ipa_jp = "äᵝĩũ"

_punctuation = ';:,.!?¡¿—-~\"\'/“”'
# 특수 기호로 간주할 문자들 (원본 vocab 참고)
_special_symbols = (
    ';:,.!?¡¿—…"\'«»“”-_()[]'
    '#%&*+=>@\\/'  # 원본 vocab에서 발견된 추가 특수기호 포함
)

symbols = [' '] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_ipa_jp)


# 모든 문자 통합 후 중복 제거
# symbol_set = set()
# symbol_set.update(_special_symbols)
# symbol_set.update(_letters)
# symbol_set.update(_letters_ipa)
# symbol_set.update(_ipa_jp)

# 정렬 (가독성과 일관성을 위해)
# symbols += sorted(symbol_set)

def vocab_checker(vocab_path:str, meta_path:str, fixes_path:str):
    vocab_map, _ = get_tokenizer(vocab_path, 'char')    
    unknown_chars: Set[str] = set()
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "|" not in line:
                continue
            try:
                _, text, *_ = line.split("|", 4)
            except ValueError:
                continue
            unknown_chars.update(ch for ch in text if ch not in vocab_map)

    # 2) 딕셔너리 → JSON 저장
    err_dict = {ch: None for ch in sorted(unknown_chars)}
    with open(fixes_path, "w", encoding="utf-8") as jf:
        json.dump(err_dict, jf, ensure_ascii=False, indent=2)

    print(f"[INFO] {len(err_dict)} unknown chars saved to {fixes_path}")
    return vocab_map, fixes_path

def vocab_changer(meta_path:str, json_path:str, output_meta:str|None=None, backup:bool=True):
    with open(json_path, "r", encoding="utf-8") as jf:
        fix_map: Dict[str, str | None] = json.load(jf)

    # 1) 출력 경로 결정
    output_meta = output_meta or f"{meta_path}.fixed"

    # 2) 백업
    if backup:
        bak_path = f"{meta_path}.bak"
        Path(meta_path).replace(bak_path)
        src_path = bak_path     # 백업본을 읽어 변환
    else:
        src_path = meta_path

    # 3) 변환 수행
    changed, untouched = 0, 0
    with open(src_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_meta, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.rstrip("\n")
            if "|" not in line:
                fout.write(line + "\n")
                continue

            parts = line.split("|", 4)
            if len(parts) < 2:
                fout.write(line + "\n")
                continue

            text = parts[1]
            new_text = "".join(
                fix_map.get(ch, ch) if fix_map.get(ch) is not None else ch
                for ch in text
            )

            if text != new_text:
                changed += 1
            else:
                untouched += 1

            parts[1] = new_text
            fout.write("|".join(parts) + "\n")

    print(f"[INFO] Lines changed: {changed}, untouched: {untouched}")
    print(f"[INFO] Fixed meta saved to: {output_meta}")