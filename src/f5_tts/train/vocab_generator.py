# phoneme_vocab_generator.py

# 기본 심볼 정의
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱᵝʲʷˠˤ˞↓↑→↗↘\'̩ᵻ"
_ipa_jp = "äᵝĩũ"

# 특수 기호로 간주할 문자들 (원본 vocab 참고)
_special_symbols = (
    ';:,.!?¡¿—…"\'«»“”-_()[]'
    '#%&*+=>@\\/'  # 원본 vocab에서 발견된 추가 특수기호 포함
)

symbols = [' ']

# 모든 문자 통합 후 중복 제거
symbol_set = set()
symbol_set.update(_special_symbols)
symbol_set.update(_letters)
symbol_set.update(_letters_ipa)
symbol_set.update(_ipa_jp)

# 정렬 (가독성과 일관성을 위해)
symbols += sorted(symbol_set)

# vocab.txt로 저장
vocab_path = "data/F5TTS_zzal_v1.vocab.txt"
with open(vocab_path, "w", encoding="utf-8") as f:
    for s in symbols:
        f.write(s + "\n")

print(f"✅ vocab.txt 생성 완료: 총 {len(symbols)}개 토큰")
