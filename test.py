from src.f5_tts.train.vocab_generator import vocab_checker, vocab_changer

vocab_path = "train/f5tts_zzal_v2/vocab.txt"
meta_path = "/home/ubuntu/Documents/dataset/audio_text/aihub131/phonemize_train.txt"
fixes_path = "/home/ubuntu/Documents/dataset/audio_text/aihub131/vocab_fix.json"
fixed_meta_path = "/home/ubuntu/Documents/dataset/audio_text/aihub131/phonemize_train.txt"
vocab_map, fixes_path = vocab_checker(vocab_path, meta_path, fixes_path)
# vocab_changer(meta_path, fixes_path, fixed_meta_path, backup=True)