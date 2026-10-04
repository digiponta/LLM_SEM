from tokenizer import Tokenizer

TEXT = '量子センサーは、原子や電子などのミクロな「量子」の性質を利用して、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。'

def main():
    tok = Tokenizer.load('model/tokenizer.json')
    unknown = sorted({ch for ch in TEXT if tok.token_to_id.get(ch, tok.unk_id) == tok.unk_id})
    print('Vocabulary size :', tok.vocab_size)
    print('Text chars      :', len(TEXT))
    print('UNK chars       :', unknown or '(none)')
    if unknown:
        print('RESULT          : FAIL - exact internalization is impossible with current tokenizer')
        raise SystemExit(1)
    print('RESULT          : PASS')

if __name__ == '__main__':
    main()
