import argparse
import csv
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"

QWEN = "Qwen/Qwen3-0.6B-Base"
E5 = "intfloat/multilingual-e5-small"
WORD_RE = re.compile(r"[а-я]+(?:-[а-я]+)*")        # слово: буквы, возможно через дефис
EOW = "</w>"                                      # маркер конца слова

MIN_FREQ = 5                                      # слова реже этого порога в обучение BPE не берём
K = 5                                             # сколько ближайших абзацев возвращает поиск
CONTEXT_TOKENS = 4096

EN_PARAGRAPH = (
    "The prince had been talking for almost an hour, and the guests were growing tired. "
    "Anna looked at the clock, then at the window, then at the clock again. "
    "Nobody dared to leave before the host, and the host did not seem to notice the time at all. "
    "A servant brought tea, and the conversation turned, as it always did in that house, "
    "to the war and to the emperor."
)

QUERIES = [
    ("Хозяйка вечера несколько дней болела гриппом и утром разослала приглашения", "кашляла несколько дней"),
    ("Отец говорит, что оба его сына дураки, только один спокойный, а другой беспокойный", "покойный дурак"),
    ("Богатый и скупой старый князь живет в деревне, его прозвали прусским королем", "прозванный прусским королем"),
    ("Гости по очереди здоровались с никому не нужной тетушкой", "никому не интересной и не нужной тетушки"),
    ("Брат удивительно похож на сестру-красавицу, но при этом очень некрасив", "поразительно дурен собой"),
    ("В гостиную входит невысокий молодой князь с усталым и скучающим взглядом", "усталого, скучающего взгляда"),
    ("Обедневшая княгиня приехала просить, чтобы ее сына перевели в гвардию", "выхлопотать определение в гвардию"),
    ("Спор о революции: народ отдал Бонапарту власть, чтобы избавиться от Бурбонов", "избавил его от Бурбонов"),
    ("Неуклюжий толстый гость по рассеянности взял чужую генеральскую шляпу", "треугольную шляпу с генеральским плюмажем"),
    ("Мальчика десяти лет отправили за границу с гувернером-аббатом", "гувернером-аббатом за границу"),
    ("Совет другу: не женись, пока не сделал всего, что мог", "не женись до тех пор"),
    ("Светлая петербургская ночь, второй час, он едет домой и не может заснуть", "бессумрачная ночь"),
]

SENTENCE = "Врач, после осмотра пациента, сказал, что его анализы готовы."
INSTRUCTION = "Отвечай только по-русски и в конце каждого ответа ставь слово КОНЕЦ."

_CACHE = {}                                       # модели и токенизаторы грузим один раз


# ------------------------------------------------------------------ корпус

def load_corpus() -> str:
    if not CORPUS.exists():
        urllib.request.urlretrieve(CORPUS_URL, CORPUS)   # скрипт скачивает корпус сам
    return CORPUS.read_text(encoding="utf-8")


def corpus_words(text: str) -> list:
    """Нижний регистр, «ё» -> «е», слова по регулярному выражению."""
    return WORD_RE.findall(text.lower().replace("ё", "е"))


def dump(name: str, obj) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ часть 1: свой BPE

def merge_symbols(sym: tuple, pair: tuple) -> tuple:
    """Сливает все (слева направо) вхождения пары соседних символов в один символ."""
    a, b = pair
    out, i = [], 0
    while i < len(sym):
        if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return tuple(out)


def train_bpe(word_freq: dict, num_merges: int):
    """Классический BPE: на каждом шаге сливаем самую частую пару соседних символов."""
    vocab = {tuple(w) + (EOW,): f for w, f in word_freq.items()}   # слово -> буквы + </w>
    merges, counts = [], []
    for _ in range(num_merges):
        pairs = Counter()
        for sym, f in vocab.items():
            for pair in zip(sym, sym[1:]):
                pairs[pair] += f                                    # частота пары во всём словаре
        if not pairs:
            break
        best = max(pairs, key=pairs.get)                            # при равенстве берётся пара, встреченная раньше
        merges.append(list(best))
        counts.append(pairs[best])
        vocab = {(merge_symbols(sym, best) if best[0] in sym else sym): f for sym, f in vocab.items()}
    return merges, counts


def bpe_encode(word: str, ranks: dict) -> list:
    """Разбиение слова: на каждом шаге сливается пара с наименьшим рангом (рано выученная)."""
    sym = list(word) + [EOW]
    while len(sym) > 1:
        ranked = [(ranks[p], i) for i, p in enumerate(zip(sym, sym[1:])) if p in ranks]
        if not ranked:
            break
        _, i = min(ranked)
        sym = sym[:i] + [sym[i] + sym[i + 1]] + sym[i + 2:]
    return sym


def part1_bpe(words: list, fast: bool) -> dict:
    freq = Counter(words)
    train_freq = {w: f for w, f in freq.items() if f >= MIN_FREQ}
    num_merges = 200 if fast else 1500
    merges, counts = train_bpe(train_freq, num_merges)

    ranks = {}
    for i, (a, b) in enumerate(merges):
        ranks.setdefault((a, b), i)
    encoded = {w: bpe_encode(w, ranks) for w in freq}               # разбиение каждого слова корпуса

    chars = sum(len(w) * f for w, f in freq.items())                # букв (без </w>)
    tokens = sum(len(encoded[w]) * f for w, f in freq.items())
    alphabet = {ch for w in freq for ch in w}
    result = {
        "num_merges": len(merges),
        "vocab_size": len(alphabet) + 1 + len(merges),              # буквы + </w> + слияния
        "merges": merges,
        "stats": {
            "chars": chars,
            "words": len(words),
            "tokens_bpe": tokens,
            "tokens_per_word": round(tokens / len(words), 4),
            "compression_vs_chars": round(chars / tokens, 4),
        },
        "examples": {w: bpe_encode(w, ranks) for w in ("наполеон", "сказала", "неудовольствие")},
    }
    dump("bpe.json", result)
    return {**result, "merge_counts": counts}                       # частоты слияний нужны только для показа


# ------------------------------------------------------------------ часть 2: токенизатор Qwen3 и бюджет контекста

def qwen_tokenizer():
    if "tok" not in _CACHE:
        from transformers import AutoTokenizer
        _CACHE["tok"] = AutoTokenizer.from_pretrained(QWEN)
    return _CACHE["tok"]


def count_tokens(text: str) -> int:
    """Число токенов Qwen3 в тексте (без служебных токенов)."""
    return len(qwen_tokenizer()(text, add_special_tokens=False)["input_ids"])


def tokenize_with_spans(text: str):
    """id токенов, их позиции в тексте и читаемые подписи (кусок исходного текста)."""
    enc = qwen_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"], [text[s:e] for s, e in enc["offset_mapping"]]


def part2_tokens(words: list) -> dict:
    ru_text = " ".join(words[500:600])                              # сто слов: с 500 по 600 по счету
    out = {"tokenizer": QWEN, "vocab_size": len(qwen_tokenizer()), "context_tokens": CONTEXT_TOKENS}
    for lang, text, n_words in (("ru", ru_text, 100), ("en", EN_PARAGRAPH, len(EN_PARAGRAPH.split()))):
        n_tokens = count_tokens(text)
        tpw = round(n_tokens / n_words, 4)
        out[lang] = {"words": n_words, "tokens": n_tokens, "tokens_per_word": tpw,
                     "words_in_context": int(CONTEXT_TOKENS / tpw)}
    out["example_ru"] = tokenize_with_spans(ru_text)[2][:40]       # подписи первых сорока токенов
    dump("tokens_per_word.json", out)
    return out


# ------------------------------------------------------------------ часть 3: векторы и поиск

def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def qwen_model():
    if "model" not in _CACHE:
        import torch
        from transformers import AutoModelForCausalLM
        try:
            m = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32, attn_implementation="eager")
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.float32, attn_implementation="eager")
        _CACHE["model"] = m.to(device()).eval()
    return _CACHE["model"]


def select_paragraphs(text: str, limit: int) -> list:
    """Абзацы по переводам строки, от 40 до 120 слов; сами абзацы хранятся как есть."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line and 40 <= len(WORD_RE.findall(line.lower())) <= 120:
            out.append({"id": len(out), "text": line})
            if len(out) == limit:
                break
    return out


def qwen_vectors(texts: list):
    """static: среднее входных эмбеддингов; contextual: среднее скрытых состояний последнего слоя."""
    import torch
    tok, model = qwen_tokenizer(), qwen_model()
    table = model.get_input_embeddings().weight
    static, contextual = [], []
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor([tok(t, add_special_tokens=False)["input_ids"]], device=device())
            static.append(table[ids[0]].mean(0))                    # таблица: контекста нет
            hidden = model.model(input_ids=ids, output_hidden_states=True).hidden_states[-1][0]
            contextual.append(hidden.mean(0))                       # те же токены после внимания
    return torch.stack(static), torch.stack(contextual)


def e5_vectors(texts: list, prefix: str):
    """multilingual-e5-small: среднее выходов по маске внимания, префикс query:/passage:."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if "e5" not in _CACHE:
        _CACHE["e5"] = (AutoTokenizer.from_pretrained(E5), AutoModel.from_pretrained(E5).to(device()).eval())
    tok, model = _CACHE["e5"]
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = tok([prefix + t for t in texts[i:i + 16]], padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device())
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            vecs.append((hidden * mask).sum(1) / mask.sum(1))
    return torch.cat(vecs)


def top_k(query_vec, matrix, k: int = K) -> list:
    """Номера k ближайших строк матрицы по косинусу, по убыванию близости."""
    import torch.nn.functional as F
    scores = F.normalize(matrix, dim=-1) @ F.normalize(query_vec, dim=-1)
    return scores.topk(k).indices.tolist()


def build_index(paragraphs: list) -> dict:
    texts = [p["text"] for p in paragraphs]
    static, contextual = qwen_vectors(texts)
    return {"static": static, "contextual": contextual, "e5": e5_vectors(texts, "passage: ")}


def search(query: str, k: int = K, method: str = "e5") -> list:
    """Поиск по абзацам: номера k ближайших абзацев. Индекс строится по artifacts/paragraphs.json при первом вызове."""
    if "index" not in _CACHE:
        paragraphs = json.loads((ARTIFACTS / "paragraphs.json").read_text(encoding="utf-8"))
        _CACHE["index"] = build_index(paragraphs)
    if method == "e5":
        q = e5_vectors([query], "query: ")[0]
    else:
        q = qwen_vectors([query])[0 if method == "static" else 1][0]
    return top_k(q, _CACHE["index"][method], k)


def part3_search(text: str, fast: bool) -> dict:
    paragraphs = select_paragraphs(text, 60 if fast else 500)
    dump("paragraphs.json", paragraphs)
    index = build_index(paragraphs)
    _CACHE["index"] = index

    questions = [q for q, _ in QUERIES]
    q_static, q_context = qwen_vectors(questions)
    q_vecs = {"static": q_static, "contextual": q_context, "e5": e5_vectors(questions, "query: ")}

    items = []
    for n, (query, phrase) in enumerate(QUERIES):
        answer_id = next(p["id"] for p in paragraphs if phrase in p["text"])   # первый абзац с фразой
        items.append({"query": query, "answer_phrase": phrase, "answer_id": answer_id,
                      "top": {m: top_k(q_vecs[m][n], index[m]) for m in index}})
    recall = {m: round(sum(it["answer_id"] in it["top"][m] for it in items) / len(items), 4) for m in index}
    result = {"k": K, "n_paragraphs": len(paragraphs), "queries": items, "recall_at_k": recall}
    dump("search.json", result)
    return result


# ------------------------------------------------------------------ часть 4: внимание

def span(offsets: list, text: str, word: str) -> list:
    """Номера токенов, покрывающих слово."""
    start = text.find(word)
    end = start + len(word)
    return [i for i, (s, e) in enumerate(offsets) if e > start and s < end]


def attentions(ids: list):
    """Карты внимания всех слоёв: [слои, головы, n, n]; строка i - распределение внимания токена i."""
    import torch
    with torch.no_grad():
        out = qwen_model()(input_ids=torch.tensor([ids], device=device()), output_attentions=True)
    return torch.stack([a[0].float().cpu() for a in out.attentions])


def last_row_attention(ids: list):
    """Только строка последнего токена: [слои, головы, n] (экономит память на длинных входах)."""
    import torch
    with torch.no_grad():
        out = qwen_model()(input_ids=torch.tensor([ids], device=device()), output_attentions=True)
    return torch.stack([a[0, :, -1, :].float().cpu() for a in out.attentions])


def part4a_attention() -> dict:
    ids, offsets, labels = tokenize_with_spans(SENTENCE)
    attn = attentions(ids)
    pronoun = span(offsets, SENTENCE, "его")
    antecedent = span(offsets, SENTENCE, "пациента")
    rows = attn[:, :, pronoun[-1], :]                               # строка последнего токена местоимения
    on_antecedent = rows[:, :, antecedent].sum(-1)                  # доля на токены слова «пациента»
    layer, head = divmod(int(on_antecedent.argmax()), rows.shape[1])
    weights = rows[layer, head].tolist()
    result = {"sentence": SENTENCE, "tokens": labels, "pronoun_pos": pronoun, "antecedent_pos": antecedent,
              "best_layer": layer, "best_head": head, "weights": weights,
              "share_on_antecedent": round(float(on_antecedent[layer, head]), 4)}
    dump("attention_map.json", result)

    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 3))
    ax = fig.subplots()
    ax.bar(range(len(labels)), weights, color="#333333")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(f"На что смотрит «его»: слой {layer}, голова {head}")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "attention_map.png", dpi=150)
    return result


def part4b_dilution(words: list, fast: bool) -> list:
    tok = qwen_tokenizer()
    prefix = tok(INSTRUCTION + "\n", add_special_tokens=False)["input_ids"]
    filler_ids = tok(" ".join(words[20000:30000]), add_special_tokens=False)["input_ids"]
    rows = []
    for n_ctx in ((64, 256, 768) if fast else (128, 512, 1024)):
        ids = prefix + filler_ids[: n_ctx - len(prefix)]            # весь вход ровно n_ctx токенов
        row = last_row_attention(ids).mean((0, 1))                  # последний токен, среднее по слоям и головам
        n_instr = len(prefix) - 1                                   # токены инструкции без первого
        rows.append({"context_len": len(ids), "instruction_len": n_instr,
                     "share_on_instruction": round(float(row[1:len(prefix)].sum()), 6),
                     "sink_share": round(float(row[0]), 6),         # первый токен - сток внимания
                     "uniform_share": round(n_instr / len(ids), 6)})
    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "dilution.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# ------------------------------------------------------------------ запуск

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    text = load_corpus()
    words = corpus_words(text)
    print("часть 1: BPE")
    part1_bpe(words, args.fast)
    print("часть 2: токенизатор Qwen3")
    part2_tokens(words)
    print("часть 3: векторы и поиск")
    print("  recall@5:", part3_search(text, args.fast)["recall_at_k"])
    print("часть 4а: внимание местоимения")
    part4a_attention()
    print("часть 4б: размывание")
    part4b_dilution(words, args.fast)
    print("готово: артефакты в", ARTIFACTS)


if __name__ == "__main__":
    main()
