import argparse, csv, importlib.util, json, math, re, sysconfig, time
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
import run

HERE, ART = run.HERE, run.ART
QWEN = "Qwen/Qwen3-0.6B-Base"
QPROMPT = "Князь Андрей вошел в комнату и"
CODE = HERE / "code_corpus.txt"  # корпус исходников, в репозиторий не кладём
CPROMPT = "def main():\n    "
PRE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|\n| +|[^\w\s]|\s")  # имена, числа, перевод строки, отступ, знаки
dev = "cuda" if torch.cuda.is_available() else "cpu"
ART.mkdir(exist_ok=True)
dump = lambda name, obj: (ART / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def split_words():  # слова корпуса и граница отложенной части
    words = re.findall(run.WORD, run.CORPUS.read_text(encoding="utf-8", errors="ignore").lower().replace("ё", "е"))
    return words, int(len(words) * 0.9)


def load_bpe():
    return run.BPE(HERE / ".." / "lab02" / "artifacts" / "bpe.json")


def load_tiny():  # обученная модель из tiny.pt
    bpe = load_bpe()
    model = run.TinyLM(len(bpe.itos)).to(dev)
    model.load_state_dict(torch.load(HERE / "tiny.pt", map_location=dev))
    return model.eval(), bpe


def load_qwen(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN), AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).to(device).eval()


def fit(model, train, val, steps, batch, every, on_step=None):  # тот же цикл, что в run.py
    opt = torch.optim.AdamW(model.parameters(), lr=run.LR)
    g = torch.Generator().manual_seed(run.SEED)
    vs = torch.linspace(0, len(val) - run.CONTEXT - 1, 20).long().to(dev)
    V, acc = model.head.out_features, []
    rows = [(0, None, run.eval_loss(model, val, vs))]
    if on_step:
        on_step(0)
    for step in range(1, steps + 1):
        starts = torch.randint(0, len(train) - run.CONTEXT - 1, (batch,), generator=g).to(dev)
        x, y = run.windows(train, starts)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        acc.append(loss.item())
        if step % every == 0:
            rows.append((step, sum(acc) / len(acc), run.eval_loss(model, val, vs)))
            acc = []
            print(step, round(rows[-1][1], 4), round(rows[-1][2], 4), flush=True)
        if on_step:
            on_step(step)
    return rows


def write_curve(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "train_loss", "val_loss"])
        for s, tr, vl in rows:
            w.writerow([s, "" if tr is None else f"{tr:.4f}", f"{vl:.4f}"])


# Бонус 1. eval.json
def bigram_nats(ws, path, alpha=0.01):  # сглаживание Лапласа как в lab01 (c(v,w)+a)/(c(v)+a*V)
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    uni, bi = d["unigram"], d["bigram"]
    V, total, unseen = len(uni), 0.0, 0
    for v, w in zip(ws, ws[1:]):
        row = bi.get(v, {})
        unseen += v not in bi  # контекст не встречался - формула даёт равномерное 1/V
        total -= math.log((row.get(w, 0) + alpha) / (sum(row.values()) + alpha * V))
    return total, len(ws) - 1, unseen


@torch.no_grad()
def tiny_nats(model, ids):  # окно 128, шаг 64, каждый токен оценивается один раз, контекст не меньше 64
    x, C, total, pos = torch.tensor(ids, device=dev), run.CONTEXT, 0.0, 1
    while pos < len(x):
        end = min(pos + C // 2, len(x))  # оцениваем токены pos..end-1
        start = max(0, end - 1 - C)  # вход до 128 токенов перед end
        logits = model(x[None, start:end - 1])[0]
        k = end - pos
        total += F.cross_entropy(logits[-k:], x[end - k:end], reduction="sum").item()
        pos = end
    return total, len(x) - 1


@torch.no_grad()
def qwen_nats(text):  # весь текст одним проходом, первый токен не оценивается
    tok, model = load_qwen(dev)
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
    logits, tgt, total = model(input_ids=ids).logits[0, :-1], ids[0, 1:], 0.0
    for i in range(0, len(tgt), 512):  # кусками, чтобы не держать лишние копии логитов
        total += F.cross_entropy(logits[i:i + 512], tgt[i:i + 512], reduction="sum").item()
    return total, len(tgt)


def do_eval(a):
    words, cut = split_words()
    ws = words[cut:cut + 2000]  # первые 2000 слов отложенной части
    text = " ".join(ws)
    model, bpe = load_tiny()
    out = {"words": len(ws)}
    bn, bt, unseen = bigram_nats(ws, a.bigram)
    tn, tt = tiny_nats(model, bpe.encode(text))
    qn, qt = qwen_nats(text)
    for name, n, t, extra in (("bigram", bn, bt, {"alpha": 0.01}), ("tiny", tn, tt, {}), ("qwen", qn, qt, {"model": QWEN})):
        out[name] = {"tokens": t, "nats_total": round(n, 2), "nats_per_word": round(n / len(ws), 4),
                     "ppl_per_word": round(math.exp(n / len(ws)), 1), **extra}
    print("биграммы: слов с контекстом вне словаря", unseen)
    dump("eval.json", out)


# Бонус 2. generate с кэшем, greedy_check.json, speed.csv
@torch.no_grad()
def cached_greedy(model, ids, n, eos=None):  # forward с кэшем, дальше только последний токен
    out, new = model(input_ids=ids, use_cache=True), []
    for _ in range(n):
        nxt = int(out.logits[0, -1].argmax())  # жадный выбор
        new.append(nxt)
        if nxt == eos:
            break
        out = model(input_ids=torch.tensor([[nxt]], device=ids.device), past_key_values=out.past_key_values, use_cache=True)
    return new


@torch.no_grad()
def full_greedy(model, ids, n):  # без кэша, на каждом шаге вся последовательность заново
    for _ in range(n):
        ids = torch.cat([ids, model(input_ids=ids).logits[0, -1].argmax().view(1, 1)], dim=1)
    return ids


def do_greedy(a):  # на CPU в float32, как проверяет тест
    tok, model = load_qwen("cpu")
    ids = tok(QPROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ours = cached_greedy(model, ids, 30, tok.eos_token_id)
    with torch.no_grad():
        hf = model.generate(ids, do_sample=False, max_new_tokens=30, pad_token_id=tok.eos_token_id)[0, ids.size(1):].tolist()
    print("совпадение:", ours == hf)
    dump("greedy_check.json", {"model": QWEN, "prompt": QPROMPT, "prompt_ids": ids[0].tolist(), "ours_ids": ours,
                               "hf_ids": hf, "match": ours == hf, "text": tok.decode(ours)})


def do_speed(a):  # на GPU, 64 новых токена
    tok, model = load_qwen(dev)
    ids = tok(QPROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
    sync = lambda: torch.cuda.synchronize() if dev == "cuda" else None
    cached_greedy(model, ids, 4); full_greedy(model, ids, 4)  # прогрев
    rows = []
    for mode in ("cache", "no_cache"):
        sync(); t0 = time.time()
        cached_greedy(model, ids, 64) if mode == "cache" else full_greedy(model, ids, 64)
        sync(); sec = round(time.time() - t0, 3)
        rows.append({"mode": mode, "new_tokens": 64, "seconds": sec, "tokens_per_second": round(64 / sec, 2)})
        print(rows[-1])
    with open(ART / "speed.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# Бонус 3. Масштабирование
def do_scaling(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    words, cut = split_words()
    bpe = load_bpe()
    train = torch.tensor([i for w in words[:cut] for i in bpe.word(w)], device=dev)
    val = torch.tensor([i for w in words[cut:] for i in bpe.word(w)], device=dev)
    res = {}
    for width in (64, 128, 256):
        run.D_MODEL, run.D_FF = width, 4 * width  # TinyLM читает эти константы при создании
        torch.manual_seed(run.SEED)
        model = run.TinyLM(len(bpe.itos)).to(dev)
        n = sum(p.numel() for p in model.parameters())
        print("ширина", width, "параметров", n)
        res[width] = (n, fit(model, train, val, 2000, 32, 100))
    run.D_MODEL, run.D_FF = 128, 512
    steps = [r[0] for r in res[64][1]]
    with open(ART / "scaling.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"{k}_{w_}" for w_ in res for k in ("train_loss", "val_loss")])
        for i, s in enumerate(steps):
            row = [s]
            for w_ in res:
                tr, vl = res[w_][1][i][1:]
                row += ["" if tr is None else f"{tr:.4f}", f"{vl:.4f}"]
            w.writerow(row)
    dump("scaling.json", {str(k): {"n_params": v[0], "final_val_loss": round(v[1][-1][2], 4),
                                   "final_train_loss": round(v[1][-1][1], 4)} for k, v in res.items()})
    fig, ax = plt.subplots(figsize=(7, 4))
    for w_, (n, rows) in res.items():
        ln, = ax.plot(steps, [r[2] for r in rows], label=f"ширина {w_}, val ({n / 1e6:.2f} млн)")
        ax.plot(steps[1:], [r[1] for r in rows[1:]], "--", color=ln.get_color(), alpha=0.6, label=f"ширина {w_}, train")
    ax.set_xlabel("шаг"); ax.set_ylabel("лосс, нат"); ax.set_ylim(3, 7.5); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ART / "scaling.png", dpi=150)


# Бонус 4. Другой корпус
class CodeBPE(run.BPE):  # тот же BPE, но алфавит из корпуса и "слова" кода
    def __init__(self, merges, chars):
        self.merges = [tuple(m) for m in merges]
        self.rank = {}
        for i, m in enumerate(self.merges):
            self.rank.setdefault(m, i)
        self.itos = list(chars) + [run.EOW]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        for x, y in self.merges:
            if x + y not in self.stoi:
                self.stoi[x + y] = len(self.itos)
                self.itos.append(x + y)
        self.cache = {}

    def encode(self, text):
        return [i for w in PRE.findall(text) for i in self.word(w)]

    def decode(self, ids):  # "слова" кода склеиваются без пробелов, маркер конца слова убираем
        return "".join(self.itos[i] for i in ids).replace(run.EOW, "")


def do_code(a):
    if not CODE.exists():  # собираем 4 МБ исходников, только ASCII
        parts, size = [], 0
        for p in sorted(Path(a.src).rglob("*.py")):
            if {"test", "tests", "idlelib", "site-packages", "__pycache__"} & set(p.parts):
                continue
            t = p.read_text(encoding="utf-8", errors="ignore").replace("\r", "").encode("ascii", "ignore").decode()
            parts.append(t); size += len(t)
            if size >= 4_000_000:
                break
        CODE.write_text("".join(parts), encoding="utf-8")
    text = CODE.read_text(encoding="utf-8")
    pieces = PRE.findall(text)
    cut = int(len(pieces) * 0.9)
    spec = importlib.util.spec_from_file_location("lab02_run", HERE / ".." / "lab02" / "run.py")
    lab02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab02)
    freq = {w: f for w, f in Counter(pieces[:cut]).items() if f >= lab02.MIN_FREQ}  # BPE заново, по обучающей части
    merges, _ = lab02.train_bpe(freq, 1500)
    bpe = CodeBPE(merges, sorted(set(text)))
    train = torch.tensor([i for w in pieces[:cut] for i in bpe.word(w)], device=dev)
    val = torch.tensor([i for w in pieces[cut:] for i in bpe.word(w)], device=dev)
    print("символов", len(text), "словарь", len(bpe.itos), "токенов train/val", len(train), len(val))
    torch.manual_seed(run.SEED)
    model = run.TinyLM(len(bpe.itos)).to(dev)
    gen, samples = run.make_generate(model, bpe), []

    def snap(step):
        if step in (0, 50, 100, 200, 500, 1000, 2000):
            torch.manual_seed(run.SEED)
            samples.append({"step": step, "text": bpe.decode(bpe.encode(CPROMPT) + gen(CPROMPT, 150, temperature=0.8))})

    rows = fit(model, train, val, 2000, 32, 100, snap)
    write_curve(ART / "code_curve.csv", rows)
    dump("code_samples.json", {"prompt": CPROMPT, "temperature": 0.8, "chars": len(text), "vocab_size": len(bpe.itos),
                               "bpe_merges": len(merges), "train_tokens": len(train), "val_tokens": len(val), "samples": samples})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["eval", "greedy", "speed", "scaling", "code"])
    ap.add_argument("--bigram", default=str(HERE / ".." / "lab01" / "weights" / "bigram_tolstoy.json"))
    ap.add_argument("--src", default=sysconfig.get_paths()["stdlib"])
    a = ap.parse_args()
    {"eval": do_eval, "greedy": do_greedy, "speed": do_speed, "scaling": do_scaling, "code": do_code}[a.what](a)
