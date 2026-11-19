import argparse, csv, json, math, re, time, urllib.request
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"
URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
WORD = r"[а-я]+(?:-[а-я]+)*"
EOW = "</w>"
ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя-"  # 32 буквы без "ё" и дефис
PROMPT = "князь андрей"
SEED, D_MODEL, N_LAYERS, N_HEADS, D_FF, CONTEXT, LR = 0, 128, 4, 4, 512, 128, 1e-3


class BPE:
    def __init__(self, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.merges = [tuple(m) for m in d["merges"]]  # порядок = ранг
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.itos = list(ALPHABET) + [EOW]  # базовые символы
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        for a, b in self.merges:  # токен слияния получает номер по порядку
            if a + b not in self.stoi:
                self.stoi[a + b] = len(self.itos)
                self.itos.append(a + b)
        self.cache = {}

    def word(self, w):
        if w not in self.cache:
            s = list(w) + [EOW]
            while len(s) > 1:
                r, p = min((self.rank.get(q, 1e9), q) for q in zip(s, s[1:]))  # лучшая пара
                if r == 1e9:
                    break
                t, i = [], 0
                while i < len(s):
                    if i < len(s) - 1 and (s[i], s[i + 1]) == p:
                        t.append(s[i] + s[i + 1]); i += 2  # склеиваем пару
                    else:
                        t.append(s[i]); i += 1
                s = t
            self.cache[w] = [self.stoi[x] for x in s]
        return self.cache[w]

    def encode(self, text):
        return [i for w in re.findall(WORD, text.lower().replace("ё", "е")) for i in self.word(w)]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids).replace(EOW, " ").strip()


class Attention(nn.Module):  # внимание с маской, несколько голов
    def __init__(self, d, h, ctx):
        super().__init__()
        self.h = h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))  # Wq, Wk, Wv и выходная
        self.register_buffer("mask", torch.tril(torch.ones(ctx, ctx, dtype=torch.bool)))  # нижний треугольник

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (f(x).view(B, T, self.h, D // self.h).transpose(1, 2) for f in (self.q, self.k, self.v))  # (B,h,T,dh)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h)  # оценки
        a = a.masked_fill(~self.mask[:T, :T], float("-inf"))  # маска будущего
        a = F.softmax(a, dim=-1)  # веса внимания
        return self.o((a @ v).transpose(1, 2).reshape(B, T, D))  # взвешенная сумма, склейка голов


class Block(nn.Module):
    def __init__(self, d, h, ff, ctx):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = Attention(d, h, ctx)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        x = x + self.att(self.ln1(x))  # pre-LN, остаточная связь
        return x + self.ff(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)  # токены
        self.pos = nn.Embedding(CONTEXT, D_MODEL)  # позиции
        self.blocks = nn.ModuleList(Block(D_MODEL, N_HEADS, D_FF, CONTEXT) for _ in range(N_LAYERS))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)  # без связывания весов
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)  # малые веса: старт около ln V
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def windows(data, starts):  # окна из потока: x и сдвинутые на 1 цели y
    w = data[starts[:, None] + torch.arange(CONTEXT + 1, device=data.device)]
    return w[:, :-1], w[:, 1:]


@torch.no_grad()
def eval_loss(model, data, starts):
    x, y = windows(data, starts)
    return F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1)).item()


def make_generate(model, bpe):
    @torch.no_grad()
    def generate(prompt, max_new_tokens, temperature=1.0, top_p=None):
        dev = next(model.parameters()).device
        ids = bpe.encode(prompt)  # затравка своим BPE
        n0 = len(ids)
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-CONTEXT:]], device=dev)  # последние 128 токенов
            logits = model(x)[0, -1]  # логиты последней позиции
            if temperature == 0:
                nxt = logits.argmax().item()  # жадный выбор
            else:
                logits = logits / temperature
                if top_p is not None:
                    sl, si = torch.sort(logits, descending=True)
                    pr = F.softmax(sl, dim=-1)
                    sl[(pr.cumsum(-1) - pr) >= top_p] = float("-inf")  # граничный токен остаётся
                    nxt = si[torch.multinomial(F.softmax(sl, dim=-1), 1)].item()
                else:
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            ids.append(nxt)
        return ids[n0:]  # только новые токены
    return generate


def first_repeat(ids, n=4):  # начало первой четвёрки, уже встречавшейся раньше
    seen = set()
    for i in range(n, len(ids) + 1):
        g = tuple(ids[i - n:i])
        if g in seen:
            return i - n
        seen.add(g)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--bpe", default=str(HERE / ".." / "lab02" / "artifacts" / "bpe.json"))
    a = ap.parse_args()
    steps, batch, every = (200, 16, 25) if a.fast else (2000, 32, 100)
    sample_steps = [0, 50, 100, steps] if a.fast else [0, 100, 500, steps]
    gen_n = 40 if a.fast else 80
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    ART.mkdir(exist_ok=True)

    bpe = BPE(a.bpe)  # данные
    if not CORPUS.exists():
        CORPUS.write_bytes(urllib.request.urlopen(URL).read())
    words = re.findall(WORD, CORPUS.read_text(encoding="utf-8", errors="ignore").lower().replace("ё", "е"))
    cut = int(len(words) * 0.9)  # последние 10 % слов отложены
    train = torch.tensor([i for w in words[:cut] for i in bpe.word(w)], device=dev)
    val = torch.tensor([i for w in words[cut:] for i in bpe.word(w)], device=dev)
    V = len(bpe.itos)

    model = TinyLM(V).to(dev)  # модель
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = make_generate(model, bpe)
    g = torch.Generator().manual_seed(SEED)  # отдельный генератор для окон
    val_starts = torch.linspace(0, len(val) - CONTEXT - 1, 20).long().to(dev)  # 20 фиксированных окон

    samples = []
    def take_sample(step):
        torch.manual_seed(SEED)
        samples.append({"step": step, "text": bpe.decode(bpe.encode(PROMPT) + gen(PROMPT, 60, temperature=0.8))})

    curve = [[0, None, eval_loss(model, val, val_starts)]]  # шаг 0: train_loss подставляется на первом батче
    take_sample(0)
    acc, t0 = [], time.time()
    for step in range(1, steps + 1):
        starts = torch.randint(0, len(train) - CONTEXT - 1, (batch,), generator=g).to(dev)
        x, y = windows(train, starts)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        if step == 1:
            curve[0][1] = loss.item()  # лосс до первого обновления
        opt.zero_grad(); loss.backward(); opt.step()
        acc.append(loss.item())
        if step % every == 0:
            curve.append([step, sum(acc) / len(acc), eval_loss(model, val, val_starts)])
            acc = []
            print(step, round(curve[-1][1], 4), round(curve[-1][2], 4), flush=True)
        if step in sample_steps:
            take_sample(step)
    if dev == "cuda":
        torch.cuda.synchronize()
    minutes = (time.time() - t0) / 60

    runs = []  # генерация руками
    for name, t, p in [("greedy", 0.0, None), ("t0.7", 0.7, None), ("t1.5", 1.5, None), ("top_p0.9", 1.0, 0.9)]:
        torch.manual_seed(SEED)
        ids = gen(PROMPT, gen_n, t, p)
        r = {"name": name, "temperature": t, "top_p": p, "text": bpe.decode(ids), "ids": ids, "n_tokens": len(ids)}
        if name == "greedy":
            r["loop_start"] = first_repeat(ids)
        runs.append(r)

    dump = lambda name, obj: (ART / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    dump("tiny_config.json", {"vocab_size": V, "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
                              "d_ff": D_FF, "context": CONTEXT, "n_params": n_params, "steps": steps, "batch": batch,
                              "lr": LR, "seed": SEED, "train_tokens": len(train), "val_tokens": len(val),
                              "bpe_merges": len(bpe.merges), "train_minutes": round(minutes, 1), "device": dev})
    with open(ART / "train_curve.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "train_loss", "val_loss"])
        for s, tr, vl in curve:
            w.writerow([s, f"{tr:.4f}", f"{vl:.4f}"])
    dump("samples.json", {"prompt": PROMPT, "temperature": 0.8, "samples": samples})
    dump("generation.json", {"model": "tiny", "prompt": PROMPT, "max_new_tokens": gen_n, "runs": runs})
    torch.save(model.state_dict(), HERE / "tiny.pt")


if __name__ == "__main__":
    main()
