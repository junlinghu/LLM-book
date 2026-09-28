# 5.8 Tokenization Artifacts and Failure Modes

A tokenizer is trained once, by counting, and then frozen. It never learns from the model's mistakes. Section 1 argued that this gap produces a family of odd model behaviors, and this section examines them one by one: trouble with spelling, with numbers, with languages other than English, with whitespace and code, with the boundary at the end of a prompt, and with tokens the model never learned. It then covers two subtler consequences, the fact that one string can have many token sequences and the fact that losses from models with different tokenizers cannot be compared directly, and ends with models that avoid a learned vocabulary altogether.

The examples use three OpenAI tokenizers from the `tiktoken` library: GPT-2's, `cl100k_base`, and `o200k_base`. The code that produces every result is collected in the appendix at the end of the section.

## Spelling and character-level tasks

A model sees token IDs, not letters. Whatever it knows about the spelling of a token, it had to infer from indirect evidence during training: seeing the word spelled out letter by letter somewhere, seeing it misspelled, seeing rhymes. How a word is tokenized depends on its context, and in particular on whether it follows a space:

| Tokenizer | "strawberry" | " strawberry" |
|---|---|---|
| GPT-2 | `st` `raw` `berry` | `␣strawberry` |
| `cl100k_base` | `str` `aw` `berry` | `␣strawberry` |
| `o200k_base` | `st` `raw` `berry` | `␣strawberry` |

In the middle of a sentence, " strawberry" is a single token in all three tokenizers. Asking how many times the letter "r" occurs in it asks the model to report on the internal contents of one opaque symbol. At the start of a line, the same word is two or three different tokens, none of which is " strawberry". The model has to know that all of these token sequences spell the same word.

Tasks that operate on characters, such as counting letters, reversing a word, finding rhymes, solving anagrams, or following instructions like "write a sentence with no letter e", are hard for this reason, not because they require deep reasoning. Models often do better when asked to spell the word out first, one letter at a time, because each letter then becomes its own token that the model can attend to.

## Numbers and arithmetic

Numbers show the problem at its sharpest. GPT-2's tokenizer lets BPE learn any digit string that was frequent in its training data, so the way a number is split depends on which substrings happened to be common:

| Number | GPT-2 tokens |
|---|---|
| ␣1234 | `␣12` `34` |
| ␣1235 | `␣12` `35` |
| ␣2023 | `␣20` `23` |
| ␣2024 | `␣2024` |
| ␣7777 | `␣7` `777` |
| ␣3.14159 | `␣3` `.` `14` `159` |
| ␣100000 | `␣100` `000` |

The year " 2024" is one token while " 2023" is two. " 7777" splits after the first digit, " 1234" in the middle. To add two numbers, a model must line up digits of equal place value, and here the tokens do not line up: the same position in two numbers can fall in different tokens, and a single token can mix hundreds and tens.

We can count how many different ways each tokenizer splits the numbers from 1 to 10,000 (each preceded by a space), recording each split as the number of digits in each piece (Appendix A.2). GPT-2 uses eleven different patterns; the most common are 2+2 (4,680 numbers), 1+3 (3,417), 3+1 (513), and 1+2 (470). Four-digit numbers alone are split as 2+2, 1+3, 3+1, or other ways, depending on the number.

`cl100k_base` and `o200k_base` use only five patterns, and all 9,000 four-digit numbers split as 3+1. The reason is a pre-tokenization rule (Section 6) that chunks digits left to right into groups of at most three. The result is consistent, but grouped from the left, while place value is aligned from the right. The original LLaMA models went further and split every number into single digits (Touvron et al. 2023).

Singh and Strouse (2024) studied the effect of this choice on arithmetic in GPT-3.5 and GPT-4, whose tokenizer groups digits in threes from the left. Forcing right-to-left grouping at inference time, by writing numbers with commas as thousands separators so that the groups align with place value, greatly improved accuracy on addition. They also found that errors under left-to-right grouping followed systematic patterns, and that the gap between the two directions shrank for larger models. The general lesson is that digit grouping is an inductive bias the tokenizer imposes, and a poor choice makes arithmetic harder to learn.

## Multilingual cost

Section 7 measured the same sentence in ten languages and found that GPT-2's tokenizer needs nearly five times as many tokens for Hindi as for English, and even `cl100k_base` needs more than three times as many. Petrov et al. (2023) documented such disparities systematically, with differences of up to 15 times in tokenized length for the same content in some cases.

This is not a cosmetic problem. A speaker of a language that tokenizes poorly pays more for every request to a service priced per token, waits longer for each answer because generation takes more steps, and gets less out of the context window, since the same content fills more of it. The model also sees less text in that language per token of training compute. The cause is the tokenizer's training data: merges are spent on whatever is frequent there, and a tokenizer trained mostly on English text learns few tokens for other scripts. Larger vocabularies trained on more balanced multilingual data, such as `o200k_base` in the same measurement, narrow the gap considerably.

## Whitespace and code

In GPT-2-style tokenizers, a space is part of the token that follows it. So the same word has different tokens with and without a leading space, and with different capitalization. In GPT-2's tokenizer, "hello" is ID 31373, " hello" is 23748, "Hello" is 15496, " Hello" is 18435, and " HELLO" is two tokens, [47899, 46].

Four spellings of the same word are four unrelated IDs, and the all-caps version needs two tokens. The model must learn separately, from data, that these are related. Most of the time it does, because each common form appears often, but for rare words the variants may be poorly learned.

Code stresses whitespace handling further. Python uses indentation for structure, and other languages use it heavily by convention. GPT-2's tokenizer has no tokens for runs of spaces. The line "return x", indented by eight spaces, becomes nine tokens in GPT-2's tokenizer (seven single spaces, then ` return` and ` x`) but only three in `cl100k_base` (seven spaces as one token, then ` return` and ` x`).

GPT-2 spends seven tokens on indentation before the eighth space attaches to " return"; `cl100k_base` uses one token for the seven spaces. Section 7 found that this kind of difference makes `cl100k_base` about twice as efficient as GPT-2's tokenizer on Python source. Tokenizers intended for code add tokens for common runs of spaces and tabs.

## Prompt boundary effects

Because spaces attach to the *following* token, a prompt that ends with a space puts the model in an unusual position. In GPT-2's tokenizer, "The capital of France is" ends with the tokens ` France` and ` is` (IDs 4881 and 318). Add a trailing space, and the prompt ends with ` is` and a lone ` ` (IDs 318 and 220).

Without the trailing space, the natural continuation is the single token " Paris", with its space included. With the trailing space, the prompt ends in a lone space token (ID 220), and the model must continue with "Paris" without a leading space. In ordinary text a word almost never follows a lone space token, because the tokenizer would have attached the space to the word. The model is being asked to continue from a token sequence it rarely saw in training, and its predictions get worse. The same thing happens when a prompt ends in the middle of a word, or in the middle of a multi-byte character.

The practical rules are to avoid trailing spaces in prompts and, for completion tasks, to end prompts at natural token boundaries. Some libraries implement a fix, sometimes called **token healing**: remove the last token or tokens of the prompt, and then constrain the model's first generated token to begin with the removed text, so the model can choose the natural tokenization itself.

## Glitch tokens

The tokenizer and the model are usually trained on different data. The tokenizer may be trained on one sample, the model on a larger or differently filtered corpus. A string that was frequent in the tokenizer's data gets its own token, but if that string is rare or absent in the model's training data, the token's embedding row is almost never updated. Such tokens are called **under-trained** or **glitch tokens**.

The best-known example is " SolidGoldMagikarp", which is a single token in GPT-2's vocabulary, ID 43453.

The string is a username, and fragments of usernames are a common source of glitch tokens: they appeared often enough in some scraped web data to earn tokens, and then were rare or missing in the data used to train the model. Prompts containing such tokens have been observed to induce unwanted behavior; a model may, for example, be unable to repeat the token back when asked to.

Land and Bartolo (2024) developed methods for finding under-trained tokens automatically. They combine an analysis of the tokenizer (for example, whether a token can even be produced by encoding any text), indicators computed from the model's weights (under-trained tokens tend to have unusual embedding and output-layer rows, because those rows were barely touched by training), and prompts that test whether the model can repeat a candidate token exactly. They found under-trained tokens across a wide range of models, including many fragments of usernames, some non-English tokens, and, in some models, ASCII control characters such as the tab. The remedies follow from the cause: train the tokenizer on data drawn from the same distribution as the model's training data, check vocabulary utilization before training (Section 7), and remove or re-examine tokens that never occur.

## One string, many token sequences

Encoding is deterministic: a string always produces the same token IDs, the **canonical** tokenization. But decoding is many-to-one. Many token sequences decode to the same string. In GPT-2's tokenizer, " hello" encodes to the single token [23748], but the sequence [932, 5439], " hel" followed by "lo", decodes to exactly the same string. The tokenizer would never produce that second sequence. During training, the model saw only canonical tokenizations, so it has learned to assign very low probability to non-canonical ones.

This matters in two situations. First, the probability of a *string* under the model is, strictly, the sum of the probabilities of all token sequences that decode to it. In practice, the canonical sequence dominates, and scoring a string by the probability of its canonical tokenization is standard, but it is an approximation. Second, when an application constrains generation, for example forcing the output to match a grammar or a JSON schema by masking out tokens that would break it, the constraints can push the model toward non-canonical sequences it has never seen, degrading quality. Well-designed constrained decoders take the tokenization into account.

## Comparing losses across tokenizers

The training loss of a language model is the average negative log-likelihood per token. That makes it a statement about the model *and* its tokenizer. A tokenizer with longer tokens has fewer, harder-to-predict tokens per text, so its per-token loss is higher even if the model is equally good at predicting the text. Per-token loss, and perplexity, which is its exponential, cannot be compared between models with different tokenizers.

The fix is to normalize by a unit that does not depend on the tokenizer, such as bytes or characters of the original text. If a model's total negative log-likelihood on a text is $`\mathcal{L}_{\text{total}}`$ nats (the sum over all $`N_{\text{tokens}}`$ tokens), and the text is $`N_{\text{bytes}}`$ bytes long, then its **bits per byte** is

```math
\text{BPB} = \frac{\mathcal{L}_{\text{total}}}{N_{\text{bytes}} \ln 2} = \frac{N_{\text{tokens}}}{N_{\text{bytes}}} \cdot \frac{\bar{\mathcal{L}}_{\text{token}}}{\ln 2},
```

where $`\bar{\mathcal{L}}_{\text{token}}`$ is the average loss per token in nats. The first factor on the right is the inverse of the tokenizer's compression. Two models with different tokenizers can be compared fairly by their bits per byte on the same text.

## Doing without a learned vocabulary

Since so many problems come from the tokenizer, why not remove it? **Byte-level models** read and write raw UTF-8 bytes, with a vocabulary of 256 plus a few special tokens. ByT5 (Xue et al. 2022) showed that a standard Transformer, with minimal modifications, can be trained on byte sequences and be competitive with a comparable token-level model, and that byte-level models are significantly more robust to noise, such as typos, and better at tasks that depend on spelling and pronunciation.

The price is sequence length. The held-out portion of tiny Shakespeare is 111,540 bytes but only 36,059 GPT-2 tokens, so a byte-level model would process about three times as many positions for the same text, and more for scripts that need several bytes per character. ByT5's authors characterized this tradeoff in parameters, training compute, and inference speed. Research on models that group bytes into larger units dynamically, inside the model, continues, but as of this writing nearly all widely used LLMs still rely on a learned subword vocabulary.

## Key takeaways

- Models see opaque token IDs, so character-level tasks such as counting letters are hard, and the same word can have different tokens depending on the preceding space.
- Inconsistent digit grouping makes arithmetic harder; newer tokenizers group digits in threes or split them individually, and right-to-left grouping aligned with place value can help (Singh and Strouse 2024).
- Tokenizers trained mostly on English make other languages much more expensive in tokens, with consequences for cost, latency, and effective context.
- Leading spaces, capitalization, and indentation all change tokenization; code needs tokens for runs of whitespace.
- A prompt ending in a space or mid-word asks the model to continue from an unusual token sequence; avoid it or use token healing.
- Glitch tokens are tokens that were frequent in the tokenizer's data but nearly absent from the model's, and can be detected from the tokenizer, the model's weights, and prompting (Land and Bartolo 2024).
- Many token sequences decode to the same string, but the model has only seen canonical ones; per-token losses are not comparable across tokenizers, so use bits per byte.
- Byte-level models avoid a learned vocabulary at the cost of much longer sequences.

## Further reading

Land, Sander, et al. "Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models." In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 2024. https://arxiv.org/abs/2405.05417.

Petrov, Aleksandar, et al. "Language Model Tokenizers Introduce Unfairness Between Languages." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.15425.

Singh, Aaditya K., et al. "Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs." arXiv preprint arXiv:2402.14903, 2024. https://arxiv.org/abs/2402.14903.

Touvron, Hugo, et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971, 2023. https://arxiv.org/abs/2302.13971.

Xue, Linting, et al. "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models." *Transactions of the Association for Computational Linguistics* 10 (2022): 291–306. https://arxiv.org/abs/2105.13626.

## Appendix: Code for Section 5.8

These listings reproduce the examples in this section. Run them in order in one Python session; they need the `tiktoken` package. A.1 defines the encodings and a helper, `show`, used by the rest.

### A.1 Setup and spelling

Load three encodings and compare "strawberry" with and without a leading space.

```python
import tiktoken

encs = {n: tiktoken.get_encoding(n) for n in ["gpt2", "cl100k_base", "o200k_base"]}
enc = encs["gpt2"]

def show(e, s):
    return [e.decode([i]) for i in e.encode(s)]
```

```python
for name, e in encs.items():
    print(name, [(w, show(e, w)) for w in ["strawberry", " strawberry"]])
```

Output:

```
gpt2 [('strawberry', ['st', 'raw', 'berry']), (' strawberry', [' strawberry'])]
cl100k_base [('strawberry', ['str', 'aw', 'berry']), (' strawberry', [' strawberry'])]
o200k_base [('strawberry', ['st', 'raw', 'berry']), (' strawberry', [' strawberry'])]
```

### A.2 Numbers

Tokenize a few numbers with GPT-2, then count digit-split patterns for 1 to 10,000.

```python
for s in [" 1234", " 1235", " 2023", " 2024", " 7777", " 3.14159", " 100000"]:
    print(repr(s), show(enc, s))
```

Output:

```
' 1234' [' 12', '34']
' 1235' [' 12', '35']
' 2023' [' 20', '23']
' 2024' [' 2024']
' 7777' [' 7', '777']
' 3.14159' [' 3', '.', '14', '159']
' 100000' [' 100', '000']
```

```python
from collections import Counter

for name, e in encs.items():
    patterns = Counter()
    for k in range(1, 10001):
        pieces = [p.strip() for p in show(e, " " + str(k))]
        patterns[tuple(len(p) for p in pieces if p)] += 1
    print(name, len(patterns), patterns.most_common(4))
```

Output:

```
gpt2 11 [((2, 2), 4680), ((1, 3), 3417), ((3, 1), 513), ((1, 2), 470)]
cl100k_base 5 [((3, 1), 9000), ((3,), 900), ((2,), 90), ((1,), 9)]
o200k_base 5 [((3, 1), 9000), ((3,), 900), ((2,), 90), ((1,), 9)]
```

### A.3 Whitespace and capitalization

Variants of "hello" and an indented line of code.

```python
for w in ["hello", " hello", "Hello", " Hello", " HELLO"]:
    print(repr(w), enc.encode(w))
```

Output:

```
'hello' [31373]
' hello' [23748]
'Hello' [15496]
' Hello' [18435]
' HELLO' [47899, 46]
```

```python
line = "        return x"
print(show(enc, line))
print(show(encs["cl100k_base"], line))
```

Output:

```
[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' return', ' x']
['       ', ' return', ' x']
```

### A.4 Prompt boundaries, glitch tokens, and non-canonical sequences

A trailing space, " SolidGoldMagikarp", and two token sequences for " hello".

```python
for s in ["The capital of France is", "The capital of France is "]:
    ids = enc.encode(s)
    print(repr(s), ids[-2:], show(enc, s)[-2:])
```

Output:

```
'The capital of France is' [4881, 318] [' France', ' is']
'The capital of France is ' [318, 220] [' is', ' ']
```

```python
print(enc.encode(" SolidGoldMagikarp"))
```

Output:

```
[43453]
```

```python
canonical = enc.encode(" hello")
alternative = enc.encode(" hel") + enc.encode("lo")
print(canonical, alternative, enc.decode(alternative) == enc.decode(canonical))
```

Output:

```
[23748] [932, 5439] True
```
