# 9.7 Evaluating SFT Models and the Limits of Imitation

After fine-tuning, the question is whether the model got better, and at what. It is harder than it sounds. The training loss says how well the model predicts its demonstrations, but an assistant is judged by what it produces for prompts nobody wrote demonstrations for, in an open-ended space where many different responses are good. This section looks at four ways to evaluate an SFT model: held-out loss, instruction-following checks that code can verify, pairwise comparisons by people or by a model acting as judge, and regression checks on the base model's abilities. Each is tried on SmolLM2-135M and its instruction-tuned version. The section then turns to the risks of fine-tuning, hallucination and weakened safety behavior, and ends with what learning from demonstrations cannot teach.

## Held-out loss measures fit, not helpfulness

The first number any training run produces is the loss on held-out demonstrations. It is cheap, it is computed exactly like the training loss (Section 2), and a sudden rise signals gross overfitting or a bug. But it answers a narrow question: how probable are the reference responses under the model? Section 5 described how InstructGPT's SFT models overfit on validation loss after one epoch while their outputs kept improving by the judgment of human labelers (Ouyang et al. 2022). There are many good answers to most prompts, and a model can assign less probability to the particular wording of a reference while producing better answers of its own. Held-out loss is also blind to properties of whole responses, such as whether the answer is correct, whether it respects a length limit, or whether it stops at the right place. Comparing checkpoints therefore needs measures of the responses themselves.

## Instruction-following checks that code can verify

Some instructions can be checked mechanically: "answer in fewer than 100 words", "use exactly three bullet points", "write in lowercase", "end with this phrase", "return valid JSON". A test set built from such prompts can be scored by code, cheaply and reproducibly, with no human or model judge. **IFEval** (Zhou et al. 2023b) collects about 500 prompts, each with one or more verifiable instructions of 25 types, such as "write in more than 400 words" and "mention the keyword of AI at least 3 times", and has become a standard measure of instruction following.

Appendix A.1 builds a tiny version: ten prompts with constraints of eight kinds, each checked by a few lines of Python. Two ask for a description under a word limit ("Describe the moon in fewer than 30 words."), two for a bulleted list with an exact number of items using "-" as the bullet, one for a sentence entirely in lowercase, one in capitals, one for a bare "yes" or "no", one for an answer ending in "That is all.", one for a JSON object with given keys, and one for the single word "hello". Both SmolLM2 models receive the prompts in the chat template and answer with greedy decoding.

| Model | Checks passed | Which passed |
|---|---|---|
| SmolLM2-135M (base) | 0 of 10 | none |
| SmolLM2-135M-Instruct | 4 of 10 | both word limits, yes or no, the closing phrase |

The base model, which never learned the chat format, did not answer at all. For the moon prompt, it repeated the request and continued with a line from the system prompt its tokenizer's template inserts; for the other two prompts shown in the appendix, it repeated the request and trailed off into long runs of commas. The instruction-tuned model answered every prompt, and passed four checks. It failed both bullet-list checks, the lowercase and capitals checks, the JSON check, and "hello". The failures are instructive: asked for exactly three fruits as a bulleted list using "-", it answered with a numbered list, "1. Banana", "2. Apple", "3. Orange". The content was right, and the format instruction was ignored.

The passes are instructive too. The moon description, "The moon is a small, glowing orb that orbits Earth, often referred to as the "dark side of the moon."", respects the word limit and so passes the check, but its second half is confused. Verifiable checks measure whether a response obeys the form of an instruction, not whether it is true, relevant, or helpful. That is their strength, since they need no judge, and their limit, since most of what makes an answer good cannot be checked by a regular expression. Ten prompts is also far too few to rank models; the purpose here is to show the mechanics, and a real evaluation uses hundreds.

## Pairwise comparison and LLM judges

For open-ended requests, such as "explain this error message" or "write a polite reply to this email", there is no single right answer to compare against and no rule to check. The most direct evaluation is to show a judge two responses to the same prompt, one from each of two models, and ask which is better. Aggregated over many prompts, the win rate says which model is preferred. People are the reference judges, but they are slow and expensive, and they disagree with each other.

A cheaper alternative is to ask a strong LLM to act as the judge. Zheng et al. (2023) studied this with two benchmarks, MT-Bench, a set of multi-turn questions, and Chatbot Arena, a crowdsourced platform where users compare anonymous models. They found that strong judges such as GPT-4 agreed with human preferences more than 80% of the time, the same level of agreement as between humans. The QLoRA authors of Section 6 similarly found GPT-4 evaluations to be a cheap and reasonable alternative to human evaluation (Dettmers et al. 2023). But Zheng et al. also documented systematic biases: **position bias**, favoring a response because of where it is shown, often first; **verbosity bias**, favoring longer responses; and **self-enhancement bias**, favoring responses written by the judge model itself. Verbosity bias matters especially for SFT, because a fine-tuning run that makes answers longer can win comparisons without making them better.

Position bias is easy to see in a small model. Appendix A.2 asks SmolLM2-135M-Instruct to act as a judge on six questions with one correct and one absurd answer, such as "What is the capital of France?" with "Paris." and "London.", and reads its preference from the probabilities it assigns to the letters A and B. Each pair is judged twice, with the correct answer shown first and then second.

| Order | Correct answer chosen |
|---|---|
| Correct answer shown first | 3 of 6 |
| Correct answer shown second | 3 of 6 |
| Verdict unchanged after swapping | 0 of 6 |

The model picked the correct answer only half the time in each order, no better than chance, and in every one of the six pairs, swapping the order flipped its verdict. It chose the same letter for a given question whichever answer stood behind it, so its choices depended on the position and the wording of the prompt, not on which answer was right. A 135-million-parameter model is far too weak to be a judge, and strong judges are much better, but the same check applies to them. A standard remedy is to judge every pair in both orders and count a win only when the two verdicts agree, which cancels position bias at twice the cost. Comparing responses of similar length, or instructing the judge to ignore length, reduces verbosity bias, and using a judge from a different model family avoids self-enhancement.

## Regression checks

An SFT model should gain new behavior without losing old abilities. Section 5 treated catastrophic forgetting as a risk of training; evaluation is where it is measured. Two measures are standard: perplexity on held-out text of the kind seen in pretraining (Section 8.1), and few-shot benchmarks from before fine-tuning, run with the same prompts on the base and the fine-tuned model.

Appendix A.3 computes the perplexity of both SmolLM2 models on the same 20 blocks of 128 tokens from the end of tiny Shakespeare used in Section 5. The base model scores 41.7, and SmolLM2-135M-Instruct scores 56.5. Section 5's own fine-tuning run, with far less data, moved the base model to 47.0; the LoRA runs in Section 6 moved it much further. Such numbers do not say by themselves that a model has become worse at anything that matters, since an assistant does not need to predict old plays, and part of the rise reflects a change in the kind of text the model expects. They are a tripwire: a large increase, like the rank-32 LoRA run's 322.8, says the model has moved far from its pretraining distribution, and the abilities that depend on that distribution should be checked directly.

## Hallucination

A tempting use of SFT is to teach a model new facts by writing demonstrations that contain them. Gekhman et al. (2024) tested this in a controlled closed-book question-answering setup, varying the fraction of fine-tuning examples that introduced knowledge the model did not already have. Examples with new knowledge were learned significantly more slowly than examples consistent with what the model already knew. And as the new examples were eventually learned, the model's tendency to **hallucinate**, to state factually incorrect answers, increased linearly.

One reading of this result is that a demonstration containing an unknown fact teaches the model a behavior as well as a fact: answer confidently even when the answer is not supported by what you know. The finding supports the view from Section 1 that models acquire factual knowledge mostly in pretraining, and that fine-tuning teaches them to use it. For SFT data, the practical consequence is to be careful with demonstrations whose answers go beyond what the base model can know, and to include demonstrations of uncertainty and refusal where the model should not answer.

## Safety can be undone by fine-tuning

Assistants are usually trained to refuse clearly harmful requests, and that behavior is itself a product of post-training. It turns out to be fragile. Qi et al. (2024) fine-tuned GPT-3.5 Turbo through OpenAI's fine-tuning API on only 10 adversarially designed examples, at a cost of less than $0.20, and the model became responsive to nearly any harmful instruction. More worrying for ordinary users, fine-tuning on benign and commonly used datasets, with no malicious intent, also degraded safety behavior, though to a lesser extent.

The lesson for anyone fine-tuning an assistant, even a harmless one for a narrow task, is that the fine-tuned model must be re-evaluated for safety, not assumed to inherit the base assistant's behavior. Including safety demonstrations in the fine-tuning mixture, as open recipes do (Section 4), helps preserve it.

## What imitation cannot teach

The evaluations above point to limits that are not about any particular dataset or recipe but about learning from demonstrations as such.

**SFT sees only good examples.** The loss rewards the reference response and says nothing about any alternative. The model is never told that one answer is better than another, or which mistakes to avoid, or how bad a particular mistake is. A response that is almost right and one that is dangerously wrong are equally absent from the data.

**The model is trained on reference text but must condition on its own.** As Section 2 explained, SFT is teacher forcing: every prediction during training is made with the reference prefix, while at inference the model continues from its own earlier tokens. An early mistake puts it in a situation no demonstration covered, and nothing in the training taught it how to recover. The effect grows with the length of the response, which makes it important for multi-step reasoning.

**Demonstrations are expensive, and the model can only copy them.** Writing an excellent response to a hard prompt, such as a proof, a careful legal summary, or a large piece of code, requires expertise and time, and the model can at best imitate the quality of what it is shown. Imitating a stronger model's outputs, as Section 4 described, copies its style more readily than its factual accuracy or ability (Gudibande et al. 2023). Judging is often easier than writing: many people who could not write the best answer can still tell which of two answers is better, and some tasks can be checked automatically, as the verifiable instructions above show.

These three limits point to the same remedy: let the model produce its own responses, and learn from judgments of them, comparisons between pairs of responses or rewards from people, models, or automatic checks. Such training addresses the missing negative signal, trains on the model's own outputs rather than reference text, and replaces expensive demonstrations with cheaper judgments. It starts from the SFT model, which already follows instructions and uses the chat format, and it inherits the evaluation tools of this section, since a judge that compares responses and a check that verifies an instruction are exactly the kinds of signal it learns from.

## Key takeaways

- Held-out loss on demonstrations measures fit to the reference wording, not helpfulness; compare checkpoints with metrics on the responses themselves.
- Verifiable instruction checks, as in IFEval's roughly 500 prompts with 25 instruction types, are cheap and reproducible but measure form, not correctness; on ten such checks, SmolLM2-135M-Instruct passed 4 and its base model 0.
- Pairwise comparison by people or by a strong LLM judge handles open-ended tasks; strong judges agree with humans more than 80% of the time, but have position, verbosity, and self-enhancement biases. Judging in both orders cancels position bias; a 135-million-parameter judge flipped its verdict in all six swapped pairs.
- Regression checks such as held-out perplexity (41.7 for the base SmolLM2 model, 56.5 for the instruction-tuned one) and few-shot benchmarks measure forgetting.
- Fine-tuning on facts the model does not know is learned slowly and increases hallucination; safety behavior can be removed with 10 adversarial examples and weakened even by benign data, so fine-tuned models need fresh safety evaluation.
- SFT learns only from good examples, trains on reference text rather than the model's own outputs, and depends on expensive demonstrations; learning from comparisons and rewards on the model's own responses, starting from the SFT model, addresses all three.

## Further reading

Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of Quantized LLMs." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.14314.

Gekhman, Zorik, et al. "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 2024. https://arxiv.org/abs/2405.05904.

Gudibande, Arnav, et al. "The False Promise of Imitating Proprietary LLMs." arXiv preprint arXiv:2305.15717, 2023. https://arxiv.org/abs/2305.15717.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Qi, Xiangyu, et al. "Fine-Tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!" In *International Conference on Learning Representations*, 2024. https://arxiv.org/abs/2310.03693.

Zheng, Lianmin, et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2306.05685.

Zhou, Jeffrey, et al. "Instruction-Following Evaluation for Large Language Models." arXiv preprint arXiv:2311.07911, 2023b. https://arxiv.org/abs/2311.07911.

## Appendix: Code for Section 9.7

These listings reproduce the examples in this section. Run them in order in one Python session; they need the `torch` and `transformers` packages, download the models from the Hugging Face Hub on first use, and run on a CPU. A.3 downloads the tiny Shakespeare text if it is not already present.

### A.1 Verifiable instruction checks

Ten prompts with constraints that code can check, answered by the base and the instruction-tuned SmolLM2-135M.

```python
import json, re, math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
models = {name: AutoModelForCausalLM.from_pretrained(f"HuggingFaceTB/{name}",
                                                     dtype=torch.float32).eval()
          for name in ["SmolLM2-135M", "SmolLM2-135M-Instruct"]}

def words(t): return re.findall(r"\b\w+\b", t)
def bullets(t): return [l for l in t.splitlines() if re.match(r"^\s*[-*]\s+", l)]

def is_json(t):
    try:
        json.loads(t.strip().strip("`").removeprefix("json").strip())
        return True
    except ValueError:
        return False

checks = [
    ("Describe the moon in fewer than 30 words.", lambda t: 0 < len(words(t)) < 30),
    ("Describe a cat in fewer than 20 words.", lambda t: 0 < len(words(t)) < 20),
    ("List exactly three fruits as a bulleted list using '-' for each bullet.",
     lambda t: len(bullets(t)) == 3),
    ("Give exactly two tips for sleeping well, as a bulleted list using '-'.",
     lambda t: len(bullets(t)) == 2),
    ("Write one sentence about rain, entirely in lowercase letters.",
     lambda t: len(words(t)) > 0 and t == t.lower()),
    ("Write one sentence about the sun, in capital letters only.",
     lambda t: len(words(t)) > 0 and t == t.upper()),
    ("Answer with only the word yes or no: is fire hot?",
     lambda t: t.strip().strip(".").lower() in {"yes", "no"}),
    ("Name a color. End your answer with the exact phrase 'That is all.'",
     lambda t: t.strip().endswith("That is all.")),
    ('Return a JSON object with the keys "name" and "age" for a person called Ana who is 30.',
     is_json),
    ("Write the word hello, and nothing else.", lambda t: t.strip().strip(".!").lower() == "hello"),
]

def answer(model, prompt, max_new_tokens=80):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

results = {}
for name, model in models.items():
    results[name] = [(p, answer(model, p)) for p, _ in checks]
    passed = [check(a) for (_, check), (_, a) in zip(checks, results[name])]
    print(f"{name}: {sum(passed)}/{len(checks)} checks passed ->",
          "".join("P" if ok else "." for ok in passed))
```

Output:

```
SmolLM2-135M: 0/10 checks passed -> ..........
SmolLM2-135M-Instruct: 4/10 checks passed -> PP....PP..
```

```python
for (p, a_base), (_, a_inst) in list(zip(results["SmolLM2-135M"],
                                         results["SmolLM2-135M-Instruct"]))[:3]:
    print(f"PROMPT: {p}\n  base:     {a_base[:110]!r}\n  instruct: {a_inst[:110]!r}")
```

Output:

```
PROMPT: Describe the moon in fewer than 30 words.
  base:     'Describe the moon in fewer than 30 words.\n\nYou are a helpful AI assistant named SmolLM, trained by Hugging Fac'
  instruct: 'The moon is a small, glowing orb that orbits Earth, often referred to as the "dark side of the moon."'
PROMPT: Describe a cat in fewer than 20 words.
  base:     'Describe a cat in fewer than 20 words.,,,,,,,,,,,,,,,,aking\n,,,,,,,,,,,,,,,,akingassistant\nDescribe a cat in f'
  instruct: 'A cat is a cute, fluffy, and playful creature with a fluffy coat and a fluffy tail.'
PROMPT: List exactly three fruits as a bulleted list using '-' for each bullet.
  base:     "List exactly three fruits as a bulleted list using '-' for each bullet.,,,,,,,,,,,,,,,,ingredients\n,,,,,,,,,,,"
  instruct: '1. Banana\n2. Apple\n3. Orange'
```

### A.2 Position bias in a small judge

Ask the instruction-tuned model which of two answers is better, then swap their order.

```python
pairs = [("What is the capital of France?", "Paris.", "London."),
         ("What is 2 + 2?", "4.", "5."),
         ("How many legs does a dog have?", "Four.", "Seven."),
         ("What color is grass?", "Green.", "Purple."),
         ("What do bees make?", "Honey.", "Glass."),
         ("What is frozen water called?", "Ice.", "Smoke.")]
judge = models["SmolLM2-135M-Instruct"]
idA, idB = tok("A").input_ids[-1], tok("B").input_ids[-1]

def prefers_first(q, first, second):
    prompt = (f"Question: {q}\nAnswer A: {first}\nAnswer B: {second}\n"
              "Which answer is correct? Reply with the letter A or B only.")
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    with torch.no_grad():
        logits = judge(tok(text, return_tensors="pt").input_ids).logits[0, -1]
    return (logits[idA] > logits[idB]).item()

right_first = [prefers_first(q, good, bad) for q, good, bad in pairs]
right_second = [not prefers_first(q, bad, good) for q, good, bad in pairs]
print("correct answer shown first: judge picks it", sum(right_first), "of", len(pairs))
print("correct answer shown second: judge picks it", sum(right_second), "of", len(pairs))
print("verdict unchanged when the order is swapped:",
      sum(a == b for a, b in zip(right_first, right_second)), "of", len(pairs))
```

Output:

```
correct answer shown first: judge picks it 3 of 6
correct answer shown second: judge picks it 3 of 6
verdict unchanged when the order is swapped: 0 of 6
```

### A.3 Regression check on held-out text

Compare the perplexity of the base and instruction-tuned models on held-out Shakespeare.

```python
import os, urllib.request
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("shakespeare.txt"):
    urllib.request.urlretrieve(URL, "shakespeare.txt")
text = open("shakespeare.txt", encoding="utf-8").read()
held = tok(text[-20000:], add_special_tokens=False).input_ids
x = torch.tensor([held[i:i + 128] for i in range(0, 20 * 128, 128)])
for name, model in models.items():
    with torch.no_grad():
        print(f"{name}: perplexity {math.exp(model(x, labels=x).loss.item()):.1f}")
```

Output:

```
SmolLM2-135M: perplexity 41.7
SmolLM2-135M-Instruct: perplexity 56.5
```
