# promptfeed

A small command-line tool for handing a large language model a story it should continue — chapter by chapter, in a voice you specify, with as much or as little outline detail as you want to give it.
It is designed for authors who want to experiment with continuing or rewriting a story in a different voice, or trying different storylines to see which ones work best. 

It's built around two ideas:

1. **The story, the voice, and the plan live in one plain-text file**, separate from the model you run it through. Write it once, and reuse it.
2. **It's designed for local models**. In fact, it ONLY works with local models. You can use [LM Studio](https://lmstudio.ai), or olama, or anything else that will load an LLM and present it to the user. That means you can point the same prompt file at several models to see how each one handles the material, or load a much bigger model than you'd normally use interactively and let it grind through a full run overnight — the program is built to tolerate multi-hour generations without falling over.

## What it does

The program relies on a **prompt file**: a text file that is a summary of the story so far, a description of the characters, a "voice card" describing the prose style to imitate, and then a series of chapter-by-chapter prompts — anywhere from a couple of sentences to a detailed, beat-by-beat outline. promptfeed sends these to a locally running model one chapter at a time, keeping the story consistent as it grows (each new chapter is written with the earlier chapters as context), and writes the results out to a output file.

Because everything about *what* to write lives in the prompt file rather than in code, the same file works unchanged across different local models. Run it against three different models loaded in LM Studio and you get three distinctly named output files you can read side by side — same story, same instructions, different writer. Change the voice card, and run the same story again, and it will write it in a different register with a very different writing style.

Note: you don’t *need* a story to begin with. prompfeed was designed originally to extend stories I was interested in, but you can skip an original story and create a new story with it, using the prompts to build it up. 


## Context and consistency

All LLMs have a context window - a set size of tokens that they can remember at once. It varies from 4,096 tokens to 1M on the largest models. Ultimately, anything the LLM ‘knows’ needs to fit into this context window. When you load a model with LM Studio, you can set the context length up to the max that your memory can handle or that the model can handle. It is important to set it as large as your hardware allows. This larger memory allows the LLM to ‘remember’ more things about the original story, the characters, the previous scenes. promptfeed.py does a number of things to maximize the context that you have, to allow for maximum consistency from scene to scene, and has been tested to show remarkable consistency 




## Installing

```bash
git clone https://github.com/Marhalt/promptfeed.git
cd promptfeed
pip install -r requirements.txt
```

You'll also need:

- **[LM Studio](https://lmstudio.ai)** installed and running, with a model loaded and its local API server turned on (defaults to `http://127.0.0.1:1234`). promptfeed talks to whatever model LM Studio currently has loaded.
- Python 3.10+ is recommended. Some dependencies (`torch`, `faiss-cpu`) are platform-sensitive — if `pip install -r requirements.txt` fails on your machine, you may need to install those two individually for your OS/GPU before retrying.

## Quick start

```bash
python src/promptfeed.py
```

Run with no filename and it looks for any file starting with `prompt` (e.g. `prompt_my_story.txt`) in the current directory and asks you to pick one if there's more than one. Or point it at a specific file:

```bash
python src/promptfeed.py prompt_my_story.txt
```

Output is written to `results_<promptfile>.txt`, or `<model-name>_<promptfile>.txt` when promptfeed can detect which model LM Studio has loaded — which is what makes running the same file across several models easy to compare afterward.

## The prompt file format

A prompt file is plain text, made of sections marked with `&&sectionname&&`. A section ends after two blank lines. Lines starting with `//` are comments and are ignored.

| Section | Purpose |
|---|---|
| `&&summary&&` | A prose summary of the story *up to the point where your prompts pick up*. Optional if you provide `&&file&&` instead, but recommended — it's cheaper to send on every request than the full source text. |
| `&&characters&&` | A paragraph per major character, describing them as they are at the start of the story. |
| `&&system&&` | High-level instructions for the model — tone, house rules, anything project-specific. Required (can be short). |
| `&&voice&&` | The prose style to imitate. Can be written inline, or point to an external file (e.g. a reusable "voice card" for a particular author). Optional. |
| `&&file&&` | Path to the original/source text, if you have it. If it's short enough it's loaded in full; if it's long, promptfeed automatically chunks it and retrieves the passages most relevant to each chapter instead of blowing the context budget. Optional. |
| `&&prompt&&` | One per chapter you want written. Can be a couple of sentences or a fully detailed beat sheet — see below. |

A `#` heading (`# prompt`, `# system`, etc.) works as an alternative to the `&&...&&` syntax if you prefer Markdown-style files.

### Outline detail is up to you

A `&&prompt&&` block can be as loose or as tight as you want:

- **High-level**: two or three sentences describing what happens. Good for a chapter whose exact texture doesn't matter much, or when you want the model to have real room to invent.
- **Beat-by-beat**: numbered beats with a word-count budget per beat. Good for chapters carrying plot-critical facts that later chapters depend on staying consistent.
- **Meta-instructions**: text in `[square brackets]` inside a prompt is read as an instruction to the model about *how* to write the passage, not as part of the story itself — e.g. `[This is the emotional climax — slow down here]` or `[Keep this brief, it's administrative]`. (If you use this, say so explicitly in `&&system&&`, since the model needs to be told the convention exists.)
- **Non-chronological**: an outline doesn't have to describe plot beats at all — you can outline a chapter as an emotional or psychological arc instead and leave the model to invent the scene mechanics that deliver it. Works best with a capable model.

The repo includes an example prompt file continuing Charles Dickens's unfinished *The Mystery of Edwin Drood*, with a run of chapters written at each of these detail levels, as a reference for how far you can push (or not push) an outline.

## Command-line options

| Flag | Effect |
|---|---|
| `-temp <float>` | Sampling temperature (default set in code). |
| `-maxcontext <int>` | Override the context window size promptfeed assumes for the loaded model. |
| `-timeout <minutes>` | Per-generation read timeout before a request is considered failed. Defaults to 4 hours — large contexts on local hardware can genuinely take that long in the later chapters of a run. |
| `-citations [X]` | Force retrieval mode for the source story (`&&file&&`) instead of the automatic full-text-if-it-fits behavior; optionally set how many chunks to retrieve per chapter. |
| `-nocontext` | Don't carry prior chapters forward as context for the next one. |
| `-refusal` | Turn off the refusal-detection/retry system (on by default). |
| `-nocache` | Disable prompt caching on the LLM server. On by default for speed; turn off only if the server starts returning stale/repetitive output. |
| `-rewrite X` | Regenerate only chapter `X` (1-indexed), using the previously saved chapters as context, and patch it back into the existing output file — instead of re-running the whole thing. |
| `-resultsfile PATH` | Use an explicit output file path instead of the auto-derived name. |
| `-verbose` | Extra logging. |

## Some things worth knowing about how it works

A few design choices that aren't obvious from just reading a prompt file:

- **Built for slow, unattended local runs.** The default per-generation timeout is 4 hours, prompt caching is on by default, and the message history is built with the stable parts first so the server's KV cache can be reused between chapters — all aimed at letting you kick off a run with a big, slow local model before bed and come back to a finished set of chapters in the morning.
- **Refusal handling.** Small local models occasionally refuse ordinary dark-fiction content (a murder, a threat, a confrontation) that's completely unremarkable in a story. promptfeed detects this and retries automatically — first with the same settings, then with progressively adjusted temperature, and then with various techniques to overcome refusals before giving up and moving on. If you want very dark stories, though, use an alliterated model. 
- **Reasoning-model support.** If the loaded model emits `<think>...</think>` chain-of-thought, promptfeed strips it out before treating the output as the actual chapter text, and anchors on a required `Ok!` marker to know where the real response starts.
- **Long source texts don't need to fit in context.** If the file in `&&file&&` is too big to send in full, promptfeed automatically chunks it and builds a FAISS index, then retrieves only the passages relevant to each chapter as it's written.
- **A basic "AI slop" check** flags suspiciously long, under-punctuated paragraphs in the output for a human to double check, rather than silently accepting them.
- **Regenerate one chapter, not the whole run**, with `-rewrite` — useful when nine out of ten chapters came out fine and only one needs another pass.

## Status

Early / actively changing. This README is a starting point — contributions and issues welcome.
