# Belief-Networks-Hidden-Markov-Models
Fall 2025 CS 362/562

## Part 2 Reflection

Give an example of a word which was correctly spelled by the user, but which was incorrectly
“corrected” by the algorithm. Why did this happen?

- fox → fos

Character-level HMM optimizes letter transitions/emissions, not word validity. Emission/transition stats likely over-weight s after o (e.g., from many “-os-” patterns), and there’s no word-level lexicon/language model to keep the valid word “fox.”

Give an example of a word which was incorrectly spelled by the user, but which was still
incorrectly “corrected” by the algorithm. Why did this happen?

- zibra → dibre

The model hasn’t seen enough patterns for z→z and b→b in this context; smoothing plus noisy emissions can push toward higher-probability letters like d/e. Without a dictionary constraint, it outputs a plausible character sequence rather than a real word.

Give an example of a word which was incorrectly spelled by the user, and was correctly corrected
by the algorithm. Why was this one correctly corrected, while the previous two were not?

- acommadate → accommodate

Training set includes multiple “accommodate” misspellings; alignment learns strong emissions (o from omitted o, mm from single m, etc.) and transitions (co→om→mm→mo). High-quality evidence + frequent transitions overcome noise, unlike “fox”/“zibra.”

How might the overall algorithm’s performance differ in the “real world” if that training dataset is
taken from real typos collected from the internet, versus synthetic typos (programmatically
generated)?

-----

Real-world typos: Include true human patterns—phonetic confusions, adjacent-key errors, doubled letters, common omissions/insertions, domain words. Training on these improves emissions and transitions the model actually needs; better generalization and fewer bizarre outputs.

Synthetic typos: Often uniform/random edits; misrepresent real error distribution. The model learns unhelpful emissions and transitions, causing over/under-corrections and poor generalization.