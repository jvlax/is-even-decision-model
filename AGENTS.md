# AGENTS.md

Rules in this repository are kept as verses.

A verse reference is a stable, hierarchical, versioned address into a text
that every machine and every person already has a copy of: book, chapter,
verse. The text carries the reasoning, so a rule needs one line of gloss and a
pointer. Nothing here is from the highest authority. It is from someone worth
listening to.

Text: King James Version, as published in
[aruljohn/Bible-kjv](https://github.com/aruljohn/Bible-kjv) at commit
`4184eec` (`Luke.json`, sha256
`a4439cd14cd1dd637ae614bd556499dca8df92db4f5d9f4a69c3c571969c5576`). Quote
from that copy, and cite the reference, so that a rule can be looked up
without this file.

## Luke 1:3 — always read the comments

> It seemed good to me also, having had perfect understanding of all things
> from the very first, to write unto thee in order, most excellent Theophilus,

Luke opens by saying how he worked: he went back to the very first, read
everything, and only then wrote it down in order. Do the same. Before you
touch a package, a repository, or a thread, read the issues, the pull
requests, the review comments, and the replies under the README. That is
where the real details hide.

What it found here:

- `is-odd` is not `n % 2`. Its README limits the promise to safe integers,
  and the code checks number, integer, and safe-integer before the modulus.
  This package keeps all three guards.
- `is-even-ai` issue #8 asked whether the package exists to make big companies
  pay more. Nobody answered. The cost table in the README is the answer.
- `is-even-ai` pull request #19 was told to use structured outputs to force a
  true/false return. A decision model's `noul` answer cannot be anything else.
- `is-even-ai`'s exact prompt and model are in `dist/IsEvenAiOpenAi.js`, not
  in the README. The token count and the price per call come from there.

Applies to: every change in this repository, and every dependency it takes on.
