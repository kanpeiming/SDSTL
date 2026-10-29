---
name: cs-paper-search
description: Search and summarize recent computer science papers, prioritizing the last two years, top venues, arXiv, official proceedings, open review pages, and relevant GitHub paper lists while keeping keywords flexible.
metadata:
  short-description: CS paper search
---

# CS Paper Search

Use this skill when the user asks to search, compare, summarize, or build a literature list for computer science research papers. It is especially useful for event cameras, SNNs, DVS, RGB-to-event transfer, domain adaptation, multimodal learning, dense prediction, and related work for the SDSTL project.

## Search Priorities

Prefer recent work from the last two years. Do not lock the search to that range: include older papers when they are foundational, highly cited, or directly explain the user's current bottleneck.

Prioritize sources in this order when available:

1. Official paper pages, proceedings, or publisher pages.
2. arXiv or OpenReview pages.
3. Project pages, GitHub repositories, and Papers with Code.
4. Curated GitHub paper lists and survey pages.
5. Search snippets only when the source page cannot be opened.

Read [references/search-sources.md](references/search-sources.md) when the task needs source selection, venue names, or seed query ideas.

## Query Strategy

Start from the user's keywords, then expand them when results are sparse or the project is blocked. For SDSTL style work, useful keyword families include:

- event camera, DVS, neuromorphic vision, event stream, event representation
- spiking neural network, SNN, LIF, temporal efficient training, surrogate gradient
- RGB-to-event, image-to-event, video-to-event, event simulation, v2e, I2E
- transfer learning, domain adaptation, cross-modal learning, multimodal fusion
- edge representation, structure decoupling, high frequency information, MaxPool, frequency bias
- attention, cross-modal attention, temporal attention, event mid-frame attention
- classification, dense prediction, detection, segmentation, optical flow, depth estimation

Use several short searches instead of one overloaded query. When the user's experiment is stuck, deliberately try adjacent terms and methods rather than repeating the same search terms.

## Output Format

For every paper that is included, provide:

- Title
- Year and venue, or arXiv date if not peer reviewed
- Source URL
- Code or project URL when available
- Core idea in 2 to 4 sentences
- Why it matters for the user's work
- Practical takeaway, limitation, or experiment idea

For a short answer, use a compact bullet list. For a literature review, use a markdown table followed by a synthesis section.

## Ranking Criteria

Prefer papers that meet several of these:

- Published in top conferences or journals.
- Released in the last two years.
- Has official code, strong benchmark evidence, or an active project page.
- Directly connects to the user's method, dataset, model, or bottleneck.
- Offers a concrete idea that can become an experiment or ablation.

Do not overvalue a paper just because it is recent. If a newer preprint is weakly connected, rank a stronger older paper above it and explain why.

## Source Handling

Always cite sources with visible URLs in the final answer. When possible, use the direct paper page rather than a search result page.

If a source is blocked or only a snippet is available, say so briefly and avoid overstating details. For paywalled papers, prefer arXiv, official accepted manuscript, project page, or DOI landing page.

Avoid long quotations from abstracts. Summarize in your own words.

## SDSTL-Specific Lens

When the task relates to `C:\coding\SDSTL`, connect each paper to one or more of these project questions:

- Does it support or challenge RGB-to-Edge-to-DVS structural decoupling?
- Does it improve synthetic DVS generation or sample alignment?
- Does it suggest a better attention design for event sequences?
- Does it address SNN low frequency bias or high frequency preservation?
- Does it provide an ablation, metric, dataset, or baseline the project can reuse?

End with a short "Recommended Next Reads" or "Recommended Next Experiments" section when the user is looking for direction.
