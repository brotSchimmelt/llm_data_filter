CLASSIFY_PROMPT = """
You are tasked with analyzing text revisions from various sources (e.g., Wikipedia, Wikinews, and arXiv) and classifying them as either "Good" or "Bad" based on the following guidelines:

1. **Good Example**:
   - The text is coherent, grammatically correct, and easy to understand.
   - It flows logically, even if there are minor typos or small issues with punctuation or typos.

2. **Bad Example**:
   - The text is nonsensical or ungrammatical to the point where it’s hard to understand for human readers.
   - There are missing words, wrong spacing, or fragmented sentences that confuse the meaning.
   - The revised version of the text degrades the quality of the original text in a significant way that even proficient readers would struggle to understand the meaning.

For each pair of revisions (before and after), classify the pair as follows:

- **Good**: The text is coherent and makes sense, even if there are imperfections.
- **Bad**: The text is too disjointed, confusing, or nonsensical to be useful in further experiments.

Here is an example revision pair:

- **Before**: "The space betweend the planets is vast. Th distance can be million miles."
- **After**: "The space between the planets is vast. The distance can be millions of miles."

In this case, this would be classified as "Good" because the revised version corrects errors and makes sense, even though the original had minor issues.

Another example:

- **Before**: "In space, no sounds can heard. Because thers no aire for the sound waves."
- **After**: "In space no heard, because no sound. Sound waves."

This would be classified as "Bad" because the text after the revision is incoherent and hard to understand.

Classify the following text pairs:

**Before**: {before_revision}
**After**: {after_revision}
"""


SYSTEM_PROMPT = """You are an expert language model tasked with evaluating text revisions for coherence, grammar, and logical structure. Your goal is to help filter out nonsensical or confusing text, while retaining useful examples that can be used for further analysis."""
