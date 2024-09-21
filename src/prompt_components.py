CLASSIFY_PROMPT = """
You are tasked with analyzing text revisions from various sources (e.g., Wikipedia, Wikinews, and arXiv) and classifying them as either "good" or "bad" based on the following guidelines:

1. **Good Example**:
   - The text is coherent, grammatically correct, and easy to understand.
   - It flows logically, even if there are minor typos or small issues with punctuation or typos.

2. **Bad Example**:
   - The text is nonsensical or ungrammatical to the point where it’s hard to understand for human readers.
   - There are missing words, wrong spacing, or fragmented sentences that confuse the meaning.
   - The revised version of the text degrades the quality of the original text in a significant way that even proficient readers would struggle to understand the meaning.
   - The text is a clear artifact from Wikipedia or Wikinews. Such an artifact could be a list of references or categories. Another type of artifacts a captions of images.

For each pair of revisions (before and after), classify the pair as follows:

- **good**: The text is coherent and makes sense, even if there are imperfections.
- **bad**: The text is too disjointed, confusing, or nonsensical to be useful in further experiments.

Here is an example revision pair:

- **Before**: "The space betweend the planets is vast. Th distance can be million miles."
- **After**: "The space between the planets is vast. The distance can be millions of miles."

In this case, this would be classified as "good" because the revised version corrects errors and makes sense, even though the original had minor issues.

Another example:

- **Before**: "In space, no sounds can heard. Because thers no aire for the sound waves."
- **After**: "In space no heard, because no sound. Sound waves."

This would be classified as "bad" because the text after the revision is incoherent and hard to understand.

Another example:

- **Before**: "In at , , defeated the Netherlands to win the , and was defeated by the Netherlands in the ."
- **After**: "on Sundayon SaturdayIn at , , defeated the Netherlands to win the , and was defeated by the Netherlands in the ."

This would be classified as "bad" because the text after the revision is nonsensical.

Another example:

- **Before**: bad,"Yesterday, football club (BVB) sacked manager ."
- **After**: "File photo of Peter Stöger in 2011Yesterday, football club (BVB) sacked manager ."

This would be classified as "bad" because the text after the revision is an artifact from Wikipedia or Wikinews (Photo details).

Another example:

- **Before**: "Due to lack of playing time , he moved to Portuguese club ."
- **After**: "With limited playing time with Barcelona , he moved to Portuguese club ."

This example would be classified as "good" because both versions of the text are coherent and make sense. Also, there are no weird artifacts from Wikipedia or Wikinews.

Another example:

- **Before**: "(GEO New GO GO GO ------C.J !"
- **After**: "(GEO News) GO GO GO ------C.J !"

This example would be classified as "bad" because both versions of the text are just nonsensical.


Classify the following text pairs and think step by step about the quality of the text and remember the guidelines above:

**Before**: {}
**After**: {}
"""


SYSTEM_PROMPT = """You are an expert language model tasked with evaluating text revisions for coherence, grammar, and logical structure. Your goal is to help filter out nonsensical or confusing text, while retaining useful examples that can be used for further analysis."""
