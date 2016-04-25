Phonaestheme networks
Mike Pham
************************

-Run main_code.py
  -I have needed the code to output various files in the past, so there is a lot of commented out code that I need to clean up
  -Code also includes handbuilt network metrics before I started using NetworkX
  
-Folders with primary outputted files included


************************
Background:

This code is part of my dissertation research that looks at phonaesthemes (e.g. gl-, sn-, sl-) — also called sound symbolism — and their relationship to prototypical morphemes (specifically prefixes in this case). There is a general sense that like conventional morphemes (e.g. re-, un-), phonaesthemes have some connection between phonological form and semantic meaning; however, they do not seem to be as consistent or morphologically productive as these conventional morphemes.

In my research, I propose that there is no fundamental difference between morphemes and phonaesthemes, contra standard views in the literature, claiming that the same decompositional algorithms that are needed to learn conventional morphemes are capable of finding phonaesthemes, though the latter have a weaker association between form and meaning.

In modeling this, I constructed semantic networks utilizing a thesaurus (Project Gutenberg Etext of Moby Thesaurus II by Grady Ward): words that share more synonyms in the thesaurus are considered more semantically related to each other. Each node in the graph is a word beginning with a two-letter string — some are attested phonaesthemes, some are conventional prefixes, many are junk — and words are connected by an edge if they are related to each other. 

The graphs show a network of relatedness among words that share a common initial string. The prediction is for conventional morphemes (e.g. un-, re-) to have larger, more related networks, indicating greater semantic relatedness among words beginning with those prefixes, junk strings to have relatively unrelated networks, and phonaesthemes (e.g. gl-) to lie somewhere in the middle. A morphological learner could then associate with varying strength the shared phonological form with a dominant community within the semantic network of words that share that form.
