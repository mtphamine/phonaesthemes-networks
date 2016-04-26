#Phonaestheme networks
By [Mike Pham](http://www.mikettpham.com/)

## Contents

- ``main.py``: Python script for running the different truncation models over the data files

- data:

    * ``mobyThes.txt``: Moby Thesaurus
    * ``thesGraphSymLinks.txt``: trimmed thesaurus only including 1-word entries (>2 letters long), and where all synonyms also appear as head words
    * ``coca_freqs_alpha.csv``: due to restrictions on using the COCA corpus data, I am unable to include this data file -- if you have access to this corpus, you can generate this file as a list of all words in the corpus with their frequency

- ``gexf_graphs/``: output graphs (unweighted edges) for all 2-letter prefixes in .gexf format for visualization with Gephi

- ``gexf_graphs_jaccard_coca/``: output graphs (weighted edges based on Jaccard Index; word frequency included based on COCA corpus) for all 2-letter prefixes in .gexf format for visualization with Gephi

- ``d3-graphs/``: cursory files for a web-friendly d3 visualization of the data (graphs stored as JSON files)

- ``gephi_graphs/``: sample graphs created in Gephi 

- ``readme.md``: this readme file

## Usage

Download this repository to your local drive by one of these two methods:

* Download and unzip https://github.com/mtphamine/phonaesthemes-networks/archive/master.zip

* Clone this repository:

    ```
    $ git clone https://github.com/mtphamine/phonaesthemes-networks.git
    $ cd phonaesthemes-networks
    ```

After this repository is downloaded, run the networks models for the accompanying datasets:

    $ python main.py

The script includes several chunks of code that produce various outputs. I haven't had time to clean up the interface for selectable parameters with regard to input-output choices; most of the code is simply commented out for now, but the discerning user can uncomment various sections to produce different outputs.

##Background:

This code is part of my dissertation research that looks at phonaesthemes (e.g. gl-, sn-, sl-) — also called sound symbolism — and their relationship to prototypical morphemes (specifically prefixes in this case). There is a general sense that like conventional morphemes (e.g. re-, un-), phonaesthemes have some connection between phonological form and semantic meaning; however, they do not seem to be as consistent or morphologically productive as these conventional morphemes.

In my research, I propose that there is no fundamental difference between morphemes and phonaesthemes, contra standard views in the literature, claiming that the same decompositional algorithms that are needed to learn conventional morphemes are capable of finding phonaesthemes, though the latter have a weaker association between form and meaning.

In modeling this, I constructed semantic networks utilizing a thesaurus (Project Gutenberg Etext of Moby Thesaurus II by Grady Ward): words that share more synonyms in the thesaurus are considered more semantically related to each other. Each node in the graph is a word beginning with a two-letter string — some are attested phonaesthemes, some are conventional prefixes, many are junk — and words are connected by an edge if they are related to each other. 

The graphs show a network of relatedness among words that share a common initial string. The prediction is for conventional morphemes (e.g. un-, re-) to have larger, more related networks, indicating greater semantic relatedness among words beginning with those prefixes, junk strings to have relatively unrelated networks, and phonaesthemes (e.g. gl-) to lie somewhere in the middle. A morphological learner could then associate with varying strength the shared phonological form with a dominant community within the semantic network of words that share that form.
