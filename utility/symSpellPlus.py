from symspellpy.symspellpy import SymSpell
import pkg_resources
from itertools import islice

from nltk.tokenize import word_tokenize, sent_tokenize
import string


def categorizeWords(sent):
    """
    sent		list(type=str), sentence, tokenized into words

    returns a list of integers equalling the length of the sentence, with
    values according to:

        0, no upper case letters
        1, first letter capitalized	(includes 'I')
        2, all caps			(not 'I'!)
        3, mixed (at least one not first letter)

    Used elsewhere to decide how to handle caps after string processing such
    as spell checking.
    """

    returnArr = []
    for word in sent:
        capCt = sum(1 for c in word if c.isupper())

        if capCt == 0:
            returnArr.append(0)
        elif capCt == len(word):
            returnArr.append(2)
        elif capCt == 1 and word[0].isupper():
            returnArr.append(1)
        else:
            returnArr.append(3)

    return returnArr


def has2VocabWords(sentence, vocab):
    """
    sentence		list(type=str)
    vocab		set, vocabulary from symspel dictionaries

    Returns True if any two words in sentence are in vocab. (Presumes that
    sentence has already been tested to be of length > 1.)
    """

    wordCt = 0
    for word in sentence:
        if word.lower() in vocab:
            wordCt += 1
        if wordCt > 1:
            return True

    return False


def initializeSymspell():
    print("inside initializeSymspell()")
    symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    print("symspell created")
    resourceNames = ["symspellpy", "frequency_dictionary_en_82_765.txt",
                     "frequency_bigramdictionary_en_243_342.txt"]
    dictionaryPath = pkg_resources.resource_filename(resourceNames[0],
                                                     resourceNames[1])
    print("dictionaryPath created")
    symspell.load_dictionary(dictionaryPath, 0, 1)
    symspell.create_dictionary_entry(key='ap', count=500000000)
    symspell.create_dictionary_entry(key='ibm', count=500000000)
    symspell.create_dictionary_entry(key="ain't", count=5000000000)
    print(list(islice(symspell.words.items(), 5)))
    print("symspell.load_dictionary() done")
    bigramPath = pkg_resources.resource_filename(resourceNames[0],
                                                 resourceNames[2])
    symspell.load_bigram_dictionary(bigramPath, 0, 1)
    symspell.create_bigram_dictionary_entry(key=('no way'), count=5000000000)
    print("symspell.load_bigram_dictionary() done")
    print(list(islice(symspell.bigrams.items(), 5)))
    print("symspell.load_bigram_dictionary() done")

    # Create vocab
    vocab = set([w for w, f in symspell.words.items()])

    return symspell, vocab


def symSpellLine(symSpell, vocab, sent, maxEditDist=2):
    """
    symSpell		symspell object
    sent		list(type=str) containing str from nltk's
                        sent_tokenize()
    """
    OK = True
    words = word_tokenize(sent)
    for word in words:
        if word not in vocab:
            OK = False
            break
    if OK:
        return sent
    else:
        line = []
        suggestions = symSpell.lookup_compound(sent, transfer_casing=True,
                                               max_edit_distance=maxEditDist)
        for i, suggestion in enumerate(suggestions):
            line.append(suggestion._term)
            print(f"{i:02d}: {type(suggestion)}\t{suggestion}")

        return " ".join(line)


def bestSymspelledLine(words, symSpell, vocab, line):
    wordCategories = categorizeWords(words)
    symspelledLine = symSpellLine(symSpell, vocab, line)
    symWords = word_tokenize(symspelledLine)
    tokenCt = max([len(words), len(symWords)])

    bestLine = ""
    j = 0
    for i in range(tokenCt):
        if words[i].lower() == symWords[j].lower():
            bestLine += words[i].lower()
        elif words[i] in string.punctuation:
            if (i > 0) and (words[i - 1] == symWords[j]):
                pass
        else:
            pass
    if words[-1] in string.punctuation:
        bestLine += words[-1]

    return bestLine


def symSpellDoc(symSpell, vocab, text):
    """
    symSpell		symspell object
    vocab		set, containing vocab from symspell dictionaries
    text		str, containing text to be fixed up

    Breaks text into blocks by splitting on '\n\n', sentence tokenizes each
    block (preserving case), spell corrects using symspell on each sentence,
    and then reconstructs text.

    Prior to sentence tokenization, requires that at least one word in text is
    in vocab, in an effort to remove blocks originating from smudges on image.
    """
    blocks = text.split('\n\n')

    paragraphs = []
    for block in blocks:
        lines = sent_tokenize(block)

        sentences = []
        for line in lines:
            words = word_tokenize(line)
            if (len(words) == 1 and words[0] in vocab) \
               or has2ValidWords(words, vocab):
                sentences.append(bestSymspelledLine(words, symspell, line))
            # else drop the line as garbage

    doc = '\n\n'.join(blocks)

    return doc
