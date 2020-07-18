from symSpellPlus import *
import numpy as np


symSpell, vocab = initializeSymspell()


def testCategorizeWords():
    sentence = ['All', 'the', 'best', 'people', 'understand', 'FOMO', ',',
                'said', 'Mr', '.', 'Andersen', 'of', 'TwerkCo', '.']

    expected = [1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 3, 0]
    actual = categorizeWords(sentence)
    assert expected == actual


def testHas2VocabWords():
    # vocab = {'a', 'aardvark', 'busy', 'candid', 'lurch', 'toad', 'yes'}

    sentence = ['eoi2w', 'izzit', 'evid', '#EDi', 'Yes', 'fuleish', 'lurch']
    assert has2VocabWords(sentence, vocab)


def testSymSpellLine():
    sentence = ("There ain't noway to figur eout what this hsould be saying,"
                " accordin gto the fail ing New YorkTimes.")
    # sentence = ("An IBM employee, she passed out, quickly recoveredand tried"
    #             " to hold her brains in for over an hour until someone "
    #             "noticed and came to heraid.")
    expected = ("There ain't no way to figure out what this should be saying"
                " according to the failing New York Times")
    actual = symSpellLine(symSpell, vocab, sentence)

    print(actual)
    assert actual == expected


# def testBestSymspelledLine(words, symspell, vocab, line):
#     rawSentence = ("She initially passed out, but quickly recoveredand tried"
#                    " to hold her brains in for over an hour until someone "
#                    "noticed and came to heraid.")
#     words = word_tokenize(rawSentence)

# def testSymSpellDoc():
#     sentence = ("There ain't no wayto figur eout what this hsould be saying,"
#                 " accordin gto the fail ing New YorkTimes.")
#     expected = ("There ain't no way to figure out what this should be saying"
#                 " according to the failing New York Times")
#     actual = symSpellLines
