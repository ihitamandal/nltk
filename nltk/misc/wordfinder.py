# Natural Language Toolkit: Word Finder
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

# Simplified from PHP version by Robert Klein <brathna@gmail.com>
# http://fswordfinder.sourceforge.net/

import random

from nltk.corpus import words


def revword(word):
    if random.randint(1, 2) == 1:
        return word[::-1]
    return word


# try to insert word at position x,y; direction encoded in xf,yf
def step(word, x, xf, y, yf, grid):
    for i in range(len(word)):
        if grid[xf(i)][yf(i)] != "" and grid[xf(i)][yf(i)] != word[i]:
            return False
    for i in range(len(word)):
        grid[xf(i)][yf(i)] = word[i]
    return True


def check(word, dir, x, y, grid, rows, cols):
    length = len(word)
    if dir == 1 and (x - length >= 0 and y - length >= 0):
        return step(word, x, lambda i: x - i, y, lambda i: y - i, grid)
    elif dir == 2 and (x - length >= 0):
        return step(word, x, lambda i: x - i, y, lambda i: y, grid)
    elif dir == 3 and (x - length >= 0 and y + length <= cols):
        return step(word, x, lambda i: x - i, y, lambda i: y + i, grid)
    elif dir == 4 and (y - length >= 0):
        return step(word, x, lambda i: x, y, lambda i: y - i, grid)
    return False


def wordfinder(words, rows=20, cols=20, attempts=50, alph="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """
    Attempt to arrange words into a letter-grid with the specified
    number of rows and columns.  Try each word in several positions
    and directions, until it can be fitted into the grid, or the
    maximum number of allowable attempts is exceeded.  Returns a tuple
    consisting of the grid and the words that were successfully
    placed.

    :param words: the list of words to be put into the grid
    :type words: list
    :param rows: the number of rows in the grid
    :type rows: int
    :param cols: the number of columns in the grid
    :type cols: int
    :param attempts: the number of times to attempt placing a word
    :type attempts: int
    :param alph: the alphabet, to be used for filling blank cells
    :type alph: list
    :rtype: tuple
    """
    words = sorted(words, key=len, reverse=True)
    grid = [[""] * cols for _ in range(rows)]
    used = []

    for word in words:
        word = word.strip().upper()
        save = word
        word = revword(word)
        length = len(word)

        for attempt in range(attempts):
            r = random.randrange(length + 1)
            dir = random.choice([1, 2, 3, 4])
            x, y = random.randrange(rows + 1), random.randrange(cols + 1)

            if dir == 1:
                x += r
                y += r
            elif dir == 2:
                x += r
            elif dir == 3:
                x += r
                y -= r
            elif dir == 4:
                y += r

            if (
                0 <= x < rows
                and 0 <= y < cols
                and check(word, dir, x, y, grid, rows, cols)
            ):
                used.append(save)
                break

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == "":
                grid[i][j] = random.choice(alph)

    return grid, used


def word_finder():
    from nltk.corpus import words

    wordlist = words.words()
    random.shuffle(wordlist)
    wordlist = wordlist[:200]
    wordlist = [w for w in wordlist if 3 <= len(w) <= 12]
    grid, used = wordfinder(wordlist)

    print("Word Finder\n")
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            print(grid[i][j], end=" ")
        print()
    print()

    for i in range(len(used)):
        print("%d:" % (i + 1), used[i])


def step(word, x, x_func, y, y_func, grid):
    length = len(word)
    for i in range(length):
        if grid[x_func(i)][y_func(i)] not in ("", word[i]):
            return False
    for i in range(length):
        grid[x_func(i)][y_func(i)] = word[i]
    return True


if __name__ == "__main__":
    word_finder()
