#Napišite Python skriptu koja ce ucitati tekstualnu datoteku naziva song.txt.
#Potrebno je napraviti rjecnik koji kao kljuceve koristi sve razlicite rijeci koje se pojavljuju u 
#datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijec (kljuc) pojavljuje u datoteci. 
#Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih.

file = open('Zadaci/song.txt')
words = []
dictionary = {}

for line in file:
    words_in_line = line.rstrip().split(' ')
    words += words_in_line

unique_words = list(set(words))
for unique_word in unique_words:
    count = 0
    for word in words:
        if word == unique_word:
            count += 1
    dictionary[unique_word] = count

one_appearance_words = []
for k,v in dictionary.items():
    if(v == 1):
        one_appearance_words.append(k)
print(f'Riječi: {one_appearance_words}')
print(f'Broj riječi: {len(one_appearance_words)}')