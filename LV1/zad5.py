#Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ SMSSpamCollection.txt[1].
#Ova datoteka sadrži 5574 SMS poruka pri cemu su neke ozna ˇ cene kao ˇ spam, a neke kao ham.
#Primjer dijela datoteke:
#ham Yup next stop.
#ham Ok lar... Joking wif u oni...
#spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!

#a) Izracunajte koliki je prosje ˇ can broj rije ˇ ci u SMS porukama koje su tipa ham, a koliko je prosjecan broj rijeci u porukama koje su tipa spam. 
#b) Koliko SMS poruka koje su tipa spam završava usklicnikom ?

ham_counter = 0
ham_words_number = 0

spam_counter = 0
spam_words_number = 0
spam_exclamation_mark = 0

with open('Zadaci/SMSSpamCollection.txt', 'r', encoding='utf-8') as fhand:
    for line in fhand:
        line = line.strip()  

        if line.startswith('ham'):
            ham_counter += 1
            ham_words_number += len(line.split()[1:]) 
        
        elif line.startswith('spam'):
            spam_counter += 1
            spam_words_number += len(line.split()[1:])  

            if line.rstrip().endswith('!'): 
                spam_exclamation_mark += 1

if ham_counter > 0:
    print("Prosječan broj riječi u ham porukama:", round(ham_words_number / ham_counter, 2))
else:
    print("Nema ham poruka.")

if spam_counter > 0:
    print("Prosječan broj riječi u spam porukama:", round(spam_words_number / spam_counter, 2))
    print("Broj spam poruka koje završavaju uskličnikom:", spam_exclamation_mark)
else:
    print("Nema spam poruka.")
