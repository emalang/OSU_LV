#Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
#nekakvu ocjenu i nalazi se izmedu 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju ¯
#sljedecih uvjeta: ´
    #>= 0.9 A
    #>= 0.8 B
    #>= 0.7 C
    #>= 0.6 D
    #< 0.6 F
#Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe).
#Takoder, ako je broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovaraju ¯ cu poruku. 

try:
        grade = float(input("Unesite ocjenu (0.0 - 1.0): "))
        
        if grade < 0.0 or grade > 1.0:
            print("Greška: Ocjena mora biti između 0.0 i 1.0.")
        elif grade >= 0.9:
            print("A")
        elif grade >= 0.8:
            print("B")
        elif grade >= 0.7:
            print("C")
        elif grade >= 0.6:
            print("D")
        else:
            print("F")
except ValueError:
        print("Greška: Molimo unesite broj u ispravnom formatu.")
