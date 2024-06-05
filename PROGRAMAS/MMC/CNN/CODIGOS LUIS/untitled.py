import csv
import sys

def main(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row)
            
# python3 untitled.py patata.csv
# Donde patata.csv es el nombre del archivo
            
if __name__ == '__main__':
    main(sys.argv[1])