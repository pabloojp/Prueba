import sys
import os

def main():
    if len(sys.argv)<2:
        print(f"Usage: {sys.argv[0]} path")
        sys.exit(1)
    path = sys.argv[1]
    for d in os.listdir(path):
        print(f":{d}:")


if __name__ == '__main__':
    main()
