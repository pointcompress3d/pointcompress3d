import yaml
from deepdiff import DeepDiff
import sys

def yaml_as_dict(my_file):
    my_dict = {}
    with open(my_file, 'r') as fp:
        docs = yaml.safe_load_all(fp)
        for doc in docs:
            for key, value in doc.items():
                my_dict[key] = value
    return my_dict

if __name__ == '__main__':
    a = yaml_as_dict(sys.argv[1])
    b = yaml_as_dict(sys.argv[2])
    ddiff = DeepDiff(a, b, ignore_order=True)
    print(ddiff)
